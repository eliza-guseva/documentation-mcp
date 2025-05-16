import os
from pathlib import Path
from vectorizing_and_retrieval.chunking import chunk_json_file, chunk_markdown_file
from data_collection.fetcher import url_to_source_identifier
import logging
from langchain_community.vectorstores import FAISS
from typing import Tuple
from langchain_community.vectorstores.faiss import DistanceStrategy

logger = logging.getLogger(__name__)


def get_local_storage_path(config):
    return Path(__file__).parent.parent / config['local_storage_path']

def get_raw_docs_path(config):
    return get_local_storage_path(config) / 'documentation_raw'

def get_vector_store_path(config):
    return get_local_storage_path(config) / 'vector_store'

def get_vector_store_path_for_url(config, url):
    return get_vector_store_path(config) / url_to_source_identifier(url).replace('.', '_')


def _initialize_embedding_function(config: dict):
    """
    Initialize the embedding function based on the configuration.
    Returns None if the embedding function cannot be initialized.
    Raises an exception if there is an error initializing the embedding function.
    
    Args:
        config: A dictionary containing the configuration for the embedding function.
        
    Returns:
        An instance of the embedding function or None if it cannot be initialized.
    """
    try:
        if config.get('embed_with_openai') and config.get('openai_api_key'):
            from langchain_openai import OpenAIEmbeddings
            openai_embedding_model = config.get('openai_embedding_model', "text-embedding-3-small")
            logger.info(f"Initializing OpenAI Embeddings (Model: {openai_embedding_model})")
            if 'OPENAI_API_KEY' not in os.environ and config.get('openai_api_key'):
                os.environ['OPENAI_API_KEY'] = config['openai_api_key']
            return OpenAIEmbeddings(model=openai_embedding_model)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            hf_embedding_model = config.get('huggingface_embedding_model', "sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"Initializing HuggingFace Embeddings (Model: {hf_embedding_model})")
            model_kwargs = {'device': config.get('embedding_device', 'cpu')}
            encode_kwargs = {'normalize_embeddings': True}
            return HuggingFaceEmbeddings(
                model_name=hf_embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    except ImportError as e:
        logger.error(f"Missing LangChain integration package: {e}. Please install langchain-openai or langchain-huggingface.")
        raise ImportError(f"Missing LangChain integration package: {e}. Please install langchain-openai or langchain-huggingface.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        raise Exception(f"Failed to initialize embedding model: {e}")


def load_existing_vector_store(index_path: Path, config: dict) -> FAISS | None:
    """
    Load an existing vector store from a given index path.
    Returns None if the vector store cannot be loaded.
    Raises an exception if there is an error loading the vector store.
    
    Args:
        index_path: The path to the index file.
        config: A dictionary containing the configuration for the vector store.
    """
    embeddings = _initialize_embedding_function(config)
    if embeddings is None:
        logger.error(f"Embedding function could not be initialized for {index_path}. Aborting vectorization.")
        return None
    if index_path.exists() and index_path.is_dir():
        try:
            logger.info(f"Attempting to load existing vector store from: {index_path}")
            vector_store = FAISS.load_local(
                folder_path=index_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except FileNotFoundError:
            logger.info(f"No index files found within {index_path}, though directory exists. Treating as new.")
            vector_store = None
        except Exception as e:
            logger.error(f"Error loading vector store from {index_path}: {e}. Will attempt to recreate.", exc_info=True)
            vector_store = None
    return None


def _get_existing_hashes(all_docs: list, source: str) -> set:
    """
    Get the set of content hashes for documents with a given source.
    Returns a set of content hashes.
    
    Args:
        all_docs: A list of documents.
        source: The source to filter by.
        
    Returns:
        A set of content hashes.
    """
    existing_hashes = set()
    for doc in all_docs:
        if doc.metadata.get('source') == source:
            content_hash = doc.metadata.get('content_hash')
            if content_hash: # Ensure content_hash exists
                existing_hashes.add(content_hash)
    return existing_hashes


def _delete_documents_for_source(vector_store: FAISS, source: str, index_path: Path) -> Tuple[int, bool]:
    """
    Deletes documents for a given source and returns the count and if recreation is needed.
    
    Args:
        vector_store: The vector store to delete documents from.
        source: The source to delete documents for.
        index_path: The path to the index file.
    """
    deleted_count = 0
    try:
        if hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
            ids_to_delete = [
                doc_id for doc_id, doc in vector_store.docstore._dict.items()
                if doc.metadata.get('source') == source
            ]
            deleted_count = len(ids_to_delete)
            if ids_to_delete:
                logger.info(f"Attempting to delete {deleted_count} documents for source: {source}")
                vector_store.delete(ids_to_delete) # May raise NotImplementedError
                logger.info("Deletion successful. Saving updated vector store.")
                vector_store.save_local(index_path) # Save after deletion
            return deleted_count, False # Return count and False (no recreation needed)
        else:
            logger.warning("Cannot access docstore for deletion. Forcing recreation.")
            return 0, True # Return 0 deleted, True for recreation needed
    except Exception as e:
        logger.error(f"Error during deletion phase {e}. Forcing recreation.", exc_info=True)
        return 0, True # Return 0 deleted, True for recreation needed


def _update_existing_vector_store(
    all_docs: list,
    file_documents: list,
    source: str,
    file_path: str,
    index_path: Path,
    vector_store: FAISS,
    file_hashes: set
    ) -> Tuple[bool, int]:
    """
    Updates the vector store for a source and returns if updated and count deleted.
    Recreating vector stores from scratch is expensive, so we try to avoid it, 
    by checking if the source has changed.
    
    Args:
        all_docs: A list of documents.
        file_documents: A list of documents to add to the vector store.
        source: The source to update the vector store for.
        file_path: The path to the file.
    """
    # Get the set of content hashes for documents with a given source
    existing_hashes = _get_existing_hashes(all_docs, source)
    deleted_count = 0
    updated = False
    # If the set of content hashes has changed, we need to update the vector store
    if file_hashes != existing_hashes:
        logger.info(f"Changes detected for source '{source}' (File: {file_path}). Updating vector store.")
        # Delete existing documents first
        deleted_count, needs_recreation = _delete_documents_for_source(vector_store, source, index_path)
        if needs_recreation:
             # Handle case where deletion failed and recreation is forced
             # This logic might need refinement. We'll do just warning for now.
             logger.warning(f"Recreation forced for {source}, skipping add for now.")
             return False, deleted_count # Indicate no update happened in *this* step

        # Add new documents
        vector_store.add_documents(file_documents)
        updated = True
        logger.info(f"Added {len(file_documents)} new documents for source '{source}'.")
        vector_store.save_local(index_path)
        logger.info(f"Saved updated vector store to {index_path} after processing {file_path}")
    else:
        logger.debug(f"No changes detected for source '{source}' (File: {file_path}). Skipping update.")

    return updated, deleted_count # Return whether an update happened and how many were deleted


def _create_new_vector_store(file_documents: list, index_path: Path, embeddings, file_path: str) -> None:
    """
    Creates a new vector store from a list of documents.
    
    Args:
        file_documents: A list of documents to add to the vector store.
        index_path: The path to the index file.
        embeddings: The embedding function to use.
        file_path: The path to the file.
    """
    logger.info(f"Creating new vector store with first file: {file_path}")
    vector_store = FAISS.from_documents(file_documents, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    vector_store.save_local(index_path)
    logger.info(f"Saved initial vector store to {index_path}")


def _process_file(
    file_path: str,
    root: str,
    ) -> Tuple[str, list]:
    """
    Chunks a JSON file and returns the source and list of documents.
    """
    full_path = os.path.join(root, file_path)
    if file_path == "llms-fulltxt.json":
        logger.info(f"Processing markdown file: {file_path}")
        file_documents = chunk_markdown_file(full_path)
    else:
        # Process the file and add its chunks to our collection
        file_documents = chunk_json_file(full_path)
    if not file_documents:
        logger.warning(f"No documents generated from file: {file_path}. Skipping.")
        return None, None

    logger.info(f"Generated {len(file_documents)} chunks for file: {file_path}")
    # Use .get() for safer access to metadata
    source = file_documents[0].metadata.get('source')
    return source, file_documents
        
        
def update_vector_db_for_url(url: str, config: dict) -> Tuple[int, int]:
    """
    Updates the vector database for a specific URL (meaning a specific documentation website)

    Returns:
        Tuple[int, int]: A tuple containing (updated_chunk_count, deleted_chunk_count).
                         'updated' refers to chunks that were replaced (deleted then added).
                         'deleted' refers to chunks from stale sources that were only deleted.
    """
    logger.info(f"Starting vector DB update for URL: {url}")
    index_path = get_vector_store_path_for_url(config, url)
    logger.info(f"Vector store index path: {index_path}")
    vector_store = load_existing_vector_store(index_path, config)
    embeddings = _initialize_embedding_function(config)

    updated_chunk_count = 0
    deleted_chunk_count = 0 # Counts only stale source deletions
    all_docs = []
    existing_sources = set()

    if vector_store:
        logger.info(f"Loaded existing vector store from {index_path}")
        # Correct way to get all documents
        all_docs = list(vector_store.docstore._dict.values()) if hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict') else []
        existing_sources = set([doc.metadata['source'] for doc in all_docs if 'source' in doc.metadata]) # Safe check for 'source'
    else:
        logger.info(f"No existing vector store found at {index_path}, will create a new one.")

    raw_doc_path = get_raw_docs_path(config)
    source_identifier = url_to_source_identifier(url)
    source_path = raw_doc_path / source_identifier
    logger.info(f"Processing files in source directory: {source_path}")
    processed_sources = set() # Keep track of sources found in current run

    for root, _, files in os.walk(source_path):
        for file_path in files:
            # Process only JSON files
            if file_path.endswith('.json'):
                source, file_documents = _process_file(file_path, root)
                if not source:
                    logger.error(f"Chunk from {file_path} is missing 'source' metadata. Skipping.")
                    continue

                processed_sources.add(source) # Add source found in this run
                file_hashes = set([doc.metadata.get('content_hash') for doc in file_documents if doc.metadata.get('content_hash')])

                if vector_store:
                    was_updated, _ = _update_existing_vector_store(
                        all_docs, file_documents, source, file_path, index_path, vector_store, file_hashes
                    )
                    if was_updated:
                        updated_chunk_count += len(file_documents) # Count added chunks as 'updated'

                    # Refresh all_docs if an update happened (as docstore might change)
                    if was_updated and hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
                         all_docs = list(vector_store.docstore._dict.values())

                else:
                    _create_new_vector_store(file_documents, index_path, embeddings, file_path)
                    # Count initial creation as updates
                    updated_chunk_count += len(file_documents)
                    # Need to reload vector_store after creation
                    vector_store = load_existing_vector_store(index_path, config)
                    if vector_store and hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
                        all_docs = list(vector_store.docstore._dict.values())
                        existing_sources = set([doc.metadata['source'] for doc in all_docs if 'source' in doc.metadata]) # Update existing sources
                    elif not vector_store:
                        logger.error(f"Failed to load vector store immediately after creation for {index_path}. Subsequent operations might fail.")
                        # Decide how to handle this - maybe return counts so far or raise?
                        return updated_chunk_count, deleted_chunk_count # Return counts up to failure


    logger.info(f"Finished processing files for URL: {url}. Found sources: {list(processed_sources)}")

    # Delete documents from sources that existed before but weren't found in this run (stale sources)
    if vector_store: # Ensure vector_store exists
        stale_sources = existing_sources - processed_sources
        if stale_sources:
            logger.info(f"Found {len(stale_sources)} stale sources to delete: {list(stale_sources)}")
            for source in stale_sources:
                # Pass index_path here as well
                deleted_for_source, _ = _delete_documents_for_source(vector_store, source, index_path)
                deleted_chunk_count += deleted_for_source # Add to the deleted count
        else:
            logger.info("No stale sources found to delete.")
    else:
        logger.warning("No vector store available at the end of processing. Cannot delete stale sources.")

    logger.info(f"Update complete for {url}. Updated chunks: {updated_chunk_count}, Deleted stale chunks: {deleted_chunk_count}")
    return updated_chunk_count, deleted_chunk_count


def update_vector_db_for_all_urls(config: dict) -> None:
    """
    Update the vector database for all URLs in the configuration.
    """
    # Flatten the list of URLs from the dictionary values
    all_urls = [url for url_list in config['documentation_urls'].values() for url in url_list]
    unique_urls = list(set(all_urls)) # Remove duplicates
    logger.info(f"Processing {len(unique_urls)} unique documentation URLs from config.")

    total_updated_chunks = 0
    total_deleted_chunks = 0

    for url in unique_urls:
        updated, deleted = update_vector_db_for_url(url, config)
        total_updated_chunks += updated
        total_deleted_chunks += deleted

    logger.info("Finished processing all URLs.")
    logger.info(f"Total chunks updated (added/replaced): {total_updated_chunks}")
    logger.info(f"Total stale chunks deleted: {total_deleted_chunks}")















