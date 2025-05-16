import re
import os
import copy
import json
import logging
import hashlib
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


logger = logging.getLogger(__name__)


def _is_not_line_number(content: str) -> bool:
    """
    Checks if a string contains only line numbers.
    Some code blocks are just line numbers, which we don't want to include in the chunks.
    """
    cleaned_content = content

    # 1. Remove potential code block fences and language identifier
    if cleaned_content.startswith("```"):
        first_line_end = cleaned_content.find('\n')
        if first_line_end != -1:
            cleaned_content = cleaned_content[first_line_end + 1:]
        else:
            cleaned_content = "" # Handle case like "```"

    if cleaned_content.endswith("```"):
        # Find the newline *before* the final ```
        # Use rfind on the string excluding the final ```
        if len(cleaned_content) > 3:
            last_line_start = cleaned_content[:-3].rfind('\n')
            if last_line_start != -1:
                 # Take content up to that newline
                 cleaned_content = cleaned_content[:last_line_start]
            else:
                 # Handle case like "content```" without preceding newline
                 cleaned_content = cleaned_content[:-3]
        else:
            # Handle cases like "```" or "" after initial cleaning
             cleaned_content = ""


    cleaned_content = cleaned_content.replace("```", "").strip().replace('\\n', '\n')

    if not cleaned_content.strip():
        return False

    # The regex `^[\d\s]+$` matches strings containing one or more digits or whitespace chars.
    if re.fullmatch(r'^[\d\s]+$', cleaned_content):
        return False

    return True


def is_chunk_meaningful(content: str, metadata: dict) -> bool:
    """
    Checks if a chunk contains meaningful content beyond just numbers,
    whitespace, literal '\\n', and code block fences.
    """
    if len(content) < 30:
        return False
    if not _is_not_line_number(content):
        return False
    return True


def split_text_and_code(doc_content: str) -> List[Dict[str, Any]]:
    """
    Split a document into alternating text and code parts in the order they appear.
    
    Args:
        doc_content: The document content to split
        
    Returns:
        A list of dictionaries, each with 'content' and 'type' keys.
        Type is either 'text' or 'code'.
    """
    all_parts = []
    
    # Split by triple backticks
    parts = doc_content.split('```')
    
    # First part is always text (might be empty)
    if parts[0]:
        text_chunk = {
            "content": parts[0],
            "type": "text",
        }
        all_parts.append(text_chunk)
    
    # Process remaining parts (alternating code and text)
    for i in range(1, len(parts)):
        if i % 2 == 1:  # Odd indices are code blocks
            code_content = '\n'.join(parts[i].replace("\\n", "\n").split("\n")[1:]).strip()
            # Create a code chunk
            code_chunk = {
                "content": f"```\n{code_content}\n```",  # Re-add backticks for markdown formatting
                "language": "python", 
                "symbols": [],
                "type": "code"
            }
            all_parts.append(code_chunk)
        else:  # Even indices are text
            if parts[i].strip():  # Only add if there's actual content
                text_chunk = {
                    "content": parts[i],
                    "type": "text",
                }
                all_parts.append(text_chunk)
    
    return all_parts


def merge_consecutive_text_chunks(all_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges consecutive text chunks into a single chunk. Pydantic AI docs are code heavy,
    so we want to merge text chunks that are part of the same section.
    """
    merged_parts = []
    merged_parts.append(all_parts[0])
    for part in all_parts[1:]:
        if part["type"] == "text" and merged_parts[-1]["type"] == "text":
            merged_parts[-1]["content"] += part["content"]
        elif part["type"] == "code" and len(part["content"]) < 100 and merged_parts[-1]["type"] == "text":
            merged_parts[-1]["content"] += part["content"]
        elif part["type"] == "code" and not is_chunk_meaningful(part["content"].replace("python", ""), {}):
            continue
        else:
            merged_parts.append(part)
    return merged_parts


def chunk_document(
    doc_content: str, 
    source_document_id: str = "unknown", 
    original_metadata: dict = None
    ) -> List[Document]:
    """
    Hybrid chunking approach that:
    1. Extracts code blocks as separate chunks
    2. Splits remaining text using LangChain's TokenTextSplitter
    3. Extracts inline code as symbols
    4. Calculates and adds a content hash to each chunk's metadata.
    
    Args:
        doc_content: The string content of the document
        source_document_id: An identifier for the source document
        original_metadata: Dictionary containing the original metadata
        
    Returns:
        A list of Langchain Document objects, each with a 'content_hash' in metadata.
    """ 
    # Default metadata for all chunks
    if original_metadata is None:
        original_metadata = {}
    
    # Make a copy to avoid modifying the input
    metadata = original_metadata.copy()
    
    # Ensure source is in the metadata
    if "source" not in metadata and source_document_id != "unknown":
        metadata["source"] = source_document_id
    
    # Initialize list for final chunks
    all_parts = split_text_and_code(doc_content)
    all_parts = merge_consecutive_text_chunks(all_parts)
    all_chunks = []
    
    # Process text and code parts
    for (position, part) in enumerate(all_parts):
        part_content = part["content"]
        part_metadata = metadata.copy()
        
        # Calculate content hash (consistent regardless of type)
        content_hash = hashlib.sha256(part_content.encode('utf-8')).hexdigest()
        part_metadata["content_hash"] = content_hash
        part_metadata["position"] = position

        if part["type"] == "code":
            part_metadata["content_type"] = "code"
            part_metadata["language"] = part["language"]
            part_metadata["symbols"] = part["symbols"]
            
            document = Document(
                page_content=part_content,
                metadata=part_metadata
            )
            if is_chunk_meaningful(document.page_content, part_metadata):
                all_chunks.append(document)
                
        elif part["type"] == "text":
            part_metadata["content_type"] = "text"
            part_metadata["content_hash"] = hashlib.sha256(part_content.encode('utf-8')).hexdigest() 
            
            document = Document(
                page_content=part_content,
                metadata=part_metadata
            )
            if is_chunk_meaningful(document.page_content, part_metadata):
                all_chunks.append(document)

    logger.debug(f"Generated {len(all_chunks)} chunks for source: {source_document_id}")
    return all_chunks


def chunk_json_file(file_path: str) -> List[Document]:
    """
    Process a single JSON file, extracting content and metadata,
    and chunking it into Document objects.
    
    Args:
        file_path: Path to the JSON file to process.
        
    Returns:
        A list of Langchain Document objects generated from the file.
        Returns empty list if the file couldn't be processed.
    """
    try:
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract page content and metadata
        page_content = data.get('page_content', '')
        if not page_content:
            logger.warning(f"No page_content found in {file_path}, skipping file.")
            return []
            
        metadata = data.get('metadata', {})
        
        # Use the source from metadata or file path as fallback
        source_id = metadata.get('source', file_path)
        
        # Chunk the document
        document_chunks = chunk_document(
            doc_content=page_content,
            source_document_id=source_id,
            original_metadata=metadata
        )
        
        return document_chunks
        
    except json.JSONDecodeError:
        logger.error(f"Could not parse JSON in {file_path}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return []


def chunk_markdown_file(file_path: str) -> List[Document]:
    """
    Process a single markdown file, extracting content and metadata,
    and chunking it into Document objects.
    """
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]

    # Initialize the splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

# Split the text
    
    json_data = json.load(open(file_path, 'r', encoding='utf-8'))
    text = json_data['page_content'].replace("\\n", "\n")
    metadata = json_data['metadata']
    splits = markdown_splitter.split_text(text)
    for (position, chunk) in enumerate(splits):
        logger.debug(f"Chunk {position} of {len(splits)}")
        metadata['position'] = position
        metadata['content_type'] = 'text'
        metadata['content_hash'] = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        metadata['source'] = 'https://ai.pydantic.dev/llms-full.txt'
        chunk.metadata = copy.deepcopy(metadata)
    return splits


def process_directory(directory_path: str) -> List[Document]:
    """
    Recursively processes all JSON files in the given directory and its subdirectories.
    Extracts content from each JSON, chunks it, and returns all chunks as Document objects.
    
    Args:
        directory_path: Path to the directory containing JSON files.
        
    Returns:
        A list of Langchain Document objects generated from all JSON files.
    """
    all_documents = []
    
    logger.info(f"Starting processing for directory: {directory_path}")
    # Walk through all files and subdirectories
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Process only JSON files
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                # Process the file and add its chunks to our collection
                if file == "llms-fulltxt.json":
                    logger.info(f"Chunking markdown file: {file_path}")
                    file_documents = chunk_markdown_file(file_path)
                else:
                    file_documents = chunk_json_file(file_path)
                all_documents.extend(file_documents)
    
    logger.info(f"Finished processing {directory_path}. Total documents generated: {len(all_documents)}")
    return all_documents


# Add the new compare_chunks function
def compare_chunks(chunk1: Document, chunk2: Document) -> bool:
    """
    Compares two Document chunks based on their source and content hash.

    Args:
        chunk1: The first Document object.
        chunk2: The second Document object.

    Returns:
        True if both source and content_hash metadata match, False otherwise.
    """
    if not isinstance(chunk1, Document) or not isinstance(chunk2, Document):
        return False
        
    source1 = chunk1.metadata.get('source')
    hash1 = chunk1.metadata.get('content_hash')
    
    source2 = chunk2.metadata.get('source')
    hash2 = chunk2.metadata.get('content_hash')
    
    # Ensure both hash and source are present and match
    return (source1 is not None and
            hash1 is not None and
            source1 == source2 and 
            hash1 == hash2)
