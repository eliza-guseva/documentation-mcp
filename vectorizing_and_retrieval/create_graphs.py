from  typing import List
import json
from pathlib import Path
from langchain_core.documents import Document
from logging import getLogger
from collections import defaultdict, namedtuple
import networkx as nx
import math
from vectorizing_and_retrieval.create_vectors import (
    get_raw_docs_path, 
    get_local_storage_path,
    url_to_source_identifier, 
    get_vector_store_path_for_url,
    load_existing_vector_store,
)


BASE_WEIGHT = 1
DECAY_RATE = 0.2
logger = getLogger(__name__)
Chunk = namedtuple("Chunk", ["id", "position"])


def get_graph_path(url: str, config: dict) -> Path:
    return get_local_storage_path(config) / "graphs" / f"{url_to_source_identifier(url).replace('.', '_')}.json"


def crate_graph_for_url(url: str, config: dict) -> List[Document]:
    graph = nx.DiGraph()
    source_path = get_raw_docs_path(config) / url_to_source_identifier(url)
    index_path = get_vector_store_path_for_url(config, url)
    
    logger.info(f"Processing files in source directory: {source_path}")
    
    all_chunks_by_source = defaultdict(list)
    vector_store = load_existing_vector_store(index_path, config)
    for (doc_id, doc) in vector_store.docstore._dict.items():
        all_chunks_by_source[doc.metadata["source"]].append(Chunk(doc_id, doc.metadata["position"]))
        graph.add_node(doc_id, metadata=doc.metadata)
        
    for source, chunks in all_chunks_by_source.items():
        all_chunks_by_source[source] = sorted(chunks, key=lambda x: x[1])
        
        for i in range(len(all_chunks_by_source[source])):
            position_i = all_chunks_by_source[source][i].position
            for j in range(len(all_chunks_by_source[source])):
                if i == j:
                    continue
                position_j = all_chunks_by_source[source][j].position
                distance = abs(position_j - position_i)
                weight = BASE_WEIGHT * math.exp( - distance * DECAY_RATE)
                graph.add_edge(
                    all_chunks_by_source[source][i].id,
                    all_chunks_by_source[source][j].id,
                    weight=weight,
                    direction = "NEXT" if position_j > position_i else "PREVIOUS",
                    relationship_type = "SEQUENTIAL",
                    distance = distance
                )
    edge_list = list(graph.edges(data=True))
    print(f"Total edges created: {len(edge_list)}")
                
                
    # After all edges are added, force a full edge list evaluation
    edge_list = list(graph.edges(data=True))
    print(f"Total edges created: {len(edge_list)}")

    # serialize the graph
    graph_path = get_graph_path(url, config)
    logger.info(f"Serializing graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {graph_path}")
    # make sure the directory exists
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "w") as f:
        json.dump(nx.node_link_data(graph), f)

    return

def create_graph_for_all_urls(config: dict):
    # Flatten the list of URLs from the dictionary values
    all_urls = [url for url_list in config['documentation_urls'].values() for url in url_list]
    unique_urls = list(set(all_urls)) # Remove duplicates
    logger.info(f"Processing {len(unique_urls)} unique documentation URLs for graph creation.")

    for url in unique_urls:
        crate_graph_for_url(url, config)
        
        
def load_graph_for_url(url: str, config: dict):
    graph_path = get_graph_path(url, config)
    with open(graph_path, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data)

                
                
            
            
            
                
                
                
                