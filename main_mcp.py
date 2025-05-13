# main_mcp.py - MCP Server using the official MCP SDK

from typing import Dict, List, Any

# Import the core MCP libraries
from mcp.server.fastmcp import FastMCP

# Import our retrieval functions
from vectorizing_and_retrieval.query_vector_graph import (
    retrieve_from_group,
    retrieve_from_group_with_graph
)
from config.utils import get_logger, setup_logging
from config import ConfigManager

# --- Configuration ---
CONFIG_FILE = 'config.json'
DEFAULT_LOG_FILE = 'mcp_main.log'  # Fallback log file


# Initialize the MCP server
mcp = FastMCP("langchain-pydantic")
logger = get_logger(__name__)

# Load the configuration
config = ConfigManager(CONFIG_FILE).config


@mcp.tool()
async def pydantic_ai_vector_retrieval(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve documents from pydantic-ai documentation using vector similarity.
    
    Args:
        query: The search query
        k: Number of results to return (default: 10)
    """
    logger.info(f"Running pydantic-ai vector retrieval for query: {query}")
    
    results = retrieve_from_group("pydantic_ai", query, k, config)
    
    # Convert to a serializable format for MCP
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)  # Ensure score is a float for JSON serialization
        } 
        for doc, score in results
    ]

@mcp.tool()
async def pydantic_ai_graph_retrieval(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve documents from pydantic-ai documentation using graph-augmented retrieval.
    
    Args:
        query: The search query
        k: Number of results to return (default: 10)
    """
    logger.info(f"Running pydantic-ai graph-augmented retrieval for query: {query}")
    
    results = retrieve_from_group_with_graph("pydantic_ai", query, k, config)
    
    # Convert to a serializable format for MCP
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } 
        for doc in results
    ]

if __name__ == "__main__":
    # Configure logging
    setup_logging()
    logger.info("Starting MCP server...")
    
    # Run the MCP server
    mcp.run(transport='stdio')  # stdio transport for Claude Desktop integration 