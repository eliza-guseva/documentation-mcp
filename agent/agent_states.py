from typing import TypedDict, Annotated, List, Dict, Any
import operator

class BaseAgentState(TypedDict):
    """
    Base state for all agent states
    - messages: List of messages
    - tools_output: Dictionary of tools output
    - retrieved_docs: List of retrieved documents
    """
    messages: Annotated[List[Any], operator.add]
    tools_output: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    
    
class MultiQueryAgentState(BaseAgentState):
    """
    State for multi-query agent. 
    This agent decomposes (if needed) a query into multiple sub-queries 
    then it retrieves relevant documents for each sub-query.
    - original_query: To store the initial query for context if needed
    - search_queries: Decomposed/rephrased queries
    """
    original_query: str # To store the initial query for context if needed
    search_queries: List[str] # Decomposed/rephrased queries
    
    
    