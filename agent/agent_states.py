from typing import TypedDict, Annotated, List, Dict, Any
import operator

class BaseAgentState(TypedDict):
    """
    Base state for all agent states
    - messages: List of messages
    - retrieved_docs: List of retrieved documents
    """
    messages: Annotated[List[Any], operator.add]
    retrieved_docs: List[Dict[str, Any]]
    
    
class MultiQueryAgentState(BaseAgentState):
    """
    State for multi-query agent. 
    This agent decomposes (if needed) a query into multiple sub-queries 
    then it retrieves relevant documents for each sub-query.
    - original_query: To store the initial query for context if needed
    - search_queries: Decomposed/rephrased queries
    - decision_type: Type of decision made by the router
    """
    original_query: str # To store the initial query for context if needed
    search_queries: List[str] # Decomposed/rephrased queries
    decision_type: str # Type of decision made by the router
    
    
    