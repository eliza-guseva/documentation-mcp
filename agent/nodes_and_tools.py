from typing import List, Dict, Any
import json
import concurrent.futures
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config.config import ConfigManager

from vectorizing_and_retrieval.query_vector_graph import retrieve_from_group_with_graph
from config.utils import get_logger
from agent.agent_states import BaseAgentState, MultiQueryAgentState

config = ConfigManager('config.json').config
logger = get_logger(__name__)



def pydantic_ai_retrieval_tool(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results = retrieve_from_group_with_graph('pydantic_ai', query, k, config)
    formatted_docs = []
    for doc in results:
        formatted_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
        })
    return formatted_docs


def retrieval_node(state: BaseAgentState) -> BaseAgentState:
    """Directly execute retrieval without ToolExecutor middleware"""
    query = state["messages"][-1].content if state["messages"] else ""
    
    # Call your retrieval function directly
    retrieval_results = pydantic_ai_retrieval_tool(query)
    
    # Return updated state
    logger.info(f"Retrieval results in retrieval node: {len(retrieval_results)}")
    return {"retrieved_docs": retrieval_results}


def _doc_based_response_system_prompt():
    return """
    You are an expert PydanticAI documentation assistant.
    
    Your task is to:
    1. Thoroughly analyze ALL the retrieved documentation. Are they relevant to the query?
    2. Synthesize the information into a coherent, comprehensive answer. Ignore the irrelevant information.
    3. Make sure that the answer you provide *is relevant to the query*
    4. Provide concrete, runnable code examples that accurately reflect PydanticAI patterns
    5. Structure your response with clear headings, steps, and explanations
    6. Include relevant technical details and parameter explanations
    7. Cite specific documentation sources when providing information
    
    Your answers should be based solely on the retrieved documentation.
    If the information that is relevant to the query isn't in the retrieved documents, 
    acknowledge this limitation clearly.
    
    Be comprehensive and detailed when explaining complex PydanticAI concepts.
    Return the answer in markdown format.
    Add code blocks to your response when relevant.
    Add source links to your response.
    """


def response_node(state: BaseAgentState) -> BaseAgentState:
    """Generate response based on retrieved documents"""
    messages = state["messages"]
    retrieval_results = state["retrieved_docs"]
    logger.info(f"Retrieval results in response node: {len(retrieval_results)}")
    
    # Extract content from retrieved documents for context
    context = ""
    for doc in retrieval_results:
        context += f"Source: {doc['metadata'].get('source')}\n"
        context += f"{doc['content']}\n\n"
    
    # Create a prompt with the context
    llm = ChatOpenAI(model="gpt-4o")
    try:
        response = llm.invoke(
            messages + [
                SystemMessage(content=_doc_based_response_system_prompt()),
                AIMessage(content=f"I found the following information:\n\n{context}")
            ]
        )
    except Exception as e:
        logger.error(f"Failed to generate response from LLM. Error: {e}. Messages: {messages}. Context: {context}")
        response = AIMessage(content=f"I am sorry! Something went horribly wrong. Please try again later. Or maybe even better: fix me! :D")
    
    # Return the AI message
    logger.warning(f"Response node messages: {messages + [response]}")
    return {"messages": messages + [response]}


def _decomposition_prompt_template():
    return """
    You are an expert query analyzer. 
    Your task is to process a user's query and decide if it should be broken down into multiple sub-queries for a vector database, or if it should be rephrased for optimal search.
    The queries are used to search a vector database of PydanticAI documentation. 
    Each chunk in the database is either a code example or a section of the documentation.
    The goal is to find the most relevant chunks for the query.
    You MUST respond with a JSON object containing a single key "queries" whose value is a list of strings.
    - If splitting the query, provide the sub-queries in the list.
    Example for "how to stream structured output with agents": {{"queries": ["streaming structured output", "streaming in agents"]}}
    - If rephrasing or using the original query, provide a single query in the list.
    Example for "how to write a simple agent": {{"queries": ["tutorial for creating a basic PydanticAI agent"]}}
    - If the query is relatively simple and direct, generate 1 query (possibly rephrased for keywords).
    """
    
    
def decomposition_node(state: MultiQueryAgentState) -> Dict[str, Any]:
    user_query_message = state["messages"][-1]
    if not isinstance(user_query_message, HumanMessage):
        logger.error("Last message is not a HumanMessage, cannot decompose.")
        # Fallback: use empty to avoid breaking flow
        original_query = ""
        search_queries = []
    else:
        original_query = user_query_message.content
    
    logger.info(f"Original query for decomposition: \"{original_query}\"")

    if not original_query: # If original query is empty (e.g. from fallback above)
        return {"search_queries": [], "original_query": original_query}

    llm = ChatOpenAI(model="gpt-4o")
    messages_to_add = []
    
    try:
        response = llm.invoke(
            [
                SystemMessage(content=_decomposition_prompt_template()),
                HumanMessage(content=f"The user's query is: \"{original_query}\"")
            ],
            response_format={"type": "json_object"} # Request JSON output
        )
        query_data = json.loads(response.content)
        search_queries = query_data.get("queries", [])
        logger.warning(f"State messages: {state['messages']}")

        if not search_queries or not isinstance(search_queries, list) or not all(isinstance(q, str) for q in search_queries):
            logger.warning(f"LLM did not return a valid list of strings for queries. Response: {response.content}. Falling back to original query.")
            search_queries = [original_query]
            messages_to_add.append(AIMessage(content=f"Searching: '{original_query}'"))
        else:
            logger.info(f"Decomposed/rephrased queries: {search_queries}")
            for query in search_queries:
                messages_to_add.append(AIMessage(content=f"Searching: '{query}'"))
            
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.error(f"Failed to parse decomposed queries from LLM. Error: {e}. Content: '{getattr(response, 'content', 'N/A')}'. Falling back to original query.")
        search_queries = [original_query]
    except Exception as e:
        logger.error(f"An unexpected error occurred during query decomposition: {e}. Falling back to original query.")
        search_queries = [original_query]
        
    logger.warning(f"Messages to add: {messages_to_add}")
    logger.warning(f"State messages: {state['messages']}")
    return {
        "search_queries": search_queries, 
        "original_query": original_query,
        "messages": messages_to_add
    }


def parallel_retrieval_node(state: MultiQueryAgentState) -> Dict[str, Any]:
    queries = state.get("search_queries", [])
    original_query = state.get("original_query", "")
    
    if not queries or all(not q.strip() for q in queries):
        logger.warning(f"No valid search queries found for original query: '{original_query}'. Skipping retrieval.")
        return {"retrieved_docs": []}
        
    all_results: List[Dict[str, Any]] = []
    valid_queries = [q for q in queries if q.strip()]

    logger.info(f"Parallel retrieval for original query: '{original_query}'. Sub-queries: {valid_queries}")
    

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(valid_queries) or 1)) as executor:
        # Submit all retrieval tasks
        future_to_query = {executor.submit(pydantic_ai_retrieval_tool, query_text): query_text for query_text in valid_queries}
        
        for future in concurrent.futures.as_completed(future_to_query):
            query_text = future_to_query[future]
            try:
                results = future.result() # Get the results from the completed future
                all_results.extend(results)
                logger.info(f"Successfully retrieved {len(results)} docs for sub-query: '{query_text}'")
            except Exception as e:
                logger.error(f"Error during retrieval for sub-query '{query_text}' (Original: '{original_query}'): {e}")
                # Optionally, collect errors or decide how to handle partial failures

    # De-duplicate results based on content
    unique_results: List[Dict[str, Any]] = []
    seen_content = set()
    for doc in all_results:
        content = doc.get("content")
        if content and content not in seen_content: # Ensure content exists and is unique
            unique_results.append(doc)
            seen_content.add(content)
            
    logger.warning(f"All messages: {state['messages']}")
    logger.info(f"Aggregated {len(all_results)} raw results into {len(unique_results)} unique documents for original query: '{original_query}'.")
    return {"retrieved_docs": unique_results}

