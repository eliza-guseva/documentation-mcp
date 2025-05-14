from typing import List, Dict, Any
import json
import concurrent.futures
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config.config import ConfigManager

from vectorizing_and_retrieval.query_vector_graph import retrieve_from_group_with_graph, retrieve_from_group
from config.utils import get_logger
from agent.agent_states import BaseAgentState, MultiQueryAgentState

config = ConfigManager('config.json').config
logger = get_logger(__name__)



def pydantic_ai_retrieval_tool(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results = retrieve_from_group('pydantic_ai', query, k, config)
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
    return {"messages": messages + [response]}


def _router_prompt_template():
    return """
    You are an expert query analyzer and conversational assistant.
    Your task is to process a user's query in the context of the recent conversation history and decide on the best course of action.
    The ultimate goal is to answer questions about PydanticAI documentation. 
    Each chunk in the documentation database is either a code example or a section of the documentation.

    You will be given the user's current query and the last few turns of the conversation history.

    Analyze the query and history according to these steps:

    1.  **Follow-up Detection:**
        *   Is the current user query a direct follow-up to the immediately preceding AI's response or user's question in the provided history?
        *   Can the query be fully and accurately answered using ONLY the information present in the provided conversation history (specifically, the last AI response)?
        *   If YES, your decision should be "DIRECT_ANSWER".

    2.  **Bad Query Detection (if not a direct answer from history):**
        *   Is the query too vague, ambiguous, nonsensical, or clearly outside the scope of PydanticAI documentation (e.g., asking about unrelated topics)?
        *   Be lenient. Err on the side of DECOMPOSE and not CLARIFY.
        *   Is the query too short or lacks sufficient detail to perform a meaningful search?
        *   If YES, your decision should be "CLARIFY".

    3.  **Decomposition/Rephrasing for Search (if not a direct answer and not a bad query):**
        *   The query is likely a new, valid question that requires searching the PydanticAI documentation.
        *   Rephrase or break down the query into one or more sub-queries optimized for a vector database search.
        *   The goal is to find the most relevant chunks for the query.
        *   If the query is relatively simple and direct, generate 1 query (possibly rephrased for keywords).
        *   Your decision should be "DECOMPOSE".

    You MUST respond with a JSON object adhering to the following structure:

    {{
        "decision_type": "DIRECT_ANSWER" | "CLARIFY" | "DECOMPOSE",
        "payload": {{
            "direct_answer_content": "...", // Present if decision_type is DIRECT_ANSWER. This is the complete answer to the user's query.
            "clarification_question": "...", // Present if decision_type is CLARIFY. This is the question to ask the user for more details.
            "search_queries": ["...", "..."]     // Present if decision_type is DECOMPOSE. List of queries for the vector database.
        }},
        "original_query": "..." // Always include the original user query you processed.
    }}

    Examples:

    Input History:
    User: "How do I use Pydantic Models with PydanticAI?"
    AI: "PydanticAI is designed to work seamlessly with Pydantic Models. You define your data structure as a Pydantic Model, and PydanticAI can then parse and validate LLM outputs against this model. For instance, `class User(BaseModel): name: str; age: int`..."
    Current User Query: "What about nested models?"

    Output JSON:
    {{
        "decision_type": "DIRECT_ANSWER",
        "payload": {{
            "direct_answer_content": "Yes, PydanticAI supports nested Pydantic Models just like Pydantic itself. You can define a Pydantic Model that includes fields typed with other Pydantic Models, and PydanticAI will handle the parsing and validation recursively."
        }},
        "original_query": "What about nested models?"
    }}


    Input History: (empty or unrelated)
    Current User Query: "invoke"

    Output JSON:
    {{
        "decision_type": "CLARIFY",
        "payload": {{
            "clarification_question": "Could you please be more specific about what you mean by 'invoke' in the context of PydanticAI? Are you referring to invoking LLMs, specific functions, or something else?"
        }},
        "original_query": "invoke."
    }}


    Input History: (empty or unrelated)
    Current User Query: "how to stream structured output with agents and also how to define custom extraction logic"

    Output JSON:
    {{
        "decision_type": "DECOMPOSE",
        "payload": {{
            "search_queries": ["streaming structured output with PydanticAI agents", "custom data extraction logic PydanticAI"]
        }},
        "original_query": "how to stream structured output with agents and also how to define custom extraction logic"
    }}

    Consider the provided conversation history carefully when making your decision.
    """
    
    
def router_node(state: MultiQueryAgentState) -> Dict[str, Any]:
    history = state["messages"][-min(len(state["messages"]), 6):]
    user_query_message = state["messages"][-1]
    messages_to_add = history
    if not isinstance(user_query_message, HumanMessage):
        logger.error("Last message is not a HumanMessage, cannot decompose.")
        return {
            "decision_type": "CLARIFY",
            "payload": {
                "clarification_question": "I am sorry! Something went horribly wrong. Please try again later. Or maybe even better: fix me! :D"
            },
            "original_query": ""
        }
    else:
        original_query = user_query_message.content
    
    logger.info(f"Original query for decomposition: \"{original_query}\"")


    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    try:
        response = llm.invoke(
            [
                SystemMessage(content=_router_prompt_template()),
                *history,
                HumanMessage(content=f"The user's query is: \"{original_query}\"")
            ],
            response_format={"type": "json_object"} # Request JSON output
        )
        query_data = json.loads(response.content)
        search_queries = []
        
        if query_data["decision_type"] == "DIRECT_ANSWER":
            messages_to_add.append(AIMessage(content=query_data["payload"]["direct_answer_content"]))
        elif query_data["decision_type"] == "CLARIFY":
            messages_to_add.append(AIMessage(content=query_data["payload"]["clarification_question"]))
        elif query_data["decision_type"] == "DECOMPOSE":
            search_queries = query_data["payload"]["search_queries"]
            for query in search_queries:
                messages_to_add.append(AIMessage(content=f"Searching: '{query}'"))
        else:
            raise ValueError(f"Invalid decision type: {query_data['decision_type']}")
        
            
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.error(f"Failed to parse decomposed queries from LLM. Error: {e}. Content: '{getattr(response, 'content', 'N/A')}'. Falling back to original query.")
        search_queries = [original_query]
        return {
            "messages": messages_to_add,
            "original_query": original_query,
            "search_queries": search_queries,
            "decision_type": "DECOMPOSE"
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred during query decomposition: {e}. Falling back to original query.")
        search_queries = [original_query]
        return {
            "messages": messages_to_add,
            "original_query": original_query,
            "search_queries": search_queries,
            "decision_type": "DECOMPOSE"
        }
        

    return {
        "messages": messages_to_add,
        "original_query": original_query,
        "search_queries": search_queries,
        "decision_type": query_data["decision_type"]
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
            
    logger.info(f"Aggregated {len(all_results)} raw results into {len(unique_results)} unique documents for original query: '{original_query}'.")
    return {"retrieved_docs": unique_results}


def return_output_node(state: MultiQueryAgentState) -> MultiQueryAgentState:
    return {"messages": state["messages"]}