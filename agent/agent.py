from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph
import operator
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config.config import ConfigManager
from langchain.agents import AgentExecutor

from vectorizing_and_retrieval.query_vector_graph import retrieve_from_group_with_graph
from config.utils import get_logger

config = ConfigManager('config.json').config
logger = get_logger(__name__)

class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    tools_output: Dict[str, Any]

def pydantic_ai_retrieval_tool(query: str, k: int = 5) -> str:
    results = retrieve_from_group_with_graph('pydantic_ai', query, k, config)
    formatted_docs = []
    for doc in results:
        formatted_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
        })
    return formatted_docs


def retrieval_node(state: AgentState) -> AgentState:
    """Directly execute retrieval without ToolExecutor middleware"""
    query = state["messages"][-1].content if state["messages"] else ""
    
    # Call your retrieval function directly
    retrieval_results = pydantic_ai_retrieval_tool(query)
    
    # Return updated state
    logger.info(f"Retrieval results in retrieval node: {len(retrieval_results)}")
    return {"tools_output": {"retrieval": retrieval_results}}


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


def response_node(state: AgentState) -> AgentState:
    """Generate response based on retrieved documents"""
    messages = state["messages"]
    retrieval_results = state["tools_output"].get("retrieval")
    logger.info(f"Retrieval results in response node: {len(retrieval_results)}")
    
    # Extract content from retrieved documents for context
    context = ""
    for doc in retrieval_results:
        context += f"Source: {doc['metadata'].get('source')}\n"
        context += f"{doc['content']}\n\n"
    
    # Create a prompt with the context
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke(
        messages + [
            SystemMessage(content=_doc_based_response_system_prompt()),
            AIMessage(content=f"I found the following information:\n\n{context}")
        ]
    )
    
    # Return the AI message
    return {"messages": messages + [response]}


def create_agent_executor() -> AgentExecutor:
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("response", response_node)
    workflow.add_edge("retrieval", "response")
    workflow.add_edge("response", "__end__")
    workflow.set_entry_point("retrieval")
    agent_executor = workflow.compile()
    return agent_executor

def stream_agent_output(agent_executor: AgentExecutor, query: str):
    """Stream the agent's response"""
    for chunk, metadata in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="messages"
    ):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content