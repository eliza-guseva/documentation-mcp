from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor

from config.utils import get_logger
from agent.agent_states import BaseAgentState, MultiQueryAgentState
from agent.nodes_and_tools import retrieval_node, response_node, decomposition_node, parallel_retrieval_node


logger = get_logger(__name__)


def create_one_action_agent_executor() -> AgentExecutor:
    workflow = StateGraph(BaseAgentState)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("response", response_node)
    workflow.add_edge("retrieval", "response")
    workflow.add_edge("response", "__end__")
    workflow.set_entry_point("retrieval")
    agent_executor = workflow.compile()
    return agent_executor


def create_multi_query_agent_executor() -> AgentExecutor:
    workflow = StateGraph(MultiQueryAgentState)
    workflow.add_node("decomposition", decomposition_node)
    workflow.add_node("parallel_retrieval", parallel_retrieval_node)
    workflow.add_node("response", response_node)
    workflow.set_entry_point("decomposition")
    workflow.add_edge("decomposition", "parallel_retrieval")
    workflow.add_edge("parallel_retrieval", "response")
    workflow.add_edge("response", "__end__")
    agent_executor = workflow.compile()
    return agent_executor


def stream_agent_output(agent_executor: AgentExecutor, query: str):
    """Stream the agent's response"""
    wait_till_close = False
    for chunk, metadata in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="messages"
    ):
        if hasattr(chunk, "content") and chunk.content and isinstance(chunk, AIMessage):
            if chunk.content.startswith("{"):
                wait_till_close = True
            if chunk.content.startswith("Searching:"):
                wait_till_close = False

            
            if not wait_till_close:
                yield chunk.content
            
