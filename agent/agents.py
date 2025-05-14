import copy

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler


from config.utils import get_logger
from agent.agent_states import BaseAgentState, MultiQueryAgentState
from agent.nodes_and_tools import (
    retrieval_node, 
    response_node, 
    router_node, 
    parallel_retrieval_node,
    return_output_node
)


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


def _route(state: MultiQueryAgentState) -> MultiQueryAgentState:
    decision_type = state["decision_type"]
    if decision_type == "DECOMPOSE":
        return "parallel_retrieval"
    else:
        return "return_output"


def create_multi_query_agent_executor() -> AgentExecutor:
    workflow = StateGraph(MultiQueryAgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("parallel_retrieval", parallel_retrieval_node)
    workflow.add_node("response", response_node)
    workflow.add_node("return_output", return_output_node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        _route,
        {
            "parallel_retrieval": "parallel_retrieval",
            "return_output": "return_output"
        }
    )
    workflow.add_edge("parallel_retrieval", "response")
    workflow.add_edge("response", "return_output")
    workflow.add_edge("return_output", "__end__")
    agent_executor = workflow.compile()
    return agent_executor


class StateCapture(BaseCallbackHandler):
    def __init__(self):
        self.messages = []
        
    def on_llm_start(self, *args, **kwargs):
        ...
            
    def on_chain_start(self, serialized, inputs, **kwargs):
        ...
            
    def on_chain_end(self, outputs, **kwargs):
        # This is where the updated state is often returned
        logger.info(f"I am in on_chain_end")
        if "messages" in outputs:
            self.messages = copy.deepcopy(outputs["messages"])
            
    def on_tool_end(self, output, **kwargs):
        ...
    

def stream_agent_output(agent_executor: AgentExecutor, query: str, state_capture: StateCapture):
    """Stream the agent's response"""
    langchain_is_being_funny = False
    for chunk, metadata in agent_executor.stream(
        {"messages": state_capture.messages + [HumanMessage(content=query)]},
        stream_mode="messages",
        config={"callbacks": [state_capture]}
    ):
        if hasattr(chunk, "content") and chunk.content and isinstance(chunk, AIMessage):
            if chunk.content.startswith("{"):
                langchain_is_being_funny = True
            if chunk.content.startswith("Searching:"):
                langchain_is_being_funny = False
            if chunk.content.startswith("}"):
                langchain_is_being_funny = False
                continue
            
            if not langchain_is_being_funny:
                yield chunk.content
            
