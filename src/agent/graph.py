from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from src.agent.tools import (
    tool_clear_collection,
    tool_collection_stats,
    tool_load_dataset,
    tool_search,
)
from src.common.log import logger
from src.common.settings import settings


class Agent:
    """ReAct agent: LLM + tools loop built on LangGraph StateGraph."""

    def __init__(self) -> None:
        _model_name = settings.llm_model.split("/", 1)[-1]
        logger.info(f"initializing llm: {_model_name}")

        self._tools = [
            tool_load_dataset,
            tool_clear_collection,
            tool_search,
            tool_collection_stats,
        ]

        self._llm = ChatOpenAI(
            model=_model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
            temperature=0,
        ).bind_tools(self._tools)

        logger.info("building react agent graph")
        self.graph = self._build_graph()
        logger.info("agent ready")

    def _agent_node(self, state: MessagesState) -> dict:
        """Call the LLM, prepending the system prompt."""
        system = SystemMessage(content=settings.system_prompt)
        messages = [system] + state["messages"]
        logger.info("agent node: calling llm")
        response = self._llm.invoke(messages)
        return {"messages": [response]}

    @staticmethod
    def _should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        """Route to tool_node if the LLM made tool calls, else end."""
        last = state["messages"][-1]
        if last.tool_calls:
            logger.info(f"routing to tools: {[tc['name'] for tc in last.tool_calls]}")
            return "tools"
        logger.info("no tool calls, ending turn")
        return END

    def _build_graph(self):
        tool_node = ToolNode(self._tools)

        builder = StateGraph(MessagesState)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "agent")

        return builder.compile()


# Module-level instance — langgraph.json points to `graph`
agent = Agent()
graph = agent.graph
