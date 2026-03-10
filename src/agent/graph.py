from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.common.log import logger
from src.common.settings import settings
from src.agent.tools import (
    tool_load_dataset,
    tool_clear_collection,
    tool_search,
    tool_collection_stats,
)

SYSTEM_PROMPT = """You are a financial glossary assistant with access to the FinRAD dataset.
You can load the dataset, search for financial terms, show database statistics, and clear the collection.
Always use the search tool to answer questions about financial terms and definitions.
"""

# Strip provider prefix (e.g. "openai/") — ChatOpenAI expects just the model name
_model_name = settings.llm_model.split("/", 1)[-1]

logger.info("initializing llm")
llm = ChatOpenAI(
    model=_model_name,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url or None,
    temperature=0,
)

tools = [
    tool_load_dataset,
    tool_clear_collection,
    tool_search,
    tool_collection_stats,
]

logger.info("creating react agent graph")
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)
