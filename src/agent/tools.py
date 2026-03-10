import asyncio

from langchain.tools import tool
from openai import OpenAI

from src.common.log import logger
from src.common.settings import settings
from src.common.vdb import (
    clear_collection,
    get_client,
    get_collection_stats,
    search_records,
)
from src.tasks.loader import load_finrad


@tool
def tool_load_dataset() -> str:
    """Load the FinRAD financial glossary dataset into Milvus vector database.
    Use this when the user asks to load, import, or populate the database."""

    logger.info("tool invoked: load_dataset")
    asyncio.run(load_finrad())
    return "dataset loaded successfully"


@tool
def tool_clear_collection() -> str:
    """Clear (drop) the Milvus collection. Use when the user asks to clear,
    reset, or delete the database/collection."""
    logger.info("tool invoked: clear_collection")
    client = get_client()
    clear_collection(client)
    return "collection cleared"


@tool
def tool_search(query: str, top_k: int = 5) -> str:
    """Search the financial glossary for terms related to the query.
    Returns the top matching term definitions.

    Args:
        query: The search query or term to look up.
        top_k: Number of results to return (default 5).
    """
    logger.info(f"tool invoked: search, query='{query}', top_k={top_k}")
    oai = OpenAI(api_key=settings.llm_api_key, base_url=settings.embedding_base_url)
    response = oai.embeddings.create(model=settings.embedding_model, input=[query])
    embedding = response.data[0].embedding
    client = get_client()
    hits = search_records(client, embedding, top_k=top_k)
    if not hits:
        return "no results found"
    lines = [
        f"- **{h['term']}**: {h['definition']} (source: {h['source']}, score: {h['score']:.4f})"
        for h in hits
    ]
    return "\n".join(lines)


@tool
def tool_collection_stats() -> str:
    """Show statistics for the Milvus collection (number of records, etc.).
    Use when the user asks about database size, record count, or collection info."""
    logger.info("tool invoked: collection_stats")
    client = get_client()
    stats = get_collection_stats(client)
    return str(stats)
