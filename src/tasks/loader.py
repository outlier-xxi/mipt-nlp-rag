import asyncio
import hashlib

from datasets import load_dataset
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

from src.common.settings import settings
from src.common.vdb import get_client, ensure_collection, upsert_records


async def load_finrad() -> None:
    logger.info(f"loading dataset '{settings.finrad_dataset_name}'")
    dataset = load_dataset(settings.finrad_dataset_name)["train"]
    total = len(dataset)
    logger.info(f"dataset loaded: {total} records")

    client = get_client()
    await asyncio.to_thread(ensure_collection, client)

    aclient = AsyncOpenAI(
        api_key="ignored",
        base_url=settings.embedding_base_url,
    )

    batch_size = settings.embedding_batch_size
    batches = range(0, total, batch_size)

    for start in tqdm(batches, desc="upserting batches", unit="batch"):
        end = min(start + batch_size, total)
        batch = dataset.select(range(start, end))

        texts = [f"{row['terms']}: {row['definitions']}" for row in batch]

        response = await aclient.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]

        records = [
            {
                "id": hashlib.sha256(f"{row['terms']}{row['definitions']}".encode()).hexdigest(),
                "term": row["terms"],
                "definition": row["definitions"],
                "source": row["source"],
                "embedding": emb,
            }
            for row, emb in zip(batch, embeddings)
        ]

        await asyncio.to_thread(upsert_records, client, records)

    logger.info(f"all {total} records upserted")


if __name__ == "__main__":
    asyncio.run(load_finrad())
