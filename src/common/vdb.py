from loguru import logger
from pymilvus import MilvusClient, DataType

from src.common.settings import settings


def get_client() -> MilvusClient:
    logger.info(f"connecting to milvus at {settings.milvus_uri}")
    return MilvusClient(settings.milvus_uri)


def ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(settings.milvus_collection):
        logger.info(f"collection '{settings.milvus_collection}' already exists, skipping creation")
        return

    logger.info(f"creating collection '{settings.milvus_collection}'")

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("term", DataType.VARCHAR, max_length=512)
    schema.add_field("definition", DataType.VARCHAR, max_length=4096)
    schema.add_field("source", DataType.VARCHAR, max_length=128)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=settings.milvus_collection,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"collection '{settings.milvus_collection}' created with hnsw cosine index")


def upsert_records(client: MilvusClient, records: list[dict]) -> None:
    client.upsert(collection_name=settings.milvus_collection, data=records)
