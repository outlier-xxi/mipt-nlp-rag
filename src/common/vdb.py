from loguru import logger
from pymilvus import MilvusClient, DataType

from .settings import settings


def get_client() -> MilvusClient:
    logger.info("connecting to milvus at {}", settings.milvus_uri)
    return MilvusClient(settings.milvus_uri)


def ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(settings.milvus_collection):
        logger.info("collection '{}' already exists, skipping creation", settings.milvus_collection)
        return

    logger.info("creating collection '{}'", settings.milvus_collection)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
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
    logger.info("collection '{}' created with hnsw cosine index", settings.milvus_collection)


def insert_records(client: MilvusClient, records: list[dict]) -> None:
    client.insert(collection_name=settings.milvus_collection, data=records)
