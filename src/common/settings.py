from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    milvus_uri          : str = "http://localhost:19530"
    milvus_collection   : str = "finrad"
    embedding_model     : str = "Octen/Octen-Embedding-4B"
    embedding_base_url  : str = "http://localhost:30001/v1"
    embedding_dim       : int = 2560
    embedding_batch_size: int = 64
    finrad_dataset_name : str = "sohomghosh/FinRAD_Financial_Readability_Assessment_Dataset"


settings = Settings()
