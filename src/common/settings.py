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
    
    log_level           : str = "INFO"
    
    llm_model           : str = "openai/gpt-oss-20b"
    llm_base_url        : str = "http://localhost:30000/v1"
    llm_api_key         : str = "ignored"

    system_prompt       : str = (
        "You are a financial glossary assistant with access to the FinRAD dataset.\n"
        "You can load the dataset, search for financial terms, show database statistics, and clear the collection.\n"
        "Always use the search tool to answer questions about financial terms and definitions.\n"
    )


settings = Settings()
