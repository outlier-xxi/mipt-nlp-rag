# Agentic Retrieval

ДЗ-1. Chat Agent, у которого есть tool search по векторой БД.

- LLM и embedding model запущены на DGX Spark GB10. Конфигурация SGLang: https://github.com/outlier-xxi/sglang

## Project architecture

```mermaid
flowchart TB
  subgraph UI["LangGraph UI"]
    Chat["Chat interface"]
  end

  subgraph Agent["Agent (LangGraph)"]
    direction TB
    START --> AgentNode["agent node<br>(LLM + system prompt)"]
    AgentNode --> Route{"tool_calls?"}
    Route -->|yes| ToolsNode["tools node"]
    Route -->|no| END
    ToolsNode --> AgentNode
  end

  subgraph Tools["Tools (src/agent/tools.py)"]
    TLoad["load_dataset"]
    TClear["clear_collection"]
    TSearch["search"]
    TStats["collection_stats"]
  end

  subgraph Backend["Backend"]
    Loader["loader.py<br>(load_finrad)"]
    VDB["vdb.py<br>(Milvus client)"]
  end

  subgraph Dataset["Dataset"]
    FinRAD["FinRAD dataset<br>(HuggingFace)"]
  end

  subgraph VectorStore["Vector Store"]
    Milvus["Milvus<br>(etcd, minio, standalone)"]
  end

  subgraph SGLang["SGLang"]
    LLM["LLM API"]
    Emb["Embedding API"]
  end

  User["User"] <--> Chat
  Chat <--> Agent
  Agent --> Tools
  TLoad --> Loader
  TClear --> VDB
  TSearch --> VDB
  TStats --> VDB
  TSearch --> Emb
  Loader --> FinRAD
  Loader --> Emb
  Loader --> VDB
  VDB <--> Milvus
  AgentNode --> LLM
```

## Screenshots


Список инструментов

![Список инструментов](doc/image/tool-list.png)


Поиск

![Поиск 1](doc/image/tool-search-1.png)

![Поиск 2](doc/image/tool-search-2.png)

![Статистика](doc/image/collection-stats.png)


## Запуск


```shell

docker compose up -d
uv run langgraph dev
```
