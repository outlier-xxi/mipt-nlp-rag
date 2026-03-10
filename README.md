# Agentic Retrieval

ДЗ-1. Chat Agent, у которого есть tool search по векторой БД.

- LLM и embedding model запущены на DGX Spark GB10. Конфигурация SGLang: https://github.com/outlier-xxi/sglang

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
