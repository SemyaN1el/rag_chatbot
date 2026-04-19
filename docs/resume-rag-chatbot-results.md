# RAG-чатбот для работы с PDF-документами — материал для резюме

## Короткое описание проекта

Личный pet-project, который я развивал от базового RAG-сервиса до controlled agentic RAG-системы. Проект отвечает на вопросы по PDF-документу, поддерживает `vector` и `hybrid` retrieval, использует `Groq API` для генерации, `Qdrant` для поиска, `Redis` для cache и session memory, `PostgreSQL` для history, а также включает agent runtime с router, guardrails, policy checks, budget controls и eval harness.

## Актуальный стек для резюме

`Python`, `FastAPI`, `Qdrant`, `Redis`, `PostgreSQL`, `LangChain`, `Groq API`, `Docker Compose`, `BM25`, `RRF`, `Pydantic`, `unittest`

## Что реально сделано в проекте

- Реализовал полный RAG-пайплайн для работы с PDF: ingestion, chunking, embeddings, retrieval и генерацию ответа по контексту.
- Добавил два режима поиска: `vector search` и `hybrid retrieval` на базе `BM25 + semantic search + Reciprocal Rank Fusion`.
- Построил `REST API` на `FastAPI`, включая отдельный endpoint `POST /agent/chat` для агентного сценария поверх retrieval.
- Перевёл проект от простого RAG к `controlled agent runtime`: добавил типизированное состояние агента, tool layer, trace и структурированный контракт ответа.
- Реализовал rule-based router с маршрутами `direct_answer`, `clarify`, `refuse`, `retrieve_vector`, `retrieve_hybrid`.
- Добавил `guardrails`, output validation, policy checks и budget controls, чтобы unsafe input, tool misuse и превышение лимитов переводились в controlled refusal, а не в падение сервиса.
- Реализовал `session memory` в `Redis`: хранение summary и recent turns, применение памяти к follow-up вопросам, обновление контекста после успешного ответа.
- Добавил observability-слой: structured logging, request tracing и step-level telemetry для agent workflow.
- Убрал legacy `RAGAS`-оценку и заменил её на собственный `agent eval harness`, который проверяет route, tool usage, refusal behavior, citations, cache/memory и latency.
- Поднял локальную multi-service инфраструктуру через `Docker Compose` для `Qdrant`, `Redis` и `PostgreSQL`.

## Проверяемые результаты

- `48/48` automated tests проходят успешно.
- `10/10` agent eval cases проходят успешно.
- Проведён live online прогон через реальный `app.main` и `POST /agent/chat`: `5/5 HTTP 200`.
- Из live-сценариев: `4` успешных ответа по документу и `1` ожидаемый refusal `out_of_scope`.
- Подтверждён рабочий path `Redis + Qdrant + Groq + PostgreSQL` в реальном HTTP-сценарии.
- Подтверждена запись history в `PostgreSQL`: после live-прогона появилось `+4` записи.
- Зафиксированы реальные latency:
  - первый retrieval miss: около `17.6s`
  - cache hit: около `0.96s`
  - follow-up с session memory: около `11.0s`

## Готовый блок для вставки в резюме

### Вариант 1. Краткий и сильный

**RAG-чатбот для работы с PDF-документами**  
Личный проект | 2026  
`Python` · `FastAPI` · `Qdrant` · `Redis` · `PostgreSQL` · `LangChain` · `Groq API` · `Docker Compose`

- Реализовал RAG-сервис для ответов по PDF-документу с `vector` и `hybrid` retrieval (`BM25 + semantic search + RRF`).
- Развил проект до controlled agentic RAG: добавил router, tool layer, session memory, guardrails, policy checks и budget controls.
- Построил API на `FastAPI` с history в `PostgreSQL`, cache и memory в `Redis`, retrieval в `Qdrant` и генерацией через `Groq API`.
- Разработал собственный `agent eval harness` вместо legacy `RAGAS`; automated suite `48/48`, eval cases `10/10`, live `/agent/chat` — `5/5 HTTP 200`.

### Вариант 2. Чуть более инженерный

**RAG-чатбот для работы с PDF-документами**  
Личный проект | 2026  
`Python` · `FastAPI` · `Qdrant` · `Redis` · `PostgreSQL` · `LangChain` · `Groq API` · `Docker Compose`

- Реализовал ingestion PDF, chunking, embeddings и retrieval-пайплайн с поддержкой `vector search` и `hybrid retrieval`.
- Спроектировал и реализовал agent runtime: typed state, routing, tool wrappers, trace, structured response contract.
- Добавил session memory, follow-up handling, input/output guardrails, policy layer и runtime budgets для controlled behavior.
- Разработал eval и тестовый контур: `48/48` automated tests, `10/10` eval cases, live online проверка `POST /agent/chat` через реальный HTTP path.

## Формулировки, которые лучше использовать на собеседовании

- "Перевёл pet-project от базового RAG к controlled agentic RAG."
- "Сделал не только retrieval, но и orchestration-слой: router, tools, memory, guardrails, policy, budget."
- "Добавил собственный eval harness для agent behavior вместо только answer-level оценки."
- "Проверял проект не только unit-тестами, но и реальным online path через `Redis + Qdrant + Groq + PostgreSQL`."

## Формулировки, которых лучше избегать

- Не стоит писать, что это `production-ready multi-agent system`.
- Не стоит писать про полноценный planner или long-term memory, потому что этого в проекте пока нет.
- Не стоит оставлять в актуальной версии стека `Ollama` и `RAGAS`, потому что текущая версия проекта уже переведена на `Groq API` и собственный eval harness.
