# RAG Chatbot

Controlled agentic RAG-сервис для работы с PDF-документом. Проект отвечает на вопросы по содержимому загруженного файла, поддерживает `vector` и `hybrid` retrieval, а поверх retrieval-слоя развивает управляемый agent runtime с router, memory, guardrails, policy checks, budget controls и eval harness.

Текущая версия проекта использует:

- `Qdrant` для retrieval;
- `Groq API` для генерации;
- `Redis` для answer-cache и session memory;
- `PostgreSQL` для истории запросов;
- `FastAPI` как публичный API-слой.

## Что умеет проект

- Индексировать PDF в `Qdrant` через чанкинг и embeddings.
- Отвечать на вопросы по документу в режимах `vector` и `hybrid`.
- Отдавать обычный RAG-ответ через `POST /chat/ask`.
- Отдавать структурированный agent-ответ через `POST /agent/chat`.
- Поддерживать session memory для follow-up вопросов.
- Кэшировать повторные вопросы в `Redis`.
- Сохранять историю запросов и ответов в `PostgreSQL`.
- Выполнять route-aware agent workflow с `direct_answer`, `clarify`, `refuse`, `retrieve_vector`, `retrieve_hybrid`.
- Прерывать небезопасные или некорректные сценарии через guardrails, validators, policy checks и runtime budgets.
- Прогонять deterministic regression suite через собственный `agent eval harness`.

## Стек

- `Python`
- `FastAPI`
- `Pydantic`
- `LangChain`
- `Qdrant`
- `Redis`
- `PostgreSQL`
- `Groq API`
- `sentence-transformers`
- `rank-bm25`
- `Docker Compose`

## Структура проекта

```text
rag_chatbot/
├── app/
│   ├── agent/
│   │   ├── evals/
│   │   ├── budget.py
│   │   ├── guardrails.py
│   │   ├── memory.py
│   │   ├── observability.py
│   │   ├── policy.py
│   │   ├── router.py
│   │   ├── runtime.py
│   │   ├── schemas.py
│   │   ├── service_tools.py
│   │   ├── state.py
│   │   ├── tools.py
│   │   ├── validators.py
│   │   └── workflow.py
│   ├── routers/
│   │   ├── agent.py
│   │   └── chat.py
│   ├── services/
│   │   ├── cache.py
│   │   ├── history.py
│   │   ├── llm.py
│   │   └── rag.py
│   └── main.py
├── data/
├── docs/
├── chat.py
├── hybrid_chat.py
├── hybrid_search.py
├── ingest.py
├── evaluate.py
├── docker-compose.yml
├── config.py
└── requirements.txt
```

## Требования

- `Python 3.12`
- `Docker Desktop`
- API-ключ `Groq`

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/SemyaN1el/rag_chatbot.git
cd rag_chatbot
```

### 2. Создать виртуальное окружение

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

Альтернатива:

```bash
pip install -e .
```

### 4. Поднять инфраструктуру

```bash
docker compose up -d
```

Будут доступны:

- `Qdrant` на `http://localhost:6333`
- `PostgreSQL` на `localhost:5432`
- `Redis` на `localhost:6379`

### 5. Настроить `.env`

```powershell
Copy-Item .env.example .env
```

Минимально стоит задать:

- `LLM_PROVIDER=groq`
- `GROQ_MODEL=llama-3.3-70b-versatile`
- `GROQ_API_KEY=...`
- `LLM_TIMEOUT_SECONDS=180`
- `AGENT_MAX_RUNTIME_SECONDS=240`

## Запуск

### 1. Индексация PDF

Если в папке `data/` лежит ровно один PDF:

```bash
python ingest.py
```

Если PDF несколько:

```bash
python ingest.py data/document.pdf
```

Важно: текущая версия пересоздаёт коллекцию в `Qdrant`, поэтому активным остаётся один документ за раз.

### 2. Консольный режим

Векторный поиск:

```bash
python chat.py
```

Гибридный поиск:

```bash
python hybrid_chat.py
```

### 3. API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

После запуска доступны:

- `GET /health`
- `POST /chat/ask`
- `POST /agent/chat`
- `GET /chat/history`
- `DELETE /chat/cache`
- Swagger UI: `http://127.0.0.1:8000/docs`

Пример обычного запроса:

```json
{
  "question": "Какая форма итоговой государственной аттестации предусмотрена документом?",
  "search_type": "vector"
}
```

Пример agent-запроса:

```json
{
  "question": "Какая форма итоговой государственной аттестации предусмотрена документом?",
  "search_type": "vector",
  "session_id": "session-123"
}
```

`POST /agent/chat` возвращает структурированный ответ с полями:

- `request_id`
- `session_id`
- `search_type`
- `cached`
- `answer`
- `citations`
- `confidence`
- `refusal_reason`
- `trace`

## Agent runtime

Поверх retrieval-слоя в проекте есть bounded agent workflow:

1. `input validation`
2. `input guardrails`
3. `session memory lookup`
4. `routing decision`
5. `cache lookup`
6. `retrieval tool execution`
7. `response validation`
8. `cache/session persistence`
9. `structured completion`

Router умеет выбирать:

- `direct_answer`
- `clarify`
- `refuse`
- `retrieve_vector`
- `retrieve_hybrid`

Дополнительно runtime поддерживает:

- `session memory` и follow-up handling;
- `structured logging` и step trace;
- `tool policy checks`;
- `max_steps`, `max_tool_calls`, `max_runtime_seconds`;
- `controlled refusal` вместо аварийного падения workflow.

## Eval и проверка качества

### Agent eval harness

Regression-прогон по встроенному датасету:

```bash
python evaluate.py
```

Свой eval-набор и путь для отчёта:

```bash
python evaluate.py data/my_agent_eval_cases.json --output data/my_agent_eval_report.json
```

По умолчанию:

- входной набор: `data/agent_eval_cases.json`
- выходной отчёт: `data/agent_eval_report.json`
- `exit code 1`, если есть failed cases или threshold failures

Harness проверяет:

- `route_accuracy`
- `tool_selection_accuracy`
- `refusal_reason_accuracy`
- `citation_validity`
- `task_success_rate`
- `cache_hit_rate`
- `latency_ms_p50`
- `latency_ms_p95`

### Последняя подтверждённая проверка

На последнем зафиксированном прогоне:

- automated suite: `48/48 OK`
- agent eval harness: `10/10 OK`
- full online `/agent/chat` через `app.main`: `5/5 HTTP 200`
- из online-сценариев: `4` success и `1` ожидаемый refusal
- первый real retrieval miss: около `17.6s`
- cache hit: около `0.96s`
- follow-up с session memory: около `11.0s`
- history write path в `PostgreSQL`: `OK`

Подробности вынесены в:

- `docs/agent-test-report-2026-04-19.md`
- `docs/agent-test-checklist-2026-04-19.md`
- `docs/current-results.md`

## Архитектура

### Индексация

```text
PDF
  -> PyPDFLoader
  -> RecursiveCharacterTextSplitter
  -> multilingual-e5-large
  -> Qdrant
```

### Векторный поиск

```text
Вопрос
  -> embedding запроса
  -> similarity search в Qdrant
  -> top-k чанков
  -> prompt + Groq API
  -> ответ
```

### Гибридный поиск

```text
Вопрос
  -> BM25
  -> vector search
  -> RRF fusion
  -> top-k чанков
  -> prompt + Groq API
  -> ответ
```

### Agent path

```text
Вопрос
  -> guardrails
  -> session memory
  -> router
  -> cache
  -> retrieval tool
  -> validation
  -> response + trace
```

## Конфигурация

Основные настройки читаются через `config.py` из `.env` и переменных окружения.

Ключевые параметры:

```python
LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TIMEOUT_SECONDS = 180
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_docs"
POSTGRES_URL = "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
AGENT_MAX_STEPS = 16
AGENT_MAX_TOOL_CALLS = 6
AGENT_MAX_RUNTIME_SECONDS = 240
```

## Ограничения

- Система пока работает с одним активным PDF за раз.
- Router пока rule-based, а не model-based planner.
- `Embeddings` и vectorstore создаются в request path, поэтому cold start даёт заметную latency.
- Нет полноценной long-term user memory.
- Нет production-grade monitoring backend и CI-запуска live benchmark.

## Документация

В репозитории уже есть дополнительные markdown-артефакты:

- `docs/current-results.md` — актуальный технический срез проекта
- `docs/implementation-log.md` — накопительный журнал задач
- `docs/story-09-agent-eval-harness.md` — постановка eval harness
- `docs/resume-rag-chatbot-results.md` — резюме-ориентированное описание проекта
