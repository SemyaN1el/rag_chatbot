# RAG Chatbot

Локальный RAG-чатбот по PDF-документу с двумя режимами retrieval:
- векторный поиск через Qdrant
- гибридный поиск через BM25 + vector search + RRF

Проект сейчас ориентирован на качественную работу с одним PDF за раз.

## Стек

- FastAPI
- LangChain
- Qdrant
- Groq API
- sentence-transformers
- rank-bm25
- PostgreSQL
- Redis
- RAGAS

## Структура проекта

```text
rag_chatbot/
├── app/
│   ├── main.py
│   ├── routers/chat.py
│   └── services/
├── config.py
├── ingest.py
├── chat.py
├── hybrid_chat.py
├── hybrid_search.py
├── evaluate.py
├── docker-compose.yml
└── requirements.txt
```

## Требования

- Python 3.12
- Docker Desktop
- API-ключ Groq

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

Альтернатива через `pyproject.toml`:

```bash
pip install -e .
```

### 4. Поднять инфраструктуру

```bash
docker compose up -d
```

Это поднимет:
- Qdrant на `http://localhost:6333`
- PostgreSQL на `localhost:5432`
- Redis на `localhost:6379`

### 5. Настроить `.env`

Скопируй пример и задай переменные:

```powershell
Copy-Item .env.example .env
```

Минимально нужны:

- `LLM_PROVIDER=groq`
- `GROQ_MODEL=llama-3.3-70b-versatile`
- `GROQ_API_KEY=...`

## Запуск

### 1. Индексация PDF

Если в папке `data/` лежит ровно один PDF:

```bash
python ingest.py
```

Если PDF несколько, передайте путь явно:

```bash
python ingest.py data/document.pdf
```

Важно: текущая реализация пересоздаёт коллекцию в Qdrant, то есть активным остаётся один документ.

### 2. Консольный чат

Векторный режим:

```bash
python chat.py
```

Гибридный режим:

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

Пример запроса:

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

Agent endpoint возвращает структурированный ответ с:

- `request_id`
- `session_id`
- `search_type`
- `answer`
- `citations`
- `confidence`
- `refusal_reason`
- `trace`

### 4. Оценка качества

Быстрый прогон:

```bash
python evaluate.py --quick
```

Полный прогон:

```bash
python evaluate.py
```

Свой eval-набор:

```bash
python evaluate.py data/my_eval_questions.json
```

Формат eval-файла:

```json
[
  {
    "question": "В каком семестре проводится итоговая аттестация?",
    "ground_truth": "Итоговая государственная аттестация проводится в восьмом семестре."
  }
]
```

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

## Метрики качества

- `faithfulness` — не галлюцинирует ли модель
- `answer_relevancy` — релевантен ли ответ вопросу
- `context_recall` — полно ли найден нужный контекст

## Конфигурация

Основные настройки находятся в `config.py`:

```python
LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
```
