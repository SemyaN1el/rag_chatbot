# RAG Chatbot

Чат-бот для работы с PDF документами на основе RAG (Retrieval-Augmented Generation).
Поддерживает векторный и гибридный поиск (BM25 + векторный + RRF).

## Стек

- **LangChain** — оркестрация RAG pipeline
- **Qdrant** — векторная база данных
- **Ollama** — локальный LLM inference
- **sentence-transformers** — модель эмбеддингов (multilingual-e5-large)
- **FAISS** — векторный поиск (Проект 1)
- **rank_bm25** — BM25 поиск
- **RAGAS** — оценка качества RAG

## Структура проекта
```
rag_chatbot/
├── config.py           # настройки проекта
├── ingest.py           # загрузка PDF и индексация в Qdrant
├── chat.py             # чат-бот на основе векторного поиска
├── hybrid_search.py    # гибридный поиск BM25 + векторный + RRF
├── hybrid_chat.py      # чат-бот на основе гибридного поиска
└── evaluate.py         # оценка качества через RAGAS
```

## Требования

- Python 3.12+
- Docker Desktop
- Ollama

## Установка

### 1. Клонируем репозиторий
```bash
git clone https://github.com/ИМЯ/rag_chatbot.git
cd rag_chatbot
```

### 2. Создаём виртуальное окружение
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Устанавливаем зависимости
```bash
pip install langchain langchain-community langchain-ollama langchain-qdrant
pip install langchain-text-splitters langchain-core
pip install qdrant-client sentence-transformers pypdf
pip install rank_bm25 ragas datasets
```

### 4. Запускаем Qdrant через Docker
```bash
docker run -p 6333:6333 \
  -v C:/путь/до/rag_chatbot/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 5. Устанавливаем и запускаем Ollama

Скачать с [ollama.com](https://ollama.com), затем:
```bash
ollama pull llama3.2
ollama serve
```

## Использование

### Шаг 1 — Индексируем PDF

Положи PDF в папку `data/` и запусти:
```bash
python ingest.py data/document.pdf
```

### Шаг 2 — Запускаем чат-бот

Векторный поиск:
```bash
python chat.py
```

Гибридный поиск (BM25 + векторный + RRF):
```bash
python hybrid_chat.py
```

### Шаг 3 — Оцениваем качество (опционально)
```bash
python evaluate.py
```

## Архитектура

### Индексация (один раз)
```
PDF
  └─> PyPDFLoader        # извлекаем текст постранично
  └─> RecursiveCharacterTextSplitter  # режем на чанки ~500 символов
  └─> multilingual-e5-large           # создаём эмбеддинги
  └─> Qdrant                          # сохраняем векторы + текст
```

### Векторный поиск (каждый запрос)
```
Вопрос
  └─> эмбеддинг запроса
  └─> Qdrant similarity search
  └─> топ-K чанков
  └─> промпт = контекст + вопрос
  └─> Ollama (llama3.2)
  └─> ответ
```

### Гибридный поиск (каждый запрос)
```
Вопрос
  └─> BM25 поиск (по ключевым словам)  ─┐
  └─> векторный поиск (по смыслу)      ─┤ RRF объединение
                                         └─> топ-K чанков
                                         └─> промпт + Ollama
                                         └─> ответ
```

## Метрики качества (RAGAS)

| Метрика | Описание |
|---------|----------|
| faithfulness | не галлюцинирует ли модель |
| answer_relevancy | релевантен ли ответ вопросу |
| context_recall | полно ли найден нужный контекст |

## Конфигурация

Все параметры в `config.py`:
```python
OLLAMA_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
```