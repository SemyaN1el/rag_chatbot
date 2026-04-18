# Текущие результаты по проекту `rag_chatbot`

## Снимок проекта

Проект сейчас представляет собой локальный RAG-чатбот по PDF-документу с двумя режимами retrieval:

- векторный поиск через Qdrant;
- гибридный поиск через BM25 + vector search + RRF.

Текущая архитектура ориентирована на один активный документ за раз. Основной сценарий работы:

1. PDF индексируется через `ingest.py`.
2. Чанки и эмбеддинги сохраняются в Qdrant.
3. Вопрос пользователя проходит через `vector` или `hybrid` retrieval.
4. LLM из Ollama отвечает только по найденному контексту.
5. История сохраняется в PostgreSQL, повторные запросы кэшируются в Redis.

Инфраструктурная база уже есть:

- API на FastAPI;
- Qdrant для retrieval;
- PostgreSQL для истории;
- Redis для кэша;
- eval-скрипт на RAGAS;
- Docker Compose для локального запуска сервисов.

## Что уже реализовано

### Базовый RAG

- Индексация PDF через `PyPDFLoader`, `RecursiveCharacterTextSplitter` и `multilingual-e5-large`.
- Векторный режим ответа через `RetrievalQA`.
- Гибридный режим через BM25 + vector search + RRF fusion.
- Консольные сценарии для обычного и гибридного чата.

### API и хранение состояния

- `GET /health` для проверки сервиса.
- `POST /chat/ask` с выбором `search_type`.
- `GET /chat/history` для просмотра истории вопросов.
- `DELETE /chat/cache` для очистки Redis-кэша.
- Сохранение истории запросов и ответов в PostgreSQL.
- TTL-кэширование одинаковых вопросов в Redis.

### Оценка качества

- Быстрый и полный eval-режим в `evaluate.py`.
- Сравнение vector и hybrid retrieval.
- Метрики `faithfulness`, `answer_relevancy`, `context_recall`.
- Подготовленные eval-наборы в `data/`.

### Начальный каркас agent runtime

- Добавлен пакет `app/agent` для постепенного перехода к controlled agent runtime.
- Зафиксированы базовые контракты ответа агента: `answer`, `citations`, `confidence`, `refusal_reason`, `trace`.
- Добавлено типизированное состояние агента с routing decision, trace и tool results.
- Добавлен минимальный runtime-слой для инициализации состояния, фиксации маршрута и финализации ответа.
- Добавлены базовые тесты на схемы и состояние агента.

### Tool layer для agent runtime

- Добавлены адаптеры, которые превращают текущие RAG/history/cache сценарии в agent-compatible tools.
- Зарегистрированы инструменты `search_vector`, `search_hybrid`, `get_chat_history`, `get_cached_answer`, `set_cached_answer`.
- `ToolRegistry.execute()` теперь возвращает структурированный `ToolResult` даже при ошибках инструмента.
- Для tool layer добавлены отдельные unit-тесты на регистрацию, успешные вызовы и обработку ошибок.

## Текущие ограничения

### Архитектурные

- Система пока является RAG-сервисом, а не полноценной агентной runtime-системой.
- Появился начальный каркас `app/agent`, но он ещё не подключён к API и реальному workflow обработки запросов.
- Появился начальный tool layer, но он ещё не встроен в полноценный orchestration-цикл обработки запросов.
- Нет отдельного router/planner-слоя, который выбирает стратегию обработки запроса на runtime.
- Memory, guardrails и validators ещё не интегрированы в единый pipeline.

### Retrieval и knowledge layer

- При ingestion коллекция пересоздаётся (`force_recreate=True`), поэтому активным остаётся только один документ.
- В `hybrid_search.py` используется `scroll(limit=1000)`, что плохо масштабируется на большие коллекции.
- Эмбеддинги, vectorstore и LLM создаются повторно во время запросов, что добавляет задержку и нагрузку.
- Нет query transformation-слоя: query rewriting, decomposition, HyDE, reranking.

### Memory и safety

- Текущая "память" ограничена историей чата и кэшем, но не оформлена как short-term, long-term и context memory.
- Нет entity memory, session memory summary и политики записи фактов о пользователе.
- Нет input/output guardrails.
- Нет защиты от prompt injection, jailbreak и tool misuse.
- Нет строгой схемы ответа с обязательными citation-ами, confidence и refusal reason.

### Production readiness

- Конфигурация и часть секретов захардкожены.
- Нет централизованного tracing и structured logging по шагам.
- Нет alerting, request budget control, step limits и policy enforcement.
- Нет CI-ориентированной проверки агентного поведения.

## Рекомендации по агентной архитектуре

На основе анализа репозитория и материалов лекций оптимальный следующий шаг для проекта — не multi-agent система, а `controlled agentic RAG`.

Рекомендуемая целевая схема:

1. `Input Guardrails`
2. `Intent Router`
3. `Planner / bounded decision layer`
4. `Tool Layer`
5. `Context Builder`
6. `LLM Answer Node`
7. `Output Validator / Guardrails`
8. `Persistence + Trace`

### Минимальный состав подсистем

- `agent/runtime` — orchestration и state machine;
- `agent/router` — выбор сценария обработки;
- `agent/tools` — строгие tool wrappers для retrieval и памяти;
- `agent/memory` — short-term, session, long-term memory;
- `agent/guardrails` — входные и выходные проверки;
- `agent/validators` — проверка структуры ответа, цитат и confidence;
- `agent/schemas` — Pydantic-схемы для tool calls и ответов.

### Почему не multi-agent сразу

Материалы по workflow и production engineering хорошо показывают, что multi-agent стоит вводить только тогда, когда один агент уже не справляется по причине:

- слишком большого числа инструментов;
- разнородных доменов;
- необходимости независимой верификации;
- слишком длинных и плохо управляемых цепочек шагов.

Для текущего проекта это пока преждевременно. Сначала полезнее получить один надёжный агент с ограниченным циклом действий, прозрачным trace и жёсткими policy-ограничениями.

## Рекомендации по memory и guardrails

### Memory

Нужно явно разделить память на три уровня:

- `short-term memory` — рабочее состояние текущей сессии, последние сообщения, tool trace, промежуточные решения;
- `long-term memory` — подтверждённые пользовательские предпочтения, важные факты и настройки;
- `context memory` — собранный контекст, который реально подаётся модели на конкретном шаге.

Практический старт для проекта:

- Redis использовать как short-term/session state;
- PostgreSQL использовать для истории, summary и long-term facts;
- Qdrant оставить как document memory, а не как пользовательскую долгосрочную память.

### Guardrails

Приоритетные guardrails для внедрения:

- проверка входа на injection/jailbreak-паттерны;
- relevance/scope check до вызова инструмента;
- allowlist доступных tools и жёсткая схема аргументов;
- ограничение числа шагов, retries и tool calls;
- проверка ответа на groundedness, citations и формат;
- отказ при недостаточном контексте вместо догадок;
- маскирование PII в логах и аудит всех решений guardrails.

Лекции отдельно подчёркивают, что один только prompt не является защитой. Защита должна быть инфраструктурной и многоуровневой.

## Рекомендации по eval и production engineering

### Eval

Нужно расширить оценку качества от RAG-метрик к агентным сценариям. Базовый набор:

- корректность ответа;
- полезность для пользователя;
- groundedness/citation faithfulness;
- корректность выбора tool;
- устойчивость к injection и jailbreak;
- refusal quality, если задача вне контекста или вне policy.

Следующий шаг — task suite, где каждая задача включает:

- пользовательский запрос;
- состояние среды;
- ожидаемое поведение;
- критерии pass/fail;
- trace для анализа ошибок.

### Production engineering

Для перехода от прототипа к production-ready системе особенно важны:

- structured logging по каждому шагу;
- tracing запросов и tool calls;
- latency/cost observability;
- fail-safe поведение при проблемах инструментов;
- budget controls: max steps, max tokens, timeout;
- model routing и контроль роста контекста;
- поэтапный rollout: offline eval -> shadow -> canary -> rollout.

## Поэтапный roadmap

### Этап 1. Подготовка архитектурного каркаса

- Вынести конфиг в `.env` и единый settings-слой.
- Перевести retrieval и память на явные сервисы/инструменты.
- Добавить `docs/implementation-log.md` как постоянный журнал реализации.
- Ввести базовый tracing и request identifiers.

### Этап 2. Controlled agentic RAG

- Добавить новый маршрут вида `/agent/chat`.
- Ввести `session_id`, agent state и bounded workflow.
- Реализовать router: direct answer, retrieve, clarify, refuse.
- Добавить structured response с `answer`, `citations`, `confidence`, `refusal_reason`.

### Этап 3. Memory и guardrails

- Разделить память на short-term, long-term и context layers.
- Добавить session summaries и политику записи фактов.
- Реализовать input/output guardrails.
- Ввести allowlist tools, schema validation и action limiters.

### Этап 4. Agent eval и production hardening

- Подготовить agent task suite и regression-набор.
- Добавить сценарии на injection resistance, refusal, tool correctness.
- Подключить tracing/observability-платформу.
- Ввести step budgets, latency/cost monitoring, error taxonomy.

### Этап 5. Осознанное расширение

- Multi-document support.
- Query transformations и reranking.
- Reflection/evaluator step с bounded retry.
- Только после этого — обсуждение multi-agent или MCP-native интеграций.

## Рабочие договорённости на будущее

- Все новые проектные отчёты и чеклисты ведутся на русском языке.
- После каждой завершённой задачи журнал `docs/implementation-log.md` дополняется новой записью.
- Если в ходе реализации появляются архитектурно значимые неопределённости, допускаются короткие уточняющие вопросы.
- Если вопрос можно выяснить по коду или конфигу, уточнение не задаётся, решение принимается по фактам репозитория.
