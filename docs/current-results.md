# Текущие результаты по проекту `rag_chatbot`

## Снимок проекта

Проект сейчас представляет собой локальный RAG-чатбот по PDF-документу с двумя режимами retrieval:

- векторный поиск через Qdrant;
- гибридный поиск через BM25 + vector search + RRF.

Текущая архитектура ориентирована на один активный документ за раз. Основной сценарий работы:

1. PDF индексируется через `ingest.py`.
2. Чанки и эмбеддинги сохраняются в Qdrant.
3. Вопрос пользователя проходит через `vector` или `hybrid` retrieval.
4. LLM через Groq API отвечает только по найденному контексту.
5. История сохраняется в PostgreSQL, а в Redis живут answer-cache и session memory.

Инфраструктурная база уже есть:

- API на FastAPI;
- Qdrant для retrieval;
- PostgreSQL для истории;
- Redis для кэша и краткой памяти сессии;
- offline agent eval harness с regression gate;
- Docker Compose для локального запуска сервисов.

## Что уже реализовано

### Базовый RAG

- Индексация PDF через `PyPDFLoader`, `RecursiveCharacterTextSplitter` и `multilingual-e5-large`.
- Векторный режим ответа через retrieval + prompt assembly с вызовом Groq API.
- Гибридный режим через BM25 + vector search + RRF fusion.
- Консольные сценарии для обычного и гибридного чата.

### API и хранение состояния

- `GET /health` для проверки сервиса.
- `POST /chat/ask` с выбором `search_type`.
- `GET /chat/history` для просмотра истории вопросов.
- `DELETE /chat/cache` для очистки Redis-кэша.
- Сохранение истории запросов и ответов в PostgreSQL.
- TTL-кэширование одинаковых вопросов в Redis.
- Краткая память сессии и summary в Redis для `/agent/chat`.

### Оценка качества

- Legacy RAGAS eval удалён и заменён детерминированным `agent eval harness`.
- `evaluate.py` теперь запускает regression suite по внешнему набору `data/agent_eval_cases.json`.
- Agent eval проверяет `route`, `tool usage`, `refusal_reason`, citations, cache/memory behavior и latency.
- JSON-отчёт сохраняется в `data/agent_eval_report.json`, а threshold failures завершают команду с non-zero exit code.

### LLM-провайдер и конфигурация

- Текущая LLM-завязка переведена с Ollama на Groq API.
- Добавлен единый адаптер `app/services/llm.py` для sync/async вызовов и structured outputs.
- Конфиг проекта теперь умеет читать `.env` без отдельной внешней зависимости.
- Добавлен `.env.example` для повторяемой настройки окружения.
- Консольные сценарии, RAG API и eval-слой используют один и тот же Groq-based LLM path.

### Начальный каркас agent runtime

- Добавлен пакет `app/agent` для постепенного перехода к controlled agent runtime.
- Зафиксированы базовые контракты ответа агента: `answer`, `citations`, `confidence`, `refusal_reason`, `trace`.
- Добавлено типизированное состояние агента с routing decision, trace и tool results.
- Добавлен минимальный runtime-слой для инициализации состояния, фиксации маршрута и финализации ответа.
- Добавлены базовые тесты на схемы и состояние агента.

### Tool layer для agent runtime

- Добавлены адаптеры, которые превращают текущие RAG/history/cache сценарии в agent-compatible tools.
- Зарегистрированы инструменты `search_vector`, `search_hybrid`, `get_chat_history`, `get_cached_answer`, `set_cached_answer`, `get_session_memory`, `set_session_memory`.
- `ToolRegistry.execute()` теперь возвращает структурированный `ToolResult` даже при ошибках инструмента.
- Для tool layer добавлены отдельные unit-тесты на регистрацию, успешные вызовы и обработку ошибок.

### Первый agent endpoint

- Добавлен отдельный маршрут `POST /agent/chat`, не ломающий legacy-эндпоинт `POST /chat/ask`.
- Реализован bounded workflow: input validation -> routing -> cache lookup -> retrieval tool -> cache write -> response.
- Ответы `/agent/chat` возвращаются в структурированном агентном формате с `request_id`, `session_id`, `citations`, `confidence`, `refusal_reason` и `trace`.
- Ошибка retrieval tool не валит endpoint, а превращается в контролируемый agent response с отказом и trace.
- Добавлены API-тесты на успешные сценарии, cache hit, контролируемый failure и валидацию пустого вопроса.

### Guardrails v1 и validator ответа

- Добавлен входной guardrail-слой с базовой защитой от injection/jailbreak-паттернов.
- Unsafe input теперь останавливает workflow до cache lookup и retrieval tool calls.
- Добавлен validator ответа, который не пропускает неподтверждённые ответы без контекста или без citations.
- Ответы без источников или без валидных citations переводятся в контролируемый refusal с понятной причиной.
- Trace теперь включает отдельные шаги `input_guardrails_checked` и `response_validated`.

### Observability и structured logging v1

- Для agent runtime добавлен отдельный observability-слой с JSON-логами через `app.agent`.
- Логи теперь фиксируют ключевые agent-события: старт запроса, routing, cache lookup, tool execution, validation, completion и runtime failure.
- В каждое событие включаются `request_id`, `session_id`, outcome/status и полезные технические поля без логирования сырого пользовательского вопроса.
- Tool steps в trace теперь содержат `duration_ms`, что даёт минимальную latency-наблюдаемость по шагам agent workflow.
- Добавлены тесты, которые проверяют и структуру логов, и наличие событий успеха/отказа.

### Session memory и summaries v1

- Добавлен отдельный memory-слой `app/agent/memory.py` для краткой памяти сессии и follow-up эвристик.
- В Redis теперь хранится session memory с summary и последними ходами через отдельные tools `get_session_memory` и `set_session_memory`.
- `/agent/chat` загружает память сессии до retrieval и при коротких/связанных вопросах обогащает поисковый запрос summary и последними turns.
- После успешного ответа агент обновляет session summary и recent turns, чтобы следующие вопросы опирались на контекст сессии.
- Добавлены unit- и API-тесты на follow-up сценарии, обновление памяти и включение memory-шагов в trace.

### Intelligent router v1

- Добавлен отдельный rule-based router `app/agent/router.py`.
- Router теперь умеет выбирать `direct_answer`, `clarify`, `refuse`, `retrieve_vector`, `retrieve_hybrid`.
- Решение теперь принимается не только по `search_type`, но и по содержанию вопроса, session memory и явным эвристикам на broad / meta / out-of-scope / ambiguous запросы.
- Причина выбора маршрута теперь попадает в `trace` и structured logs через `routing_decision_applied`.
- Добавлены unit-тесты на routing logic и API-тесты на direct answer, clarify, refuse и override из `vector` в `hybrid`.

### Budget controls и policy checks v1

- Добавлены отдельные модули `app/agent/budget.py` и `app/agent/policy.py`.
- `AgentRuntime` теперь поддерживает конфигурируемые лимиты `max_steps`, `max_tool_calls`, `max_runtime_seconds`.
- Лимиты вынесены в `.env` через `AGENT_MAX_STEPS`, `AGENT_MAX_TOOL_CALLS`, `AGENT_MAX_RUNTIME_SECONDS`.
- Перед каждым tool call теперь выполняются явные preflight-проверки бюджета и policy.
- Нарушение budget или policy больше не приводит к аварийному падению workflow, а переводит запрос в контролируемый refusal с понятной причиной в `trace` и логах.
- Неприоритетные side-effect шаги вроде `set_session_memory` теперь умеют мягко пропускаться через `optional_tool_skipped`, если бюджет почти исчерпан.
- Добавлены API-тесты на `max_steps`, `max_tool_calls`, timeout, policy deny и controlled degradation.

### Agent eval harness v1

- Добавлен пакет `app/agent/evals` с отдельными schema, runner и metrics aggregation для agent-level regression checks.
- Внешний датасет `data/agent_eval_cases.json` покрывает `direct_answer`, `clarify`, `refuse`, `unsafe_input`, `retrieve_vector`, `retrieve_hybrid`, `cache_hit`, `follow-up with memory`, `policy refusal` и `budget timeout`.
- Runner исполняет реальные agent workflow-сценарии на детерминированных fixtures и сравнивает ожидаемые route/tool/citation/refusal/caching свойства с наблюдаемым trace.
- Зафиксированы thresholds для `route_accuracy`, `tool_selection_accuracy`, `refusal_reason_accuracy`, `citation_validity`, `task_success_rate`, `cache_hit_rate`, `latency_ms_p95` и `estimated_cost_usd_mean`.
- Добавлены unit-тесты на parsing, regression scoring и smoke-прогон полного eval suite.

## Текущие ограничения

### Архитектурные

- Система пока является RAG-сервисом, а не полноценной агентной runtime-системой.
- Появился рабочий router, но он пока полностью rule-based и эвристический.
- Нет отдельного planner-слоя, который выбирает стратегию обработки запроса по содержанию самого вопроса.
- Guardrails и validator появились в базовом виде, но пока покрывают только простые pattern-based проверки и валидацию citations/context.
- Memory всё ещё не интегрирована в единый pipeline как short-term/long-term/context system.

### Retrieval и knowledge layer

- При ingestion коллекция пересоздаётся (`force_recreate=True`), поэтому активным остаётся только один документ.
- В `hybrid_search.py` используется `scroll(limit=1000)`, что плохо масштабируется на большие коллекции.
- Эмбеддинги, vectorstore и LLM создаются повторно во время запросов, что добавляет задержку и нагрузку.
- Нет query transformation-слоя: query rewriting, decomposition, HyDE, reranking.

### Memory и safety

- Появилась базовая session memory в Redis с кратким summary и recent turns для текущей сессии.
- История в PostgreSQL по-прежнему хранится отдельно и ещё не объединена с session memory в общий memory pipeline.
- Нет entity memory, долгосрочной пользовательской памяти и политики записи фактов о пользователе.
- Есть базовые input/output guardrails, но пока только первого уровня.
- Есть начальная защита от простых prompt injection и jailbreak-паттернов.
- Есть базовая проверка на отсутствие подтверждающего контекста и citations.
- Появился базовый policy layer для allowlist tools, route-aware tool permissions и write-after-context правил.
- При этом всё ещё нет более сильного output policy layer-а, PII-protection и доменно-специфичных safety-политик.

### Production readiness

- Часть LLM-конфига уже вынесена в `.env`, но у инфраструктурных настроек всё ещё есть захардкоженные fallback-значения.
- Внутри agent workflow появился базовый structured logging и request-level tracing по шагам.
- В runtime появились базовые budget controls по шагам, tool calls и общему времени выполнения запроса.
- Policy layer теперь умеет блокировать неразрешённые tool calls до фактического исполнения.
- При этом пока нет единой observability-схемы для всего приложения, отдельных метрик, alerting и внешнего trace backend-а.
- Нет alerting, adaptive budgets, cost monitoring и внешнего enforcement/backend-а для политик.
- Появилась локальная и CI-friendly offline проверка агентного поведения, но она пока не привязана к реальному CI pipeline.

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

Что уже появилось в коде:

- Redis теперь используется не только для answer-cache, но и для краткой памяти сессии;
- agent workflow умеет загружать summary/recent turns и применять их к follow-up вопросам;
- после успешных ответов summary сессии автоматически обновляется.

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

Базовый agent eval harness уже появился. Следующее усиление стоит делать не через возврат к RAGAS, а через расширение agent-сценариев и автоматизацию прогона. Приоритетный набор направлений:

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

- Довести конфиг до полностью централизованного settings-слоя без чувствительных fallback-значений в коде.
- Перевести retrieval и память на явные сервисы/инструменты.
- Добавить `docs/implementation-log.md` как постоянный журнал реализации.
- Ввести базовый tracing и request identifiers.

### Этап 2. Controlled agentic RAG

- Развивать уже добавленный маршрут `/agent/chat` в сторону более умного agent workflow.
- Расширить использование `session_id`, agent state и bounded workflow за пределы текущего rule-based сценария.
- Развивать уже добавленный router от rule-based эвристик к более осмысленному decision layer.
- Обогащать structured response и downstream-контракты без потери обратной совместимости.

### Этап 3. Memory и guardrails

- Развить уже добавленную session memory до полного short-term/context memory pipeline.
- Добавить более осмысленные session summaries и политику записи фактов.
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

## Ближайшие истории

История 8 закрыта: в runtime появились budget controls, policy checks и controlled degradation.

История 9 закрыта: в проекте появился отдельный agent eval harness, а старая RAGAS-оценка удалена из активного потока.

### Последняя проверка агента

- Полный automated suite: `48/48 OK`
- Agent eval harness: `10/10 OK`
- `Redis`: доступен
- `Qdrant`: доступен, коллекция `pdf_docs` содержит `157` points
- `PostgreSQL`: startup `app.main` и запись history через live `/agent/chat` проходят
- Live direct-answer сценарий: проходит
- Прямой вызов `Groq`: проходит
- Full online `/agent/chat` через `app.main`: `5/5 HTTP 200`, из них `4` success и `1` ожидаемый refusal
- Первый real vector miss: около `17.6s`
- Cache hit на том же вопросе: около `0.96s`
- Follow-up с session memory: около `11.0s`, `memory_applied=true`
- History в `PostgreSQL`: `delta = +4` записи после live-прогона
- Подробный отчёт: `docs/agent-test-report-2026-04-19.md`
- Краткий чеклист: `docs/agent-test-checklist-2026-04-19.md`

### История 9. Agent eval harness

- Выполнена.
- Offline regression suite уже встроен в проект через `evaluate.py`.
- Следующее усиление истории 9 теперь связано не с реализацией harness, а с его запуском в CI и расширением live-сценариев.

- Оценивать не только ответ, но и `route/tool choice`.
- Добавить regression-набор для `direct_answer`, `clarify`, `refuse`, `retrieve_vector`, `retrieve_hybrid`.
- Зафиксировать pass/fail критерии для agent behavior.
- Добавить нормальные агентные метрики вместо исходного упора только на базовые RAG-метрики.
- Включить как минимум: `route accuracy`, `tool selection accuracy`, `refusal quality`, `citation validity`, `groundedness`, `task success rate`, `latency`, `cost per request`, `cache hit rate`.
- Подробная постановка вынесена в `docs/story-09-agent-eval-harness.md`.

### История 10. Context builder и memory policy

- Выделить отдельный слой сборки контекста.
- Определить, что писать в session memory, а что не писать.
- Развести `short-term memory` и просто хранение последних ходов.

### История 11. Stronger output validation

- Усилить groundedness-checks и проверки citations.
- Сделать более строгую fail-closed валидацию ответа.
- Нормализовать формат подтверждений и причин отказа.

### История 12. Production observability v2

- Добавить route/outcome counters и latency breakdown.
- Ввести error taxonomy для agent runtime.
- Подготовить слой под внешний monitoring backend.

## Рабочие договорённости на будущее

- Все новые проектные отчёты и чеклисты ведутся на русском языке.
- После каждой завершённой задачи журнал `docs/implementation-log.md` дополняется новой записью.
- Если в ходе реализации появляются архитектурно значимые неопределённости, допускаются короткие уточняющие вопросы.
- Если вопрос можно выяснить по коду или конфигу, уточнение не задаётся, решение принимается по фактам репозитория.
