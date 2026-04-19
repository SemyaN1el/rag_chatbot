# Журнал реализации

Этот файл ведётся накопительно. После каждой завершённой задачи в него добавляется новый раздел, а не перезаписывается весь документ.

## Шаблон записи

### [YYYY-MM-DD] Название задачи

**Статус:** выполнено | частично выполнено | заблокировано

**Цель:** коротко описать, что именно делалось.

**Чеклист выполненного:**

- [ ] Пункт 1
- [ ] Пункт 2
- [ ] Пункт 3

**Изменённые файлы:**

- `path/to/file`

**Проверка:**

- Что проверено вручную
- Какие команды запускались
- Что осталось непроверенным

**Известные follow-up пункты:**

- Следующий логичный шаг
- Открытый риск или ограничение

---

## Записи

### [2026-04-19] Настройка markdown-процесса проекта

**Статус:** выполнено

**Цель:** ввести постоянный процесс проектной документации через папку `docs/`, создать стартовый markdown со срезом проекта и завести единый накопительный журнал реализации.

**Чеклист выполненного:**

- [x] Создана папка `docs/` для проектных markdown-артефактов.
- [x] Создан файл `docs/current-results.md` с текущим состоянием проекта и roadmap по agent system.
- [x] Создан файл `docs/implementation-log.md` как единый накопительный журнал.
- [x] Зафиксирован шаблон будущих записей в журнале.
- [x] Добавлена первая запись по текущей задаче.
- [x] Содержание `current-results.md` основано на фактическом анализе репозитория и выжимке из изученных PDF-лекций.

**Изменённые файлы:**

- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Проверено, что папка `docs/` раньше отсутствовала и создана в рамках этой задачи.
- Проверено, что оба markdown-файла существуют и доступны в репозитории.
- Проверено, что журнал оформлен как накопительный файл и уже содержит первую запись.
- Код и runtime проекта не изменялись; задача затрагивает только документацию.

**Известные follow-up пункты:**

- При следующей реальной реализации дописать в журнал новый раздел, не меняя предыдущие записи.
- При первом этапе agent runtime можно вынести из `current-results.md` отдельный технический дизайн-документ, если описание начнёт разрастаться.

### [2026-04-19] История 1: каркас agent runtime и контракты

**Статус:** выполнено

**Цель:** создать минимальный пакет `app/agent` с типизированными контрактами, состоянием агента, каркасом runtime и базовыми тестами, не меняя текущее поведение `/chat/ask`.

**Чеклист выполненного:**

- [x] Добавлен пакет `app/agent` с модулями `schemas.py`, `state.py`, `tools.py`, `runtime.py`.
- [x] Зафиксирован единый контракт ответа агента: `answer`, `citations`, `confidence`, `refusal_reason`, `trace`.
- [x] Добавлены типы для tool call и tool result.
- [x] Добавлено состояние агента с `request_id`, `session_id`, routing decision, trace, tool results и response.
- [x] Добавлен минимальный `AgentRuntime` для инициализации состояния, фиксации routing decision и финализации ответа.
- [x] Обновлён `pyproject.toml`, чтобы новый пакет входил в сборку.
- [x] Добавлены базовые тесты на схемы и состояние агента.
- [x] Обновлён `docs/current-results.md`, чтобы снимок проекта отражал новые результаты.

**Изменённые файлы:**

- `app/agent/__init__.py`
- `app/agent/schemas.py`
- `app/agent/state.py`
- `app/agent/tools.py`
- `app/agent/runtime.py`
- `tests/test_agent_schemas.py`
- `tests/test_agent_state.py`
- `pyproject.toml`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запустить unit-тесты на схемы и состояние агента.
- Проверить, что текущие API-модули не были переподключены и поведение `/chat/ask` не изменено.
- Проверить, что новый пакет включён в `pyproject.toml`.

**Известные follow-up пункты:**

- Следующий шаг: завернуть текущие retrieval/history/cache сценарии в реальный tool layer.
- После этого можно добавлять `/agent/chat` и bounded workflow поверх новых контрактов.

### [2026-04-19] История 2: tool layer поверх текущего RAG

**Статус:** выполнено

**Цель:** добавить реальный слой агентных инструментов над текущими retrieval/history/cache сервисами, чтобы runtime работал через `ToolCall` и `ToolResult`, а не через прямые вызовы сервисов.

**Чеклист выполненного:**

- [x] Добавлены factory-функции для агентных инструментов над retrieval, history и cache сценариями.
- [x] Зарегистрированы инструменты `search_vector`, `search_hybrid`, `get_chat_history`, `get_cached_answer`, `set_cached_answer`.
- [x] У каждого инструмента зафиксирован единый формат `output`.
- [x] `ToolRegistry.execute()` переведён на структурированный `ToolResult` при ошибках выполнения.
- [x] Добавлены unit-тесты на registry, registration, output shape и error handling.
- [x] Обновлён срез проекта в `docs/current-results.md`.

**Изменённые файлы:**

- `app/agent/__init__.py`
- `app/agent/tools.py`
- `app/agent/service_tools.py`
- `tests/test_agent_tools.py`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запустить unit-тесты на схемы, состояние и новый tool layer.
- Проверить, что старые API-модули и `/chat/ask` не менялись.
- Проверить, что зарегистрированный набор tools соответствует ожидаемому agent toolkit.

**Известные follow-up пункты:**

- Следующий шаг: подключить tool registry и новый bounded workflow в `/agent/chat`.
- После этого можно добавлять guardrails v1 и трассировку шагов уже на реальном запросе.

### [2026-04-19] История 3: первый agent endpoint

**Статус:** выполнено

**Цель:** подключить `AgentRuntime`, `ToolRegistry` и bounded workflow к реальному API-маршруту `/agent/chat`, не ломая существующий `/chat/ask`.

**Чеклист выполненного:**

- [x] Добавлены `AgentChatRequest` и `AgentChatResponse` для нового публичного API-контракта.
- [x] Реализован workflow `input validation -> routing -> cache lookup -> retrieval -> cache write -> response`.
- [x] Добавлен новый маршрут `POST /agent/chat`.
- [x] Сохранён legacy-маршрут `POST /chat/ask` без изменения поведения.
- [x] Добавлена контролируемая деградация при ошибке retrieval tool.
- [x] Добавлены API-тесты на vector/hybrid сценарии, cache hit, tool failure и пустой вопрос.
- [x] Обновлён `readme.md` и текущий срез проекта в `docs/current-results.md`.

**Изменённые файлы:**

- `app/agent/__init__.py`
- `app/agent/schemas.py`
- `app/agent/runtime.py`
- `app/agent/workflow.py`
- `app/routers/agent.py`
- `app/main.py`
- `tests/test_agent_router.py`
- `readme.md`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запустить unit/API-тесты через `python -m unittest discover -s tests -v`.
- Проверить, что `/agent/chat` возвращает структурированный agent response.
- Проверить, что cache hit не приводит к вызову retrieval tool.
- Проверить, что failure внутри retrieval tool не валит endpoint.

**Известные follow-up пункты:**

- Следующий шаг: добавить guardrails v1 и отдельный validator output-а.
- После этого можно расширять routing и memory без поломки публичного контракта `/agent/chat`.

### [2026-04-19] История 4: guardrails v1 и validator ответа

**Статус:** выполнено

**Цель:** добавить базовые guardrails на входе и validator на выходе нового `/agent/chat`, чтобы unsafe input и неподтверждённые ответы переводились в контролируемый refusal.

**Чеклист выполненного:**

- [x] Добавлен входной guardrail-слой с проверкой на простые injection/jailbreak-паттерны.
- [x] Unsafe input теперь останавливает workflow до cache/retrieval tool calls.
- [x] Добавлен validator ответа на достаточность контекста и наличие валидных citations.
- [x] Ответы без источников или без подтверждающих citations переводятся в refusal.
- [x] В trace добавлены отдельные шаги валидации входа и ответа.
- [x] Добавлены тесты на unsafe input, пустой контекст и ответ без citations.
- [x] Обновлён текущий срез проекта в `docs/current-results.md`.

**Изменённые файлы:**

- `app/agent/__init__.py`
- `app/agent/guardrails.py`
- `app/agent/validators.py`
- `app/agent/workflow.py`
- `tests/test_agent_router.py`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запустить `python -m unittest discover -s tests -v`.
- Проверить, что unsafe input больше не доходит до tool layer.
- Проверить, что ответы без контекста или без citations отдаются как refusal, а не как обычный success.

**Известные follow-up пункты:**

- Следующий шаг: расширить guardrails до scope/relevance checks и более сильного output validator-а.
- После этого можно переходить к observability или session memory, не теряя уже введённую safety-базу.

### [2026-04-19] Переход на Groq и `.env`-конфиг

**Статус:** выполнено

**Цель:** заменить текущую LLM-завязку на Groq API, вынести LLM-конфиг в `.env` и перевести retrieval/chat/eval-слой на единый адаптер провайдера.

**Чеклист выполненного:**

- [x] `config.py` переведён на чтение `.env` и переменных окружения.
- [x] Добавлен единый Groq-адаптер `app/services/llm.py`.
- [x] Убраны прямые зависимости от `ChatOllama` в `app/services/rag.py`, `chat.py`, `hybrid_chat.py`, `evaluate.py`.
- [x] В проект добавлен `.env.example` для воспроизводимой настройки.
- [x] Локально создан `.env` с LLM-настройками для Groq.
- [x] Обновлены `requirements.txt` и `pyproject.toml` под новый LLM-path.
- [x] Добавлены unit-тесты для LLM helper-функций.
- [x] Обновлены `readme.md` и текущий срез проекта.

**Изменённые файлы:**

- `config.py`
- `app/services/llm.py`
- `app/services/rag.py`
- `chat.py`
- `hybrid_chat.py`
- `evaluate.py`
- `requirements.txt`
- `pyproject.toml`
- `.env.example`
- `tests/test_llm_service.py`
- `readme.md`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запущено `python -m unittest discover -s tests -v`.
- Проверены импорты `chat`, `evaluate`, `app.services.rag`, `app.services.llm`.
- Дополнительно проверено, что следов `ChatOllama` в проектном коде больше нет.

**Известные follow-up пункты:**

- Следующий шаг: при желании ввести provider abstraction не только для Groq, но и для fallback-провайдера.
- Если потребуется production-hardening, отдельно стоит добавить rotation-friendly secret management вместо локального `.env`.

### [2026-04-19] Уборка служебных `__pycache__` из git

**Статус:** выполнено

**Цель:** убрать из отслеживаемых файлов служебные Python-артефакты, чтобы `git status` не засорялся `.pyc`-файлами.

**Чеклист выполненного:**

- [x] Проверено, что `.gitignore` уже игнорирует `__pycache__/` и `*.py[cod]`.
- [x] Все ранее отслеживаемые `.pyc`-файлы удалены из git-индекса через `git rm --cached`.
- [x] Рабочее дерево очищено от лишних служебных изменений, связанных с байткодом.

**Изменённые файлы:**

- `app/__pycache__/main.cpython-312.pyc`
- `app/routers/__pycache__/chat.cpython-312.pyc`
- `app/services/__pycache__/cache.cpython-312.pyc`
- `app/services/__pycache__/history.cpython-312.pyc`
- `app/services/__pycache__/rag.cpython-312.pyc`
- `docs/implementation-log.md`

**Проверка:**

- Проверить, что `git ls-files | rg __pycache__` больше не возвращает отслеживаемые служебные файлы после коммита.
- Проверить, что новые локальные `.pyc` больше не появляются в составе изменений.

**Известные follow-up пункты:**

- Следующий осмысленный продуктовый шаг: вернуться к roadmap агентной системы и начать `Историю 5` про observability и structured logging.

### [2026-04-19] История 5: observability и structured logging

**Статус:** выполнено

**Цель:** добавить базовую наблюдаемость в agent workflow, чтобы шаги `/agent/chat` были видны и в trace, и во внешних структурированных логах.

**Чеклист выполненного:**

- [x] Добавлен отдельный observability-слой `app/agent/observability.py` с JSON-логами через logger `app.agent`.
- [x] В runtime добавлено логирование инициализации, routing decision, tool execution и runtime failure.
- [x] В workflow добавлено логирование старта запроса, guardrails, cache lookup, response validation и завершения запроса.
- [x] Tool trace steps теперь содержат `duration_ms`.
- [x] В логи не пишется сырой текст вопроса; вместо этого сохраняются технические признаки вроде `question_length`.
- [x] Добавлены тесты на structured logs для success/refusal сценариев.
- [x] Обновлён текущий срез проекта в `docs/current-results.md`.

**Изменённые файлы:**

- `app/agent/observability.py`
- `app/agent/runtime.py`
- `app/agent/workflow.py`
- `app/agent/__init__.py`
- `tests/test_agent_router.py`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запущено `python -m unittest discover -s tests -v`.
- Проверено, что все тесты проходят: `26/26 OK`.
- Проверено, что `/agent/chat` пишет структурированные события `request_started`, `tool_executed`, `response_validated`, `request_completed`.
- Проверено, что tool-шаги в trace содержат `duration_ms`.

**Известные follow-up пункты:**

- Следующий шаг: выделить метрики latency/outcome в отдельный счётчик или backend наблюдаемости.
- После этого логично идти в session memory, router intelligence или budget controls, уже опираясь на появившийся telemetry layer.

### [2026-04-19] История 6: session memory и summaries

**Статус:** выполнено

**Цель:** превратить историю сессии из пассивного архива в рабочую память агента, чтобы follow-up вопросы могли использовать краткий контекст предыдущих ходов.

**Чеклист выполненного:**

- [x] Добавлен memory-слой `app/agent/memory.py` с summary builder, follow-up эвристикой и сборкой memory-augmented question.
- [x] В Redis добавлено отдельное хранение session memory через `get_session_memory` и `set_session_memory`.
- [x] Tool layer расширен инструментами `get_session_memory` и `set_session_memory`.
- [x] `/agent/chat` теперь загружает память сессии до retrieval и при необходимости применяет её к поисковому запросу.
- [x] После успешного ответа agent workflow обновляет summary и recent turns текущей сессии.
- [x] Trace и structured logs дополнены шагами `session_memory_loaded`, `session_memory_applied`, `session_memory_updated`.
- [x] Добавлены unit-тесты на memory helper-функции и API-тесты на follow-up/memory-update сценарии.
- [x] Обновлены `.env.example` и текущий срез проекта.

**Изменённые файлы:**

- `config.py`
- `.env.example`
- `app/services/cache.py`
- `app/agent/memory.py`
- `app/agent/service_tools.py`
- `app/agent/workflow.py`
- `app/agent/runtime.py`
- `app/agent/__init__.py`
- `tests/test_agent_memory.py`
- `tests/test_agent_router.py`
- `tests/test_agent_tools.py`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запущено `python -m unittest discover -s tests -v`.
- Проверено, что все тесты проходят: `31/31 OK`.
- Проверено, что короткий follow-up вопрос использует summary и recent turns при вызове retrieval tool.
- Проверено, что после успешного ответа session memory обновляется и попадает в trace.

**Известные follow-up пункты:**

- Следующий шаг: сделать router умнее и научить его выбирать direct answer / retrieve / clarify по содержанию вопроса, а не только по `search_type`.
- Отдельно стоит решить, какая часть session memory должна сохраняться в PostgreSQL как более долговременная память пользователя, а не только жить в Redis.

### [2026-04-19] История 7: умный router

**Статус:** выполнено

**Цель:** перестать выбирать маршрут обработки только по `search_type` и добавить отдельный rule-based router, который умеет решать, когда нужен direct answer, clarify, refuse, vector retrieval или hybrid retrieval.

**Чеклист выполненного:**

- [x] Добавлен отдельный модуль `app/agent/router.py`.
- [x] В routing enum добавлено состояние `clarify`.
- [x] Router теперь поддерживает решения `direct_answer`, `clarify`, `refuse`, `retrieve_vector`, `retrieve_hybrid`.
- [x] Workflow переведён на исполнение результата router-а вместо прямой привязки к `search_type`.
- [x] Причина маршрутизации теперь попадает в `trace` и structured logs через `routing_decision_applied`.
- [x] Добавлены unit-тесты для router logic.
- [x] Добавлены API-тесты на direct answer, clarify, out-of-scope refuse и override `vector -> hybrid`.
- [x] Обновлён текущий срез проекта в `docs/current-results.md`.

**Изменённые файлы:**

- `app/agent/router.py`
- `app/agent/state.py`
- `app/agent/runtime.py`
- `app/agent/workflow.py`
- `app/agent/__init__.py`
- `tests/test_agent_route_logic.py`
- `tests/test_agent_router.py`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запущено `python -m unittest discover -s tests -v`.
- Проверено, что все тесты проходят: `39/39 OK`.
- Проверено, что meta-вопросы об агенте идут в `direct_answer` без retrieval.
- Проверено, что короткие неяcные вопросы без контекста идут в `clarify`.
- Проверено, что out-of-scope вопросы отсекаются до retrieval.
- Проверено, что широкий сравнительный вопрос может перевести `vector`-запрос в `hybrid`.

**Известные follow-up пункты:**

- Следующий шаг: добавить budget controls и policy checks на уровне agent runtime.
- После этого можно переходить к agent-eval сценарию с проверкой correctness не только ответа, но и route/tool choice.

### [2026-04-19] Обновление backlog после Истории 7

**Статус:** выполнено

**Цель:** зафиксировать в документации ближайшие продуктовые истории после внедрения умного router-а.

**Чеклист выполненного:**

- [x] В `docs/current-results.md` добавлен блок `Ближайшие истории`.
- [x] Зафиксированы следующие шаги: `История 8`–`История 12`.
- [x] Для каждой ближайшей истории добавлены краткие инженерные цели.

**Изменённые файлы:**

- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Проверить, что в `docs/current-results.md` появился раздел `Ближайшие истории`.
- Проверить, что список следующих историй соответствует текущему состоянию проекта после `Истории 7`.

**Известные follow-up пункты:**

- Следующий рабочий шаг по коду: `История 8: budget controls и policy checks`.

### [2026-04-19] Уточнение Истории 9 по метрикам

**Статус:** выполнено

**Цель:** уточнить backlog так, чтобы `История 9` включала не только eval harness, но и полноценные агентные метрики, а не только исходные RAG-метрики.

**Чеклист выполненного:**

- [x] В `docs/current-results.md` расширено описание `Истории 9`.
- [x] Добавлено требование к нормальным агентным метрикам.
- [x] Зафиксирован минимальный набор метрик для будущей реализации.

**Изменённые файлы:**

- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Проверить, что `История 9` теперь включает и eval harness, и набор агентных метрик.
- Проверить, что список метрик не сводится только к старым RAG-метрикам из исходного кода.

**Известные follow-up пункты:**

- При реализации `Истории 9` использовать agent-level метрики как основной слой оценки, а RAG-метрики оставить вспомогательными.

### [2026-04-19] История 8: budget controls и policy checks

**Статус:** выполнено

**Цель:** ограничить агентный runtime по числу шагов, числу tool call-ов и времени выполнения, добавить policy-checks до исполнения инструмента и ввести controlled degradation для необязательных side-effect шагов.

**Чеклист выполненного:**

- [x] Добавлен модуль `app/agent/budget.py` с конфигурируемыми runtime-лимитами и отдельным budget-exception типом.
- [x] Добавлен модуль `app/agent/policy.py` с route-aware allowlist tools и проверками на допустимость side-effect операций.
- [x] `AgentRuntime` переведён на централизованные preflight-проверки бюджета и policy перед tool execution.
- [x] При превышении бюджета или policy-нарушении workflow теперь отдаёт контролируемый refusal с понятной причиной.
- [x] Для необязательных шагов `set_cached_answer` и `set_session_memory` добавлен мягкий skip-path через `optional_tool_skipped`.
- [x] Новые budget-лимиты вынесены в `.env.example`.
- [x] Добавлены API-тесты на `max_steps`, `max_tool_calls`, timeout, policy deny и controlled degradation.
- [x] Обновлён текущий срез проекта в `docs/current-results.md`.

**Изменённые файлы:**

- `app/agent/budget.py`
- `app/agent/policy.py`
- `app/agent/runtime.py`
- `app/agent/state.py`
- `app/agent/workflow.py`
- `app/agent/__init__.py`
- `config.py`
- `.env.example`
- `tests/test_agent_router.py`
- `docs/current-results.md`
- `docs/implementation-log.md`

**Проверка:**

- Запущено `python -m unittest discover -s tests -v`.
- Проверено, что все тесты проходят: `44/44 OK`.
- Проверено, что превышение `max_steps`, `max_tool_calls` и `workflow timeout` приводит к controlled refusal, а не к падению endpoint-а.
- Проверено, что policy deny блокирует tool execution до вызова handler-а.
- Проверено, что поздний необязательный side-effect шаг может быть пропущен по бюджету без потери успешного ответа пользователю.

**Известные follow-up пункты:**

- Следующий шаг: `История 9` с eval harness и агентными метриками на route/tool choice.
- После этого стоит вынести route/outcome/budget counters в отдельный metrics layer, а policy — постепенно усилить до richer output checks и audit-правил.
