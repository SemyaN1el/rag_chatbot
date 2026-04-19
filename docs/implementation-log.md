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
