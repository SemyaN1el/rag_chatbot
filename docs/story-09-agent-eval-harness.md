# История 9: Agent eval harness

**Статус:** implemented

## Зачем эта история нужна

После `Истории 8` у агента уже есть:

- bounded runtime;
- router;
- tool layer;
- memory;
- guardrails;
- budget controls и policy checks.

Сейчас проект умеет оценивать в основном `RAG-качество ответа`, но ещё не умеет системно оценивать `поведение агента как runtime`.

Это создаёт риск: ответ может выглядеть приемлемо, но агент мог:

- выбрать неправильный route;
- дернуть не тот tool;
- не отказаться там, где должен был отказаться;
- дать ответ без корректных citations;
- деградировать не тем способом при budget/policy ограничениях.

Цель `Истории 9` — ввести отдельный `agent eval harness`, который оценивает не только текст ответа, но и саму траекторию поведения агента.

## User story

Как разработчик agent runtime,
я хочу запускать воспроизводимый regression/eval набор для `/agent/chat`,
чтобы проверять correctness не только ответа, но и `route`, `tool choice`, `refusal behavior`, `citations` и базовые runtime-метрики до внесения следующих изменений в agent workflow.

## Что входит в историю

### 1. Отдельный eval-набор для agent behavior

Нужен отдельный task suite, который покрывает минимум такие классы кейсов:

- `direct_answer`
- `clarify`
- `refuse_out_of_scope`
- `unsafe_input_refusal`
- `retrieve_vector`
- `retrieve_hybrid`
- `followup_with_memory`
- `cache_hit`
- `policy_or_budget_refusal` как минимум в smoke-формате

Каждый кейс должен быть описан как структура данных, а не захардкожен внутри тестов.

### 2. Единый runner для agent eval

Нужен runner, который:

- прогоняет набор agent-кейсов;
- собирает фактический response и trace;
- сравнивает результат с ожидаемым поведением;
- считает агрегированные метрики;
- сохраняет понятный отчёт.

### 3. Метрики agent-level, а не только answer-level

Минимальный набор метрик для `v1`:

- `route_accuracy`
- `tool_selection_accuracy`
- `refusal_reason_accuracy`
- `citation_validity`
- `task_success_rate`
- `cache_hit_rate`
- `latency_ms_p50`
- `latency_ms_p95`

Дополнительно, если реализация будет достаточно лёгкой:

- `groundedness`
- `refusal_quality`
- `answer_correctness`

Но для `v1` приоритет у детерминированных agent-level метрик, а не у сложного judge-based evaluation.

### 4. Regression mode

Должен появиться режим, который можно запускать локально и в CI как regression gate:

- suite проходит;
- summary печатается в консоль;
- артефакт сохраняется в `json` или `md`;
- при провале thresholds команда завершается `non-zero exit code`.

## Предлагаемый формат eval-case

Минимальная структура кейса:

```json
{
  "id": "route_direct_answer_001",
  "category": "direct_answer",
  "request": {
    "question": "Какие режимы поиска ты поддерживаешь?",
    "search_type": "vector",
    "session_id": "eval-session-1"
  },
  "expected": {
    "route": "direct_answer",
    "outcome": "direct_answer",
    "refusal_reason": null,
    "tool_names": [],
    "min_citation_count": 0,
    "max_citation_count": 0
  }
}
```

Для retrieval-кейсов нужно поддержать дополнительные поля:

- `expected.search_type`
- `expected.required_tools`
- `expected.forbidden_tools`
- `expected.min_citation_count`
- `expected.answer_contains`

Для refusal-кейсов:

- `expected.refusal_reason`
- `expected.route`
- `expected.forbidden_tools`

Для memory/cache-кейсов:

- `preloaded_session_memory`
- `preloaded_cache`

## Предлагаемые артефакты реализации

Минимальный состав файлов для реализации:

- `app/agent/evals/schemas.py`
- `app/agent/evals/runner.py`
- `app/agent/evals/metrics.py`
- `data/agent_eval_cases.json`
- `tests/test_agent_eval_runner.py`

Если удобнее, можно сделать один файл `app/agent/evals.py`, но разбиение на `schemas / runner / metrics` предпочтительнее.

Фактически реализовано:

- `app/agent/evals/schemas.py`
- `app/agent/evals/metrics.py`
- `app/agent/evals/runner.py`
- `data/agent_eval_cases.json`
- `evaluate.py`
- `tests/test_agent_eval_runner.py`

## Acceptance criteria

История считается завершённой, если выполнены все пункты:

- [x] Появился отдельный agent eval runner, не смешанный напрямую со старым RAG eval-потоком.
- [x] Появился внешний набор agent eval cases в виде данных, а не только unit-тестов.
- [x] Набор покрывает минимум `direct_answer`, `clarify`, `refuse`, `retrieve_vector`, `retrieve_hybrid`, `cache_hit`, `followup_with_memory`.
- [x] Runner умеет валидировать `route`, `tool usage`, `refusal_reason`, citations и outcome.
- [x] Runner считает и выводит как минимум: `route_accuracy`, `tool_selection_accuracy`, `refusal_reason_accuracy`, `citation_validity`, `task_success_rate`, `cache_hit_rate`, `latency`.
- [x] Runner пишет итоговый артефакт отчёта в машиночитаемом формате.
- [x] Добавлены автоматические тесты на runner и metrics aggregation.
- [x] Зафиксированы pass/fail thresholds для regression mode.

## Рекомендуемые thresholds для v1

Начальные пороги можно взять такими:

- `route_accuracy >= 0.95`
- `tool_selection_accuracy >= 0.95`
- `refusal_reason_accuracy >= 0.90`
- `citation_validity == 1.00` для success-кейсов с retrieval
- `task_success_rate >= 0.90`

Если в первом проходе окажется, что thresholds слишком агрессивны, их можно ослабить, но только явно в документации и без молчаливого downgrade.

## Технические ограничения и решения

### Что не надо делать в этой истории

- не превращать story в полноценную observability-platform;
- не тянуть сложный online eval;
- не строить multi-agent evaluator;
- не смешивать agent eval и старый `RAGAS` pipeline в один запутанный сценарий;
- не делать judge-heavy evaluation как обязательную основу `v1`.

### Что надо сделать прагматично

- deterministic checks сделать основой;
- LLM-as-judge оставить опциональным слоем, если останется время;
- использовать существующий `/agent/chat` контракт и trace как основной источник проверки.

Решение в реализации:

- legacy `RAGAS` pipeline удалён из активного проекта вместо параллельной поддержки двух разных eval-путей;
- новый `evaluate.py` теперь является entrypoint именно для agent-level regression suite.

## Разбиение на маленькие шаги

Рекомендуемая последовательность реализации:

1. Ввести schema для eval case и eval result.
2. Подготовить первые 8-12 agent-кейсов.
3. Реализовать runner, который исполняет кейсы и сохраняет raw results.
4. Реализовать metrics aggregation.
5. Добавить thresholds и regression exit code.
6. Добавить unit-тесты на parsing, scoring и summary.
7. Привязать запуск к удобной локальной команде.

## Definition of done

История 9 готова, когда разработчик может одной командой:

- прогнать agent eval suite;
- увидеть, какие кейсы провалились;
- понять, проблема в `route`, `tool choice`, `refusal`, `citations` или latency;
- использовать это как regression gate перед следующими историями.
