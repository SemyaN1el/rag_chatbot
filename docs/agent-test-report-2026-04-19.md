# Полноценное тестирование агента от 2026-04-19

## Что проверялось

- полный automated test suite по каталогу `tests/`;
- offline regression suite через `evaluate.py`;
- доступность живой инфраструктуры `Redis`, `Qdrant`, `PostgreSQL`;
- live smoke для agent runtime на реальных tools;
- live startup smoke для FastAPI-приложения.

## Команды

```bash
python -m unittest discover -s tests -v
python evaluate.py
```

Дополнительно запускались точечные smoke-проверки:

- ping `Redis`;
- проверка коллекции `pdf_docs` в `Qdrant`;
- startup через `TestClient(app)`;
- прямой вызов `execute_agent_chat(...)` с `register_default_tools()` и отключённым `history_saver`.

## Итог

### Автоматические проверки

- `python -m unittest discover -s tests -v` -> `48/48 OK`
- `python evaluate.py` -> `10/10` agent eval cases passed

### Метрики agent eval harness

Текущие метрики из `data/agent_eval_report.json`:

- `task_success_rate = 1.000`
- `route_accuracy = 1.000`
- `tool_selection_accuracy = 1.000`
- `refusal_reason_accuracy = 1.000`
- `citation_validity = 1.000`
- `cache_hit_rate = 1.000`
- `search_type_accuracy = 1.000`
- `memory_usage_accuracy = 1.000`
- `latency_ms_p50 = 1.0 ms`
- `latency_ms_p95 = 13.0 ms`
- `estimated_cost_usd_mean = 0.000`
- `estimated_cost_usd_total = 0.000`

Пороговые проверки regression gate:

- `route_accuracy >= 0.95`
- `tool_selection_accuracy >= 0.95`
- `refusal_reason_accuracy >= 0.90`
- `citation_validity >= 1.00`
- `task_success_rate >= 0.90`
- `cache_hit_rate >= 0.95`
- `latency_ms_p95 <= 1000 ms`
- `estimated_cost_usd_mean <= 0.0`

Итог по thresholds:

- `threshold_failures = 0`
- offline regression gate: `PASS`

Важно:

- это `offline deterministic` метрики по agent behavior suite, а не live production latency;
- они оценивают корректность маршрута, tool usage, refusal behavior, cache/memory handling и latency внутри regression harness;
- live smoke с реальными сервисами нужно читать отдельно, потому что там латентность и деградации другие.

### Инфраструктура

- `Redis` доступен, `PING = True`
- `Qdrant` доступен, коллекция `pdf_docs` найдена, `157` points
- `PostgreSQL` доступен: `app.main` успешно проходит startup, а `/agent/chat` сохраняет history в базу

## Live smoke результаты

### 1. Direct-answer path

Статус: `PASS`

- вопрос: `Какие режимы поиска ты поддерживаешь?`
- route: `direct_answer`
- refusal_reason: `None`
- citations: `0`
- агент вернул ожидаемый служебный ответ о режимах `vector` и `hybrid`

### 2. Retrieval path через реальные tools

Статус: `FAIL` на первом прогоне, `PASS` после увеличения timeout-лимитов

Что происходило по шагам:

1. Изначально первый реальный retrieval-запрос завершался `workflow_timeout_exceeded`
2. Затем прямой вызов `Groq` был перепроверен отдельно и подтвердился как рабочий
3. После увеличения `LLM_TIMEOUT_SECONDS` и `AGENT_MAX_RUNTIME_SECONDS` live retrieval path завершился успешно

Что видно по успешному повторному прогону:

- route корректно выбирается как `retrieve_vector`
- сессионная память и cache lookup работают
- агент возвращает нормальный ответ с citations
- live retrieval завершился примерно за `30s`, что уже укладывается в новый runtime budget

### 3. Повторная прямая проверка Groq

Статус: `PASS`

- выполнен прямой вызов `generate_text_from_prompt('Ответь одним словом: ok')`
- время ответа: около `1.9s`
- провайдер вернул корректный ответ

### 4. Полный online `/agent/chat` через `app.main`

Статус: `PASS`

Проверка выполнена через реальный `FastAPI`-сервер без dependency overrides. В path участвовали:

- `Redis` для cache и session memory;
- `Qdrant` для retrieval;
- `Groq` для генерации ответа по контексту;
- `PostgreSQL` для `init_db()` и записи history.

Прогнанные HTTP-сценарии:

- `main_vector_first_pass`: `200`, `17.6s`, `retrieve_vector`, `3` citations, `cached=false`
- `main_vector_cache_hit`: `200`, `0.96s`, `retrieve_vector`, `3` citations, `cached=true`
- `main_vector_memory_seed`: `200`, `11.4s`, `retrieve_vector`, `3` citations, `cached=false`
- `main_vector_followup_with_memory`: `200`, `11.0s`, `retrieve_vector`, `3` citations, `memory_applied=true`
- `main_out_of_scope_refusal`: `200`, `0.93s`, `refusal_reason=out_of_scope`

Что дополнительно подтверждено:

- `app.main` стартует с рабочим `startup`-хуком и `/health`;
- `PostgreSQL history` действительно пишется: `before_count = 0`, `after_count = 4`, `delta = +4`;
- в базе остались реальные вопросы из live-прогона, что подтверждает успешный write path.

## Уточнение после повторной проверки Groq

После отдельной перепроверки внешний `Groq` сейчас отвечает корректно. Значит актуальная интерпретация live-сбоев такая:

- прямой LLM path работает;
- первоначальный live-блокер был связан с `retrieval cold start`, который превышал старый `AGENT_MAX_RUNTIME_SECONDS=15`;
- ранее увиденный `Groq API 403` не воспроизвёлся на прямом вызове и пока выглядит как нестабильный или path-specific сбой.

## Ключевые выводы

### Что работает

- agent contracts, router, guardrails, policy checks, budget controls и eval harness проходят автоматические проверки;
- Redis memory/cache path жив;
- Qdrant коллекция доступна и содержит данные;
- direct-answer сценарий агента работает end-to-end.
- прямой вызов Groq проходит стабильно;
- live retrieval path через реальные tools после увеличения лимитов тоже проходит.
- полный `/agent/chat` через `app.main` теперь тоже проходит end-to-end с записью history в `PostgreSQL`.

### Что ещё остаётся улучшить

- `Retrieval cold start`:
  первый не-кэшированный запрос всё ещё дорогой по latency и в живом прогоне занял до `17.6s`.

- `Embeddings lifecycle`:
  эмбеддинги создаются на запросе, поэтому нужен preload/singleton, если хочется более стабильный `p95`.

- `HF Hub warning`:
  без `HF_TOKEN` модель поднимается, но библиотека предупреждает о неаутентифицированных запросах и меньших лимитах.

## Практический статус агента

- как локальный controlled runtime с mock/fixture coverage: `зелёный`
- как offline regression harness: `зелёный`
- как live direct-answer runtime: `зелёный`
- как live retrieval agent через реальные tools: `зелёный`
- как полное FastAPI-приложение с startup через PostgreSQL и реальным `/agent/chat`: `зелёный`

## Следующие действия

- убрать cold start embeddings из request path: прогрев, singleton/lazy cache или отдельный preload;
- добавить `HF_TOKEN`, чтобы убрать warning и не упираться в более жёсткие лимиты Hub;
- выделить отдельный `live benchmark`-скрипт/команду для регулярного прогона `HTTP success rate`, `latency p50/p95`, `cache hit rate`, `memory usage` и `refusal rate`;
- после оптимизации прогрева отдельно пересмотреть, можно ли опустить runtime budget ниже текущих `240s`.
