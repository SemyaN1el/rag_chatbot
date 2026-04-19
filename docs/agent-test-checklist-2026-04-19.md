# Чеклист тестирования агента от 2026-04-19

- [x] Запущен полный automated test suite `python -m unittest discover -s tests -v`
- [x] Запущен offline agent eval harness `python evaluate.py`
- [x] Проверена доступность `Redis`
- [x] Проверена доступность `Qdrant` и наличие коллекции `pdf_docs`
- [x] Проверен startup FastAPI-приложения через `TestClient`
- [x] Проверен live direct-answer path на реальных tools
- [x] Проверен live retrieval path на реальных tools
- [x] Повторно проверен прямой вызов Groq вне retrieval path
- [x] Проверен реальный `POST /agent/chat` через `app.main` по HTTP
- [x] Проверена запись history в `PostgreSQL` после live `/agent/chat`
- [x] Зафиксированы найденные блокеры live-сценария в отдельном markdown-отчёте

## Результат

- automated tests: `48/48 OK`
- agent eval: `10/10 OK`
- Redis: `OK`
- Qdrant: `OK`
- PostgreSQL startup path: `OK`
- live direct-answer: `OK`
- live retrieval: `OK` после увеличения timeout-лимитов
- direct Groq call: `OK`
- full online `/agent/chat` via `app.main`: `5/5 HTTP 200`
- PostgreSQL history write path: `OK`, `delta = +4`
