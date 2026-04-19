from fastapi import FastAPI
from app.routers.agent import router as agent_router
from app.routers.chat import router as chat_router
from app.services.history import init_db

app = FastAPI(
    title="RAG Chatbot API",
    description="Чат-бот для работы с PDF документами",
    version="1.0.0"
)

# Подключаем роутеры
app.include_router(chat_router)
app.include_router(agent_router)


@app.on_event("startup")
def startup():
    """Выполняется при старте сервера"""
    init_db()  # создаём таблицы в PostgreSQL
    print("База данных инициализирована")


@app.get("/health")
def health():
    """Проверка что сервер работает"""
    return {"status": "ok"}
