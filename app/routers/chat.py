from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag import ask_vector, ask_hybrid
from app.services.history import save_to_history, get_history
from app.services.cache import get_cached, set_cached, clear_cache

router = APIRouter(prefix="/chat", tags=["chat"])


# Pydantic модели — описывают структуру запроса и ответа
class AskRequest(BaseModel):
    question: str
    search_type: str = "vector"  # vector или hybrid


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    search_type: str
    cached: bool = False


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Задать вопрос по документу"""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    if request.search_type not in ["vector", "hybrid"]:
        raise HTTPException(status_code=400, detail="search_type должен быть vector или hybrid")

    # Проверяем кэш
    cached = get_cached(request.question, request.search_type)
    if cached:
        return AskResponse(**cached, cached=True)

    # Получаем ответ
    if request.search_type == "hybrid":
        result = ask_hybrid(request.question)
    else:
        result = ask_vector(request.question)

    # Сохраняем в кэш и историю
    set_cached(request.question, request.search_type, result)
    save_to_history(request.question, result["answer"], request.search_type)

    return AskResponse(**result, cached=False)


@router.get("/history")
def history(limit: int = 10):
    """Получить историю чатов"""
    return get_history(limit)


@router.delete("/cache")
def delete_cache():
    """Очистить кэш"""
    count = clear_cache()
    return {"deleted_keys": count}