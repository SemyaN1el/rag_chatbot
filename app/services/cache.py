import redis
import hashlib
import json
from config import REDIS_HOST, REDIS_PORT, REDIS_TTL, SESSION_MEMORY_TTL

client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True  # возвращать строки а не байты
)


def make_key(question, search_type):
    ###Создаём уникальный ключ для кэша на основе вопроса и типа поиска.  MD5 чтобы ключ был коротким и без спецсимволов.

    raw = f"{search_type}:{question.lower().strip()}"
    return "rag:" + hashlib.md5(raw.encode()).hexdigest()


def make_session_memory_key(session_id):
    raw = session_id.strip().lower()
    return "rag:session:" + hashlib.md5(raw.encode()).hexdigest()


def get_cached(question, search_type):
    key = make_key(question, search_type)
    value = client.get(key)
    if value:
        print(f"   Cache hit: {key}")
        return json.loads(value)
    return None


def set_cached(question, search_type, result):
    key = make_key(question, search_type)
    client.setex(
        name=key,
        time=REDIS_TTL,   # TTL в секундах
        value=json.dumps(result, ensure_ascii=False)
    )
    print(f"   Cached: {key}")


def get_session_memory(session_id):
    key = make_session_memory_key(session_id)
    value = client.get(key)
    if value:
        return json.loads(value)
    return None


def set_session_memory(session_id, memory):
    key = make_session_memory_key(session_id)
    client.setex(
        name=key,
        time=SESSION_MEMORY_TTL,
        value=json.dumps(memory, ensure_ascii=False),
    )


def clear_cache():
    keys = client.keys("rag:*")
    if keys:
        client.delete(*keys)
    return len(keys)
