import redis
import hashlib
import json
from config import REDIS_HOST, REDIS_PORT, REDIS_TTL

client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True  # возвращать строки а не байты
)


def make_key(question, search_type):
    ###Создаём уникальный ключ для кэша на основе вопроса и типа поиска.  MD5 чтобы ключ был коротким и без спецсимволов.

    raw = f"{search_type}:{question.lower().strip()}"
    return "rag:" + hashlib.md5(raw.encode()).hexdigest()


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


def clear_cache():
    keys = client.keys("rag:*")
    if keys:
        client.delete(*keys)
    return len(keys)