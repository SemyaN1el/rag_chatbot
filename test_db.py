import psycopg
import os

os.environ["PGPASSWORD"] = "ragpassword"

try:
    conn = psycopg.connect(
        "host=localhost port=5432 dbname=ragdb user=raguser password=ragpassword sslmode=disable"
    )
    print("Подключение успешно!")
    conn.close()
except Exception as e:
    print(f"Ошибка: {e}")