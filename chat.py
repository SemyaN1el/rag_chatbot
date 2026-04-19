from qdrant_client import QdrantClient

from langchain_qdrant import QdrantVectorStore

from langchain_huggingface import HuggingFaceEmbeddings

from app.services.rag import get_vector_chain
from config import *


def build_chain():
    return get_vector_chain()


def chat():
    print("Чат-бот по документу запущен.")
    print("   Введи 'выход' для остановки.\n")


    chain = build_chain()

    while True:
        question = input("Вопрос: ").strip()

        if question.lower() in ["выход", "exit", "quit"]:
            print("До свидания!")
            break

        if not question:
            continue

        result = chain.invoke({"query": question}) # основной вызов

        print(f"\nОтвет:\n{result['result']}")

        print(f"\nИсточники:")
        seen_pages = set()
        for doc in result["source_documents"]:
            page = doc.metadata.get("page", "?")
            if page not in seen_pages:
                seen_pages.add(page)
                print(f"  Стр. {page + 1}: {doc.page_content[:100]}...")
        print()


if __name__ == "__main__":
    chat()
