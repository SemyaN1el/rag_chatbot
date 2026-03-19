from qdrant_client import QdrantClient

from langchain_qdrant import QdrantVectorStore

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from config import *

PROMPT_TEMPLATE = """Ты — помощник, отвечающий на вопросы по документу.
Используй ТОЛЬКО информацию из контекста ниже.
Если ответа нет в контексте — скажи "В документе нет информации по этому вопросу."
Отвечай на том же языке, на котором задан вопрос.

Контекст:
{context}

Вопрос: {question}

Ответ:"""


def build_chain():
    client = QdrantClient(url=QDRANT_URL)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # страницы
    )
    return chain


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