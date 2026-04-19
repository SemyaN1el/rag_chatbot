from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from app.services.llm import generate_text_from_prompt
from hybrid_search import hybrid_search
from config import *

PROMPT_TEMPLATE = """Ты — помощник, отвечающий на вопросы по документу.
Используй ТОЛЬКО информацию из контекста ниже.
Если ответа нет в контексте — скажи "В документе нет информации по этому вопросу."
Отвечай на том же языке, на котором задан вопрос.

Контекст:
{context}

Вопрос: {question}

Ответ:"""


class SimpleVectorChain:
    def invoke(self, payload: dict) -> dict:
        question = payload["query"]
        documents = retrieve_vector_documents(question)
        context = "\n\n".join(doc.page_content for doc in documents)
        answer = generate_answer_from_context(question, context)
        return {
            "result": answer,
            "source_documents": documents,
        }


def get_vectorstore() -> QdrantVectorStore:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    client = QdrantClient(url=QDRANT_URL)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    return vectorstore


def get_vector_chain() -> SimpleVectorChain:
    """Строим цепочку векторного поиска"""
    return SimpleVectorChain()


def retrieve_vector_documents(question: str):
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(question, k=TOP_K)


def generate_answer_from_context(question: str, context: str) -> str:
    filled_prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return generate_text_from_prompt(filled_prompt, temperature=LLM_TEMPERATURE)


def ask_vector(question: str) -> dict:
    """Ответ через векторный поиск"""
    chain = get_vector_chain()
    result = chain.invoke({"query": question})
    sources = [
        {
            "page": doc.metadata.get("page", 0) + 1,
            "text": doc.page_content[:200]
        }
        for doc in result["source_documents"]
    ]
    return {
        "answer": result["result"],
        "sources": sources,
        "search_type": "vector"
    }


def ask_hybrid(question: str) -> dict:
    """Ответ через гибридный поиск"""
    results = hybrid_search(question, top_k=TOP_K)
    context = "\n\n".join([r["text"] for r in results])
    answer = generate_answer_from_context(question, context)

    sources = [
        {
            "rrf_score": r["rrf_score"],
            "text": r["text"][:200]
        }
        for r in results
    ]
    return {
        "answer": answer,
        "sources": sources,
        "search_type": "hybrid"
    }
