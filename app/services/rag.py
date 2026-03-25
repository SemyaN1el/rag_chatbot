from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from qdrant_client import QdrantClient
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


def get_vector_chain():
    """Строим цепочку векторного поиска"""
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
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


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

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    filled = prompt.format(context=context, question=question)
    answer = llm.invoke(filled).content

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
