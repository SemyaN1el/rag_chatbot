from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import re
from config import *


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def get_all_chunks() -> list:
    client = QdrantClient(url=QDRANT_URL)

    # scroll() — получает все точки из коллекции постранично
    # limit=1000 — максимум за один запрос
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,   # нужен текст
        with_vectors=False   # векторы не нужны, только текст
    )
    return results


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:


    chunks = get_all_chunks()

    # В Qdrant LangChain хранит текст в поле "page_content"
    texts = [chunk.payload.get("page_content", "") for chunk in chunks]
    ids   = [chunk.id for chunk in chunks]

    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k * 2]  # берём с запасом для RRF
    # [(индекс, score), ...]

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

    # similarity_search_with_score возвращает [(Document, score), ...]
    vector_results = vectorstore.similarity_search_with_score(
        query,
        k=top_k * 2
    )

    #  текст документа - его позиция в векторном списке
    vector_ranked = {
        doc.page_content: rank
        for rank, (doc, score) in enumerate(vector_results)
    }

    K = 60  # стандарт
    rrf_scores = {}

    # BM25 вклад
    for rank, (idx, _) in enumerate(bm25_ranked):
        text = texts[idx]
        rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (K + rank + 1)

    #  векторный вклад
    for text, rank in vector_ranked.items():
        rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (K + rank + 1)

    final_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [
        {"text": text, "rrf_score": round(score, 4)}
        for text, score in final_results
    ]


if __name__ == "__main__":
    print("Тестируем гибридный поиск\n")

    queries = [
        "метод Ритца для функционала",
        "собственные значения Штурма-Лиувилля",
        "конечно-разностный метод Эйлера"
    ]

    for query in queries:
        print(f"{'='*60}")
        print(f"Запрос: '{query}'")
        print(f"{'='*60}")
        results = hybrid_search(query)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. RRF score: {r['rrf_score']}")
            print(f"   {r['text'][:150]}...")
        print()

