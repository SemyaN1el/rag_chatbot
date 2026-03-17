from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore

import sys


from config import *

def ingest(pdf_path):
    print(f"Загрузка PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Страниц: {len(pages)}")

    print("\n Чанкинг. . .")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"Чанков: {len(chunks)}")
    print(f"Пример: '{chunks[0].page_content[:150]}")

    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL,
                                       encode_kwargs={"normalize_embeddings": True})
    vectorstore = QdrantVectorStore.from_documents(
        documents = chunks,
        embedding = embeddings,
        url = QDRANT_URL,
        collection_name = COLLECTION_NAME,
        force_recreate = True
    )

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/МУ_ЛР_ВИиОУ.pdf"
    ingest(pdf_path)