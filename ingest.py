from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore

import sys
from pathlib import Path


from config import *


def resolve_pdf_path(cli_arg: str | None) -> Path:
    if cli_arg:
        pdf_path = Path(cli_arg)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Ожидался PDF-файл, получено: {pdf_path}")
        return pdf_path

    data_dir = Path("data")
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "В папке data не найдено PDF-файлов. Передай путь явно: python ingest.py data/your_file.pdf"
        )

    if len(pdf_files) > 1:
        available = "\n".join(f" - {pdf}" for pdf in pdf_files)
        raise ValueError(
            "Найдено несколько PDF-файлов. Укажи нужный файл явно:\n"
            f"{available}"
        )

    return pdf_files[0]


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
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else None
    ingest(str(resolve_pdf_path(pdf_arg)))
