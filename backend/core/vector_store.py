"""
Vector store wrapper around ChromaDB + LangChain.
"""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from core.config import settings

_embedding_fn = None
_store = None


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    return _embedding_fn


def get_vector_store(collection_name: str = "youtube_rag") -> Chroma:
    """Return (or create) the persistent Chroma vector store."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
