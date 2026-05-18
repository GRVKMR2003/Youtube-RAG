"""
LLM helper – returns a configured ChatOpenAI instance.
"""
from langchain_openai import ChatOpenAI
from core.config import settings


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=temperature,
        openai_api_key=settings.OPENAI_API_KEY,
    )
