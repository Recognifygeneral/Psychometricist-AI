"""Shared LLM / embedding client factories.

Every module that needs an OpenAI chat or embedding model should import
from here instead of constructing its own client, ensuring consistent
model selection, temperature, and timeout configuration.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.settings import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME

# Default network timeout (seconds) for all OpenAI requests.
_REQUEST_TIMEOUT: int = 30


def get_chat_llm(
    *,
    temperature: float = 0.7,
    request_timeout: int = _REQUEST_TIMEOUT,
) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=temperature,
        request_timeout=request_timeout,
    )


def get_embeddings_model(
    *,
    request_timeout: int = _REQUEST_TIMEOUT,
) -> OpenAIEmbeddings:
    """Return a configured OpenAIEmbeddings instance."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        request_timeout=request_timeout,
    )
