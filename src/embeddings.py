import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_embedding_function(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """Factory to retrieve the specified embedding function.

    Defaults to free local HuggingFace embeddings (`all-MiniLM-L6-v2`) which uses
    zero API tokens and runs fast on CPU. Supports OpenAI embeddings as an option.
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "huggingface")).strip().lower()

    if provider in ("huggingface", "local", "hf"):
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=model_name)
            except ImportError:
                raise ImportError(
                    "To use local embeddings, please install langchain-huggingface or sentence-transformers: "
                    "`pip install langchain-huggingface sentence-transformers`"
                )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model_name)

    else:
        # Default fallback to HuggingFace
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
