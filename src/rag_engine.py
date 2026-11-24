import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure module import resolution from any working directory
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from embeddings import get_embedding_function
from llm_manager import MultiProviderLLMManager
from cache import FAQCache

load_dotenv()

SYSTEM_PROMPT = """You are an official ISRO Recruitment FAQ Assistant.
Answer the candidate's question strictly using ONLY the retrieved notification context below.
Rules:
1. Cite the Advertisement Number (advt_no) and section/source for every claim.
2. Do NOT extrapolate, guess, or rely on outside knowledge from other recruitment cycles.
3. If the answer is not explicitly stated in the provided context, reply exactly:
   "This information is not specified in the uploaded ISRO recruitment notification."

Context:
{context}
"""


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents with Advertisement number and header metadata for grounding."""
    formatted = []
    for d in docs:
        advt = d.metadata.get("advt_no", "Unknown Advt")
        source = d.metadata.get("source", "Notice")
        sec = d.metadata.get("H2") or d.metadata.get("H1") or "General"
        formatted.append(f"[{advt} | {source} | {sec}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def bm25_tokenizer(text: str) -> List[str]:
    """Tokenize text into alphanumeric words and post codes (e.g., BE001, BE-003, 65%)."""
    return re.findall(r"[a-zA-Z0-9_\-]+", text.lower())


class ISROFAQEngine:
    """Production-grade RAG FAQ Engine for ISRO recruitment notifications.

    Features:
    - Zero-token local HuggingFace embeddings (`all-MiniLM-L6-v2`)
    - Layout-aware Hybrid Ensemble Retrieval (Regex-tokenized BM25 + Chroma vector search)
    - Multi-provider LLM fallback hierarchy: Groq -> Google Gemini -> OpenAI
    - SQLite FAQ response cache for high-frequency candidate questions
    """

    def __init__(
        self,
        persist_dir: str = "./isro_vectordb",
        mode: str = "hybrid",
        k: int = 6,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        enable_cache: Optional[bool] = None,
        cache_db: str = "./isro_faq_cache.sqlite",
    ):
        self.persist_dir = persist_dir
        self.mode = mode
        self.k = k
        self.embedding_provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", "huggingface")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        if enable_cache is None:
            self.enable_cache = os.getenv("ENABLE_FAQ_CACHE", "1").strip() in ("1", "true", "True")
        else:
            self.enable_cache = enable_cache

        self.cache = FAQCache(db_path=cache_db) if self.enable_cache else None
        self.llm_manager = MultiProviderLLMManager()

        self._init_retriever()

    def _init_retriever(self):
        embeddings = get_embedding_function(
            provider=self.embedding_provider,
            model_name=self.embedding_model,
        )
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
        )

        raw_data = self.vectorstore.get()
        documents = raw_data.get("documents", [])
        metadatas = raw_data.get("metadatas", [])

        if not documents:
            all_docs = []
        else:
            all_docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(documents, metadatas)
            ]

        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        if all_docs:
            self.bm25_retriever = BM25Retriever.from_documents(
                all_docs,
                preprocess_func=bm25_tokenizer,
            )
            self.bm25_retriever.k = self.k
        else:
            self.bm25_retriever = None

        if self.mode == "sparse" and self.bm25_retriever:
            self.retriever = self.bm25_retriever
        elif self.mode == "dense" or not self.bm25_retriever:
            self.retriever = self.vector_retriever
        else:
            self.retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[0.5, 0.5],
            )

    def query(self, question: str, bypass_cache: bool = False) -> Dict[str, Any]:
        """Execute grounded RAG query with FAQ caching and multi-provider LLM fallback."""
        # 1. Check SQLite FAQ Cache
        if self.cache and not bypass_cache:
            cached_result = self.cache.get(question)
            if cached_result:
                docs = [
                    Document(
                        page_content=s.get("page_content", ""),
                        metadata=s.get("metadata", {}),
                    )
                    for s in cached_result.get("sources", [])
                ]
                return {
                    "question": question,
                    "answer": cached_result["answer"],
                    "retrieved_docs": docs,
                    "provider": "SQLite Cache (0 Tokens Used)",
                    "model_used": cached_result["model_used"],
                    "cached": True,
                    "hit_count": cached_result["hit_count"],
                }

        # 2. Retrieve relevant context
        docs = self.retriever.invoke(question) if self.retriever else []
        formatted_context = format_docs(docs)

        # 3. Build Prompt Messages
        system_content = SYSTEM_PROMPT.format(context=formatted_context)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question),
        ]

        # 4. Generate Answer via Multi-Provider LLM Hierarchy (Groq -> Gemini -> OpenAI)
        answer, provider_name, model_name = self.llm_manager.invoke_with_fallback(messages)

        # 5. Store in SQLite FAQ Cache
        if self.cache:
            serializable_sources = [
                {
                    "page_content": d.page_content,
                    "metadata": d.metadata,
                }
                for d in docs
            ]
            self.cache.set(
                query=question,
                answer=answer,
                sources=serializable_sources,
                model_used=f"{provider_name} ({model_name})",
            )

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": docs,
            "provider": provider_name,
            "model_used": model_name,
            "cached": False,
            "hit_count": 1,
        }

    def invoke(self, question: str) -> str:
        """Compatibility wrapper that returns only the answer string."""
        result = self.query(question)
        return result["answer"]


class RAGChainWrapper:
    """Wrapper to maintain backwards compatibility with existing chain.invoke() interface."""

    def __init__(self, engine: ISROFAQEngine):
        self.engine = engine

    def invoke(self, question: str) -> str:
        return self.engine.invoke(question)


def build_rag_chain(
    persist_dir: str = "./isro_vectordb",
    mode: str = "hybrid",
    k: int = 6,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    enable_cache: Optional[bool] = None,
):
    """Backwards-compatible builder function returning (chain_wrapper, retriever)."""
    engine = ISROFAQEngine(
        persist_dir=persist_dir,
        mode=mode,
        k=k,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        enable_cache=enable_cache,
    )
    return RAGChainWrapper(engine), engine.retriever
