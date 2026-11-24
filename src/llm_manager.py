import os
import re
import sys
import logging
from typing import Tuple, Optional, Any, List
from pathlib import Path
from dotenv import load_dotenv

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()
logger = logging.getLogger("isro_rag.llm_manager")


class MultiProviderLLMManager:
    """Manages multi-provider LLM execution with automatic fallback:

    Primary: Groq (high-speed free tier - up to 14,400 req/day)
    Fallback 1: Google Gemini (gemini-3.5-flash-lite - free tier, automatic fallback on 429)
    Fallback 2: OpenAI (gpt-4o-mini - secondary safety net)
    """

    def __init__(
        self,
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        ).strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    def get_provider_status(self) -> dict:
        """Return the availability status of each configured provider."""
        return {
            "groq": {
                "configured": bool(self.groq_api_key and not self.groq_api_key.startswith("your_")),
                "model": self.groq_model,
                "role": "Primary LLM",
            },
            "gemini": {
                "configured": bool(self.gemini_api_key and not self.gemini_api_key.startswith("your_")),
                "model": self.gemini_model,
                "role": "Automatic Fallback (429 Retry)",
            },
            "openai": {
                "configured": bool(self.openai_api_key and not self.openai_api_key.startswith("your_")),
                "model": self.openai_model,
                "role": "Secondary Fallback",
            },
        }

    def _build_groq_llm(self, model: Optional[str] = None) -> Optional[BaseChatModel]:
        if not self.groq_api_key or self.groq_api_key.startswith("your_"):
            return None
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model or self.groq_model,
                temperature=0.0,
                groq_api_key=self.groq_api_key,
                max_retries=1,
            )
        except Exception as e:
            logger.warning(f"Failed to instantiate Groq LLM: {e}")
            return None

    def _build_gemini_llm(self) -> Optional[BaseChatModel]:
        if not self.gemini_api_key or self.gemini_api_key.startswith("your_"):
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=0.0,
                google_api_key=self.gemini_api_key,
                max_retries=1,
            )
        except Exception as e:
            logger.warning(f"Failed to instantiate Gemini LLM: {e}")
            return None

    def _build_openai_llm(self) -> Optional[BaseChatModel]:
        if not self.openai_api_key or self.openai_api_key.startswith("your_"):
            return None
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.openai_model,
                temperature=0.0,
                api_key=self.openai_api_key,
                max_retries=1,
            )
        except Exception as e:
            logger.warning(f"Failed to instantiate OpenAI LLM: {e}")
            return None

    def invoke_with_fallback(self, messages: Any) -> Tuple[str, str, str]:
        """Invoke LLMs with fallback hierarchy:

        1. Groq (primary)
        2. Gemini (automatic fallback on 429 rate limit or error)
        3. OpenAI (if available)

        Returns:
            Tuple of (answer_text, provider_name, model_name)
        """
        errors = []

        # 1. Try Groq (Primary)
        groq_llm = self._build_groq_llm()
        if groq_llm:
            try:
                response = groq_llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                return content, "Groq (Primary)", self.groq_model
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Groq primary invocation failed: {err_str}")
                errors.append(f"Groq ({self.groq_model}): {err_str}")

                # If primary model is not found, dynamically try active alternative Groq models
                if "model_not_found" in err_str:
                    for alt_groq in ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen/qwen3.8-27b"]:
                        if alt_groq != self.groq_model:
                            try:
                                alt_llm = self._build_groq_llm(model=alt_groq)
                                if alt_llm:
                                    response = alt_llm.invoke(messages)
                                    content = response.content if hasattr(response, "content") else str(response)
                                    return content, f"Groq ({alt_groq})", alt_groq
                            except Exception:
                                pass

        # 2. Try Gemini (Automatic Fallback on 429 rate limit or Groq failure)
        gemini_llm = self._build_gemini_llm()
        if gemini_llm:
            try:
                response = gemini_llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                return content, "Google Gemini (Fallback)", self.gemini_model
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini fallback failed: {err_str}. Attempting OpenAI fallback...")
                errors.append(f"Gemini ({self.gemini_model}): {err_str}")

        # 3. Try OpenAI (Secondary Fallback)
        openai_llm = self._build_openai_llm()
        if openai_llm:
            try:
                response = openai_llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                return content, "OpenAI (Secondary Fallback)", self.openai_model
            except Exception as e:
                err_str = str(e)
                logger.error(f"OpenAI invocation failed: {err_str}")
                errors.append(f"OpenAI: {err_str}")

        # If all providers failed or none configured
        if not errors:
            raise ValueError(
                "No LLM provider is configured. Please provide at least one valid API key in `.env` "
                "(GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY)."
            )
        else:
            raise RuntimeError(
                f"All LLM providers failed:\n" + "\n".join(f"- {err}" for err in errors)
            )
