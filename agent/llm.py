"""LLM Client interface focused on Google Gemini API with automatic fallback and model discovery."""
import os
import re
from typing import Optional, List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    genai = None
    HAS_GEMINI = False


def get_available_gemini_models(api_key: Optional[str] = None) -> List[str]:
    """Retrieve list of supported generative models from Google Gemini."""
    default_fallbacks = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    key_to_use = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key_to_use or not HAS_GEMINI:
        return default_fallbacks

    try:
        genai.configure(api_key=key_to_use)
        discovered = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                discovered.append(name)
        
        # Sort so 2.5-flash and 2.5-pro are top priority
        priority = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
        sorted_models = [p for p in priority if p in discovered]
        for d in discovered:
            if d not in sorted_models:
                sorted_models.append(d)
        return sorted_models if sorted_models else default_fallbacks
    except Exception:
        return default_fallbacks


class GeminiLLMClient:
    """Manages interactions with Google Gemini API with smart model fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self._model = None
        self._initialized = False
        self.last_error = None

        if self.api_key and HAS_GEMINI:
            self._init_client(self.model_name)

    def _init_client(self, model_name: str):
        try:
            genai.configure(api_key=self.api_key)
            self.model_name = model_name
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": self.temperature},
            )
            self._initialized = True
            self.last_error = None
        except Exception as e:
            self._initialized = False
            self.last_error = str(e)

    def is_available(self) -> bool:
        return bool(self._initialized and self._model)

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate response from Gemini with automatic fallback on model 404s."""
        if not self.api_key:
            return self._offline_heuristic_generate(prompt)

        # Build full prompt
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"System Instructions:\n{system_instruction}\n\nTask:\n{prompt}"

        # Candidate models to try in case of 404 or deprecation
        candidates = [self.model_name]
        for fallback in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"]:
            if fallback not in candidates:
                candidates.append(fallback)

        last_err = None
        for candidate in candidates:
            try:
                self._init_client(candidate)
                if not self._model:
                    continue
                response = self._model.generate_content(full_prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                err_str = str(e)
                last_err = err_str
                # If it's a 404 / model not found error, try next candidate
                if "404" in err_str or "not found" in err_str.lower() or "is no longer available" in err_str.lower():
                    continue
                else:
                    return f"[Gemini API Call Failed: {err_str}]\n\nPlease verify your API key and quota."

        return f"[Gemini API Call Failed with all model candidates: {last_err}]\n\nFallback generation will be used."

    def _offline_heuristic_generate(self, prompt: str) -> str:
        """Heuristic offline generator for testing without an active API key."""
        return (
            "> [!NOTE]\n"
            "> **Offline Mode Active**: Google Gemini API key is not configured. "
            "To unlock full LLM intelligence, set `GEMINI_API_KEY` in your `.env` file or provide it in the UI/CLI.\n\n"
            "### Document Summary (Extractive Fallback)\n"
            "This document was processed using local structural parsing. Full synthesis and nuanced Q&A requires Gemini."
        )
