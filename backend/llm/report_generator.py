"""Real-provider clinical report generation with no mock fallback."""

import logging
from typing import Any, Dict

import httpx

from database.mongodb import settings
from llm.prompt import PromptBuilder

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate a report only through a configured real LLM provider."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        logger.info("Initializing LLM Report Generator with provider: %s", self.provider)

    async def generate_report(
        self,
        patient_data: Dict[str, Any],
        predictions: Dict[str, Any],
        explainability: Dict[str, Any],
        rag_context: list,
    ) -> Dict[str, Any]:
        """Generate a provider-backed report or raise a clear configuration/API error."""
        system_prompt = PromptBuilder.build_system_prompt()
        user_prompt = PromptBuilder.build_user_prompt(
            patient_data, predictions, explainability, rag_context
        )

        try:
            if self.provider == "openai":
                report_text = await self._call_openai(system_prompt, user_prompt)
            elif self.provider == "anthropic":
                report_text = await self._call_anthropic(system_prompt, user_prompt)
            elif self.provider == "ollama":
                report_text = await self._call_ollama(system_prompt, user_prompt)
            elif self.provider == "huggingface":
                report_text = await self._call_huggingface(system_prompt, user_prompt)
            else:
                raise RuntimeError(
                    "No real LLM provider is configured. Set LLM_PROVIDER=ollama and "
                    "OLLAMA_BASE_URL in backend/.env."
                )
        except Exception as error:
            logger.exception("LLM report generation failed")
            raise RuntimeError(f"Clinical report was not generated: {error}") from error

        if not report_text or not report_text.strip():
            raise RuntimeError("Clinical report provider returned an empty response.")

        references = [
            chunk.get("metadata", {}).get("source", "Unknown") for chunk in rag_context
        ]
        return {
            "generated_report": report_text.strip(),
            "retrieved_references": list(dict.fromkeys(references)),
            "disclaimer": PromptBuilder.MEDICAL_DISCLAIMER,
        }

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Use the official OpenAI Responses API; do not retain report content."""
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
            raise RuntimeError("OPENAI_API_KEY is missing in backend/.env.")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0)
        response = await client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=900,
            store=False,
        )
        if response.error:
            raise RuntimeError(response.error.message)
        return response.output_text

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is missing in backend/.env.")
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        return response.content[0].text

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call a locally running Ollama generate endpoint without a read timeout."""
        ollama_url = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        model_name = settings.OLLAMA_MODEL or "llama2"

        try:
            health = await httpx.AsyncClient().get(f"{ollama_url}/api/tags", timeout=10)
            health.raise_for_status()
        except Exception as exc:
            logger.error("Ollama is not reachable at %s: %s", ollama_url, exc)
            raise RuntimeError(
                "Ollama server is not running or unreachable. Start it with: brew install ollama && ollama serve"
            ) from exc

        payload = {
            "model": model_name,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            },
        }

        timeout = httpx.Timeout(None, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("Sending request to Ollama model '%s'...", model_name)
            response = await client.post(f"{ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

        if not data:
            raise RuntimeError("Ollama returned an empty response.")

        generated = data.get("response", "")
        if not generated or not generated.strip():
            raise RuntimeError(
                f"Ollama model '{model_name}' returned no text. Ensure the model is installed locally."
            )
        return generated.strip()

    async def _call_huggingface(self, system_prompt: str, user_prompt: str) -> str:
        """Fallback hook kept for compatibility, but this project uses local Ollama by default."""
        if not settings.HF_API_KEY or settings.HF_API_KEY.startswith("your_"):
            raise RuntimeError("HF_API_KEY is missing in backend/.env.")

        raise RuntimeError(
            "This project is configured for local Ollama. Set LLM_PROVIDER=ollama and use the local endpoint instead of Hugging Face."
        )


report_generator = ReportGenerator()
