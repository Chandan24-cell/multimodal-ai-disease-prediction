# backend/llm/report_generator.py
import os
import json
import logging
import httpx
from typing import Dict, Any, Optional

from database.mongodb import settings
from llm.prompt import PromptBuilder

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Orchestrates LLM calls to generate structured medical reports.
    Supports OpenAI, Anthropic, and local Ollama based on .env configuration.
    """
    def __init__(self):
        self.provider = getattr(settings, "LLM_PROVIDER", "ollama").lower()
        logger.info(f"Initializing LLM Report Generator with provider: {self.provider}")

    async def generate_report(
        self,
        patient_data: Dict[str, Any],
        predictions: Dict[str, Any],
        explainability: Dict[str, Any],
        rag_context: list
    ) -> Dict[str, Any]:
        """
        Main entry point to generate the report.
        Returns a dictionary matching the ReportDB schema.
        """
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
            else:
                logger.warning(f"Unknown provider '{self.provider}'. Falling back to mock report.")
                report_text = self._generate_mock_report()

            # Extract references from RAG context for the database schema
            references = [chunk["metadata"].get("source", "Unknown") for chunk in rag_context]

            return {
                "generated_report": report_text,
                "retrieved_references": list(set(references)), # Unique sources
                "disclaimer": PromptBuilder.MEDICAL_DISCLAIMER
            }

        except Exception as e:
            logger.error(f"LLM Report Generation failed: {e}")
            return {
                "generated_report": self._generate_mock_report(),
                "retrieved_references": ["Fallback due to LLM error"],
                "disclaimer": PromptBuilder.MEDICAL_DISCLAIMER
            }

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo", # Or gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3 # Low temperature for factual, deterministic medical output
        )
        return response.choices[0].message.content

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        response = await client.messages.create(
            model="claude-3-haiku-20240307", # Fast, cost-effective
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return response.content[0].text

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Calls a local Ollama instance (e.g., llama3, mistral)."""
        ollama_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = "llama3" # Default, can be made configurable
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "options": {"temperature": 0.3}
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            
            # Ollama streams responses, but for simplicity we parse the final JSON
            # If streaming is needed, we'd iterate over response.aiter_lines()
            lines = response.text.strip().split('\n')
            full_response = ""
            for line in lines:
                if line:
                    data = json.loads(line)
                    full_response += data.get("message", {}).get("content", "")
                    
            return full_response

    def _generate_mock_report(self) -> str:
        """Fallback report if LLM APIs fail or are not configured."""
        return f"""## 📋 Patient Summary
Patient data processed successfully.

## 🔬 AI Diagnostic Findings
Multimodal fusion analysis complete. (Mock output: LLM provider not configured or failed).

## 🧠 Model Explainability
- SHAP and Attention maps were generated and are available in the dashboard.

## 📚 Clinical References
- Fallback reference document.

## ⚠️ Disclaimer
{PromptBuilder.MEDICAL_DISCLAIMER}
"""

# Global instance
report_generator = ReportGenerator()