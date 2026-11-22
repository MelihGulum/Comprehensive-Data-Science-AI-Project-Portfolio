from __future__ import annotations
import json, os, time
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class LLMUsage:
    provider: str; model: str; input_tokens: int = 0; output_tokens: int = 0
    latency_ms: float = 0.0; estimated_cost_usd: float = 0.0

class LLMClient:
    """Small provider boundary. Gemini is optional; tests use FixtureLLM."""
    def generate_json(self, system: str, payload: dict[str, Any]) -> tuple[dict[str, Any], LLMUsage]:
        raise NotImplementedError

class GeminiLLM(LLMClient):
    def __init__(self, model: str | None = None):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

    def generate_json(self, system: str, payload: dict[str, Any]):
        from google.genai import types
        t = time.perf_counter()
        r = self.client.models.generate_content(
            model=self.model,
            contents=json.dumps(payload, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        latency = (time.perf_counter() - t) * 1000
        data = json.loads(r.text or "{}")
        meta = getattr(r, "usage_metadata", None)
        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(meta, "candidates_token_count", 0) or 0
        usage = LLMUsage("gemini", self.model, input_tokens, output_tokens, latency, 0.0)
        return data, usage

class FixtureLLM(LLMClient):
    def __init__(self, response: dict[str, Any]): self.response=response
    def generate_json(self, system: str, payload: dict[str, Any]):
        return self.response, LLMUsage("fixture","fixture-model",10,10,1.0,0.0)

def default_llm() -> LLMClient | None:
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    return GeminiLLM()

def usage_dict(u: LLMUsage) -> dict[str, Any]: return asdict(u)
