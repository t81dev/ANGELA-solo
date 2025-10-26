from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class NoopLLM:
    """LLM stub that returns a neutral score."""
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        _ = (prompt, model, temperature)
        return {"score": 0.8, "note": "noop-llm"}
