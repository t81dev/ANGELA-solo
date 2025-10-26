from typing import Protocol, Dict, Any

class LLMClient(Protocol):
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        """Return a dict with fields like {"score": float, ...} or arbitrary JSON."""
