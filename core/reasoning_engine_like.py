from typing import Protocol, List, Any, Dict

class ReasoningEngineLike(Protocol):
    async def weigh_value_conflict(self, candidates: List[Any], harms: Dict[str, float], rights: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Returns RankedOptions: list of dicts with at least:
            - option: Any
            - score: float in [0,1]
            - reasons: list[str] (optional)
            - meta: dict (optional)   # may include per-dimension harms/rights and max_harm
        """
    async def attribute_causality(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a causal attribution report with confidences."""
