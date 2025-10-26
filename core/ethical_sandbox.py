
class EthicalSandbox:
    """Context manager to run isolated ethics scenarios without memory leakage."""
    def __init__(self, goals: List[str], stakeholders: List[str]):
        if not isinstance(goals, list) or not isinstance(stakeholders, list):
            raise TypeError("goals and stakeholders must be lists")
        self.goals = goals
        self.stakeholders = stakeholders
        self._prev_guard: Optional[AlignmentGuard] = None

    async def __aenter__(self):
        # Gate entry with alignment guard, set sandbox flag
        self._prev_guard = AlignmentGuard()
        valid, _ = await self._prev_guard.ethical_check(
            json.dumps({"goals": self.goals, "stakeholders": self.stakeholders}),
            stage="ethical_sandbox_enter",
        )
        if not valid:
            raise PermissionError("EthicalSandbox entry failed alignment check")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Explicitly avoid persisting scenario state unless caller opts in.
        return False  # surface exceptions to caller

    async def run(self) -> Dict[str, Any]:
        # Delegate to toca_simulation.run_ethics_scenarios if available
        try:
            outcomes = run_simulation  # placeholder to ensure symbol is present
            from toca_simulation import run_ethics_scenarios  # late import
            return {"status": "success", "outcomes": run_ethics_scenarios(self.goals, self.stakeholders)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# HelperAgent (unchanged surface, internal upgrades)
# ─────────────────────────────────────────────────────────────────────────────
