
class ConstitutionSync:
    """Synchronize constitutional values among agents with τ audit + ceiling adherence."""
    def __init__(self, max_harm_ceiling: float = 1.0):
        if not (0.0 <= float(max_harm_ceiling) <= 1.0):
            raise ValueError("max_harm_ceiling must be in [0,1]")
        self.max_harm_ceiling = float(max_harm_ceiling)

    async def sync_values(self, peer_agent: HelperAgent, drift_data: Optional[Dict[str, Any]] = None, task_type: str = "") -> bool:
        if not isinstance(peer_agent, HelperAgent): raise TypeError("peer_agent must be a HelperAgent instance")
        if drift_data is not None and not isinstance(drift_data, dict): raise TypeError("drift_data must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        # τ: enforce harm ceiling on proposed constitution updates
        if drift_data:
            harm = float(drift_data.get("harm", 0.0))
            if harm > self.max_harm_ceiling:
                return False

        try:
            if drift_data and not MetaCognition().validate_drift(drift_data):
                return False
            # apply
            peer_agent.meta.constitution.update(drift_data or {})
            # audit
            mm = getattr(peer_agent.meta, "memory_manager", None)
            if mm:
                await mm.store(
                    query=f"ConstitutionSync::{peer_agent.name}::{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps({"applied": list((drift_data or {}).keys())}),
                    layer="Ethics",
                    intent="constitution_sync_audit",
                    task_type=task_type,
                )
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder Reasoner (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
