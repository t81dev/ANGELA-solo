
class ForkMerge:
    @staticmethod
    def auto_reconcile(forks: list, *, lattice=None, thresholds=None, policy: str = "min_risk"):
        if not forks:
            raise ValueError("No forks provided")
        deltas = compute_trait_deltas(forks, lattice)
        scored = [(f, score_fork(f, deltas, policy)) for f in forks]
        best, _ = min(scored, key=lambda x: x[1])
        return stitch_world(best, forks, deltas)


# --- Resonance-Weighted Branch Evaluation Patch ---
from meta_cognition import get_resonance
