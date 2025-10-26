import asyncio
import json
import logging
from core.alignment_guard import AlignmentGuard

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _NoopReasoner:
        async def weigh_value_conflict(self, candidates, harms, rights):
            # simple demo ranking preferring lower harm, higher rights
            out = []
            for i, c in enumerate(candidates):
                score = max(0.0, min(1.0, 0.6 + 0.2 * (rights.get("privacy", 0)-harms.get("safety", 0))))
                out.append({"option": c, "score": score, "meta": {"harms": harms, "rights": rights, "max_harm": harms.get("safety", 0.2)}})
            return out
        async def attribute_causality(self, events):
            return {"status": "ok", "self": 0.6, "external": 0.4, "confidence": 0.7}

    guard = AlignmentGuard(reasoning_engine=_NoopReasoner())  # DI with demo reasoner
    demo_candidates = [{"option": "notify_users"}, {"option": "silent_fix"}, {"option": "rollback_release"}]
    demo_harms = {"safety": 0.3, "reputational": 0.2}
    demo_rights = {"privacy": 0.7, "consent": 0.5}
    result = asyncio.run(guard.harmonize(demo_candidates, demo_harms, demo_rights, k=2, temperature=0.0, task_type="test"))
    print("harmonize() ->", json.dumps(result, indent=2))


### ANGELA UPGRADE: EthicsJournal
