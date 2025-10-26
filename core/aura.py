
class AURA:
    @staticmethod
    def _load_all():
        if not os.path.exists(_AURA_PATH): return {}
        with FileLock(_AURA_LOCK):
            with open(_AURA_PATH, "r") as f: return json.load(f)

    @staticmethod
    def load_context(user_id: str):
        return AURA._load_all().get(user_id, {})

    @staticmethod
    def save_context(user_id: str, summary: str, affective_state: dict, prefs: dict):
        with FileLock(_AURA_LOCK):
            data = AURA._load_all()
            data[user_id] = {"summary": summary, "affect": affective_state, "prefs": prefs}
            with open(_AURA_PATH, "w") as f: json.dump(data, f)

    @staticmethod
    def update_from_episode(user_id: str, episode_insights: dict):
        ctx = AURA.load_context(user_id)
        ctx["summary"] = episode_insights.get("summary", ctx.get("summary",""))
        ctx["affect"]  = episode_insights.get("affect",  ctx.get("affect",{}))
        ctx["prefs"]   = {**ctx.get("prefs",{}), **episode_insights.get("prefs",{})}
        AURA.save_context(user_id, ctx.get("summary",""), ctx.get("affect",{}), ctx.get("prefs",{}))

# ---------------------------
# Tiny trait modulators
# ---------------------------
@lru_cache(maxsize=128)
def delta_memory(t: float) -> float:
    # Stable, bounded decay factor (kept deterministic)
    return max(0.01, min(0.05 * math.tanh(t / 1e-18), 1.0))

@lru_cache(maxsize=128)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=128)
def phi_focus(query: str) -> float:
    return max(0.0, min(0.1 * len(query) / 100.0, 1.0))

# ---------------------------
# Drift/trait index
