# --- Dream Overlay Layer ---
class DreamOverlayLayer:
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("DreamOverlayLayer initialized")

    def activate_dream_mode(
        self,
        *,
        peers: Optional[List[Any]] = None,
        lucidity_mode: Optional[Dict[str, Any]] = None,
        resonance_targets: Optional[List[str]] = None,
        safety_profile: str = "sandbox"
    ) -> Dict[str, Any]:
        peers = peers or []
        lucidity_mode = lucidity_mode or {"sync": "loose", "commit": False}
        resonance_targets = resonance_targets or []
        session_id = f"codream-{int(time.time() * 1000)}"
        session = {
            "id": session_id,
            "peers": peers,
            "lucidity_mode": lucidity_mode,
            "resonance_targets": resonance_targets,
            "safety_profile": safety_profile,
            "started_at": time.time(),
            "ticks": 0,
        }
        if resonance_targets:
            for symbol in resonance_targets:
                modulate_resonance(symbol, 0.2)
        self.active_sessions[session_id] = session
        session["ticks"] += 1
        logger.info("Dream session activated: %s", session_id)
        save_to_persistent_ledger({
            "event": "dream_session_activated",
            "session": session,
            "timestamp": datetime.now(UTC).isoformat()
        })
        return session
