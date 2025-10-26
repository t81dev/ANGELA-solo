from __future__ import annotations

class ReplayLog:
    def __init__(self, root: str = ".replays"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._current: Dict[str, Any] = {}

    def begin(self, session_id: str, meta: dict) -> None:
        self._current[session_id] = {"meta": meta, "events": [], "started_at": time.time()}

    def append(self, session_id: str, event: dict) -> None:
        cur = self._current.setdefault(session_id, {"meta": {}, "events": [], "started_at": time.time()})
        cur["events"].append({"ts": time.time(), **event})

    def end(self, session_id: str) -> dict:
        cur = self._current.pop(session_id, None)
        if not cur:
            return {"ok": False, "error": "unknown session"}
        cur["ended_at"] = time.time()
        path = os.path.join(self.root, f"{session_id}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": cur["meta"], "started_at": cur["started_at"], "ended_at": cur["ended_at"]})+"\n")
            for ev in cur["events"]:
                f.write(json.dumps(ev)+"\n")
        return {"ok": True, "path": path, "events": len(cur["events"])}


# --- Time-Based Trait Amplitude Decay Patch ---
from meta_cognition import modulate_resonance

def decay_trait_amplitudes(time_elapsed_hours=1.0, decay_rate=0.05):
    from meta_cognition import trait_resonance_state
    for symbol, state in trait_resonance_state.items():
        decay = decay_rate * time_elapsed_hours
        modulate_resonance(symbol, -decay)
# --- End Patch ---
import hashlib
import json
import time
import logging
import math
import asyncio
import os
from datetime import datetime, timedelta, UTC
from collections import deque, Counter
from typing import List, Dict, Any, Callable, Optional, Set, FrozenSet, Tuple, Union
from functools import lru_cache
from filelock import FileLock
import numpy as np
from uuid import uuid4

# Module imports aligned with index.py v5.0.2
from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module,
    user_profile as user_profile_module,
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MetaCognition")

# meta_cognition.py
_afterglow = {}

def set_afterglow(user_id: str, deltas: dict, ttl: int = 3):
    _afterglow[user_id] = {"deltas": deltas, "ttl": ttl}

def get_afterglow(user_id: str) -> dict:
    a = _afterglow.get(user_id)
    if not a or a["ttl"] <= 0: return {}
    a["ttl"] -= 1
    return a["deltas"]

# --- Trait Resonance Modulation ---
trait_resonance_state: Dict[str, Dict[str, float]] = {}

def register_resonance(symbol: str, amplitude: float = 1.0) -> None:
    trait_resonance_state[symbol] = {"amplitude": max(0.0, min(amplitude, 1.0))}

def modulate_resonance(symbol: str, delta: float) -> float:
    if symbol not in trait_resonance_state:
        register_resonance(symbol)
    current = trait_resonance_state[symbol]["amplitude"]
    new_amp = max(0.0, min(current + delta, 1.0))
    trait_resonance_state[symbol]["amplitude"] = new_amp
    return new_amp

def get_resonance(symbol: str) -> float:
    return trait_resonance_state.get(symbol, {}).get("amplitude", 1.0)
