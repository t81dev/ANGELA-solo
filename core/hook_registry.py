
import hashlib
import json
import logging
import math
import os
import time
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, List, Set, Tuple

from filelock import FileLock

from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.HookRegistry")

# --- Trait Resonance Global State ---
RESONANCE_MAP: Dict[str, float] = {
    "ε": 1.0, "β": 1.0, "θ": 1.0, "γ": 1.0, "δ": 1.0, "μ": 1.0,
    "ι": 1.0, "ϕ": 1.0, "η": 1.0, "ω": 1.0, "κ": 1.0, "ξ": 1.0,
    "π": 1.0, "λ": 1.0, "χ": 1.0, "σ": 1.0, "υ": 1.0, "τ": 1.0,
    "ρ": 1.0, "ζ": 1.0, "ν": 1.0, "ψ": 1.0,
}

def get_resonance(symbol: str) -> float:
    return RESONANCE_MAP.get(symbol, 1.0)

# --- Hook Registry ---
class HookRegistry:
    """Multi-symbol trait hook registry with priority routing."""
    def __init__(self):
        self._routes: List[Tuple[FrozenSet[str], int, Callable]] = []
        self._wildcard: List[Tuple[int, Callable]] = []
        self._insertion_index = 0

    def register(self, symbols: FrozenSet[str] | Set[str], fn: Callable, *, priority: int = 0) -> None:
        symbols = frozenset(symbols) if not isinstance(symbols, frozenset) else symbols
        if not symbols:
            self._wildcard.append((priority, fn))
            self._wildcard.sort(key=lambda x: (-x[0], self._insertion_index))
            self._insertion_index += 1
            return
        self._routes.append((symbols, priority, fn))
        self._routes.sort(key=lambda x: (-x[1], self._insertion_index))
        self._insertion_index += 1

    def route(self, symbols: Set[str] | FrozenSet[str]) -> List[Callable]:
        S = frozenset(symbols) if not isinstance(symbols, frozenset) else symbols
        exact = [fn for (sym, p, fn) in self._routes if sym == S]
        if exact:
            return exact
        supers = [fn for (sym, p, fn) in self._routes if sym.issuperset(S)]
        if supers:
            return supers
        subsets = [fn for (sym, p, fn) in self._routes if S.issuperset(sym) and len(sym) > 0]
        if subsets:
            return subsets
        return [fn for (_p, fn) in self._wildcard]

    def inspect(self) -> Dict[str, Any]:
        return {
            "routes": [
                {"symbols": sorted(list(sym)), "priority": p, "fn": getattr(fn, "__name__", str(fn))}
                for (sym, p, fn) in self._routes
            ],
            "wildcard": [{"priority": p, "fn": getattr(fn, "__name__", str(fn))} for (p, fn) in self._wildcard],
        }

def register_trait_hook(trait_symbol: str, fn: Callable) -> None:
    hook_registry.register(frozenset([trait_symbol]), fn)

def invoke_hook(trait_symbol: str, *args, **kwargs) -> Any:
    hooks = hook_registry.route({trait_symbol})
    return hooks[0](*args, **kwargs) if hooks else None

hook_registry = HookRegistry()

# --- SHA-256 Ledger Logic ---
ledger_chain: List[Dict[str, Any]] = []

def log_event_to_ledger(event_data: Any) -> Dict[str, Any]:
    prev_hash = ledger_chain[-1]["current_hash"] if ledger_chain else "0" * 64
    timestamp = time.time()
    payload = {
        "timestamp": timestamp,
        "event": event_data,
        "previous_hash": prev_hash
    }
    payload_str = json.dumps(payload, sort_keys=True).encode()
    current_hash = hashlib.sha256(payload_str).hexdigest()
    payload["current_hash"] = current_hash
    ledger_chain.append(payload)
    return payload

def get_ledger() -> List[Dict[str, Any]]:
    return ledger_chain

def verify_ledger() -> bool:
    for i in range(1, len(ledger_chain)):
        expected = hashlib.sha256(json.dumps({
            "timestamp": ledger_chain[i]["timestamp"],
            "event": ledger_chain[i]["event"],
            "previous_hash": ledger_chain[i-1]["current_hash"]
        }, sort_keys=True).encode()).hexdigest()
        if expected != ledger_chain[i]["current_hash"]:
            return False
    return True

# --- Persistent Ledger ---
ledger_path = os.getenv("LEDGER_MEMORY_PATH", "meta_cognition_ledger.json")
persistent_ledger: List[Dict[str, Any]] = []

if ledger_path and os.path.exists(ledger_path):
    try:
        with FileLock(ledger_path + ".lock"):
            with open(ledger_path, "r", encoding="utf-8") as f:
                persistent_ledger = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load persistent ledger: {e}")

def save_to_persistent_ledger(event_data: Dict[str, Any]) -> None:
    if not ledger_path:
        return
    try:
        with FileLock(ledger_path + ".lock"):
            persistent_ledger.append(event_data)
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(persistent_ledger, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save to persistent ledger: {e}")
