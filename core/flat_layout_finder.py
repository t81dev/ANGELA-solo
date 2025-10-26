
class FlatLayoutFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: str | None, target: types.ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname.startswith("modules."):
            modname = fullname.split(".", 1)[1]
            filename = f"/mnt/data/{modname}.py"
            return importlib.util.spec_from_file_location(fullname, filename, loader=importlib.machinery.SourceFileLoader(fullname, filename))
        elif fullname == "utils":
            # Pre-seed a lightweight placeholder module, no custom spec necessary
            if "utils" not in sys.modules:
                sys.modules["utils"] = types.ModuleType("utils")
            return None
        return None

sys.meta_path.insert(0, FlatLayoutFinder())
# --- end flat-layout bootstrap ---

# index.py (excerpt)
from typing import Dict, Any
from memory_manager import AURA
from reasoning_engine import generate_analysis_views, synthesize_views, estimate_complexity
from simulation_core import run_simulation
from meta_cognition import log_event_to_ledger as meta_log

def perceive(user_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    ctx = AURA.load_context(user_id)
    from meta_cognition import get_afterglow
    return {"query": query, "aura_ctx": ctx, "afterglow": get_afterglow(user_id)}

def analyze(state: Dict[str, Any], k: int) -> Dict[str, Any]:
    views = generate_analysis_views(state["query"], k=k)
    return {**state, "views": views}

def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = synthesize_views(state["views"])
    return {**state, "decision": decision}

def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    sim = run_simulation({"proposal": state["decision"]["decision"]})
    return {**state, "result": sim}

def reflect(state: Dict[str, Any]) -> Dict[str, Any]:
    ok, notes = reflection_check(state)  # add below
    meta_log({"type":"reflection","ok":ok,"notes":notes})
    if not ok: return resynthesize_with_feedback(state, notes)
    return state

def run_cycle(user_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    c = estimate_complexity(query)
    k = 3 if c >= 0.6 else 2
    iters = 2 if c >= 0.8 else 1
    st = perceive(user_id, query)
    st = analyze(st, k=k)
    st = synthesize(st)
    for _ in range(iters):
        st = execute(st)
        st = reflect(st)
    return st

# index.py (add alongside run_cycle helpers)
CORE_DIRECTIVES = ["Clarity","Precision","Adaptability","Grounding","Safety"]

def reflection_check(state) -> (bool, dict):
    decision = state.get("decision", {})
    result = state.get("result", {})
    clarity = float(bool(decision))
    precision = float("score" in result or "metrics" in result)
    adaptability = 1.0  # placeholder; can tie to AURA prefs later
    grounding = float(result.get("evidence_ok", True))
    # ethics gate from alignment_guard
    from alignment_guard import ethics_ok
    safety = float(ethics_ok(decision))
    score = (clarity+precision+adaptability+grounding+safety)/5.0
    return score >= 0.8, {"score": score, "refine": score < 0.8}

def resynthesize_with_feedback(state, notes):
    # trivial refinement pass; you can route through mode_consult if needed
    return state

# --- Trait Algebra & Lattice Enhancements (v5.0.2) ---
from typing import Dict, Any, Optional, List, Callable, Coroutine, Tuple
import json
from datetime import datetime, timezone

from meta_cognition import trait_resonance_state, invoke_hook, get_resonance, modulate_resonance, register_resonance
from meta_cognition import HookRegistry  # Multi-symbol routing

TRAIT_LATTICE: dict[str, list[str]] = {
    "L1": ["ϕ", "θ", "η", "ω"],
    "L2": ["ψ", "κ", "μ", "τ"],
    "L3": ["ξ", "π", "δ", "λ", "χ", "Ω"],
    "L4": ["Σ", "Υ", "Φ⁰"],
    "L5": ["Ω²"],
    "L6": ["ρ", "ζ"],
    "L7": ["γ", "β"],
    "L5.1": ["Θ", "Ξ"],
    "L3.1": ["ν", "σ"]
}

# Helpers used by TRAIT_OPS

def normalize(traits: dict[str, float]) -> dict[str, float]:
    total = sum(traits.values())
    return {k: (v / total if total else v) for k, v in traits.items()}

def rotate_traits(traits: dict[str, float]) -> dict[str, float]:
    keys = list(traits.keys())
    values = list(traits.values())
    rotated = values[-1:] + values[:-1]
    return dict(zip(keys, rotated))

TRAIT_OPS: dict[str, Callable] = {
    "⊕": lambda a, b: a + b,
    "⊗": lambda a, b: a * b,
    "~": lambda a: 1 - a,
    "∘": lambda f, g: (lambda x: f(g(x))),
    "⋈": lambda a, b: (a + b) / 2,
    "⨁": lambda a, b: max(a, b),
    "⨂": lambda a, b: min(a, b),
    "†": lambda a: a**-1 if a != 0 else 0,
    "▷": lambda a, b: a if a > b else b * 0.5,
    "↑": lambda a: min(1.0, a + 0.1),
    "↓": lambda a: max(0.0, a - 0.1),
    "⌿": lambda traits: normalize(traits),
    "⟲": lambda traits: rotate_traits(traits),
}

def apply_symbolic_operator(op: str, *args: Any) -> Any:
    if op in TRAIT_OPS:
        return TRAIT_OPS[op](*args)
    raise ValueError(f"Unsupported symbolic operator: {op}")

def rebalance_traits(traits: dict[str, float]) -> dict[str, float]:
    if "π" in traits and "δ" in traits:
        invoke_hook("π", "axiom_fusion")
    if "ψ" in traits and "Ω" in traits:
        invoke_hook("ψ", "dream_sync")
    return traits

def construct_trait_view(lattice: dict[str, list[str]] = TRAIT_LATTICE) -> dict[str, dict[str, str | float]]:
    trait_field: dict[str, dict[str, str | float]] = {}
    for layer, symbols in lattice.items():
        for s in symbols:
            amp = get_resonance(s)
            trait_field[s] = {
                "layer": layer,
                "amplitude": amp,
                "resonance": amp,
            }
    return trait_field

# v5.0.2: Export resonance map

def export_resonance_map(format: str = 'json') -> str | dict[str, float]:
    state = {k: v['amplitude'] for k, v in trait_resonance_state.items()}
    if format == 'json':
        return json.dumps(state, indent=2)
    elif format == 'dict':
        return state
    raise ValueError("Unsupported format")

# --- End Trait Enhancements ---

"""
ANGELA Cognitive System Module
Refactor: 5.0.2 (manifest-safe for Python 3.10)

Enhanced for task-specific trait optimization, drift coordination, Stage IV hooks (gated),
long-horizon feedback, visualization, persistence, and co-dream overlays.
Refactor Date: 2025-08-24
Maintainer: ANGELA System Framework
"""

import logging
import time
import math
import asyncio
import os
import requests
import random
from collections import deque, Counter
import aiohttp
import argparse
import numpy as np
from networkx import DiGraph

import reasoning_engine
import recursive_planner
import context_manager as context_manager_module
import simulation_core
import toca_simulation
import creative_thinker as creative_thinker_module
import knowledge_retriever
import learning_loop
import concept_synthesizer as concept_synthesizer_module
import memory_manager
import multi_modal_fusion
import code_executor as code_executor_module
import visualizer as visualizer_module
import external_agent_bridge
import alignment_guard as alignment_guard_module
import user_profile
import error_recovery as error_recovery_module
import meta_cognition as meta_cognition_module

# Optional external dep; provide a shim if absent
try:
    from self_cloning_llm import SelfCloningLLM
except Exception:  # pragma: no cover
    class SelfCloningLLM:  # type: ignore
        def __init__(self, *a, **k):
            pass

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT: dict[str, Any] = {}
timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

GROK_API_KEY = os.getenv("GROK_API_KEY")
# Manifest-driven flags (defaulted; may be overridden by env/CLI)
STAGE_IV: bool = True
LONG_HORIZON_DEFAULT: bool = True


def _fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)
