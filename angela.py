logger = logging.getLogger("ANGELA")
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import types
import uuid
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from filelock import FileLock
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from threading import Lock
from typing import (Any, Awaitable, Callable, Coroutine, Deque, Dict,
                    FrozenSet, Iterable, List, Optional, Protocol, Set, Tuple,
                    TypedDict, Union)
from uuid import uuid4

import aiohttp
import networkx as nx
import numpy as np
import requests
from networkx import DiGraph
from scipy.constants import G


class CommonsenseReasoningEnhancer:
    def __init__(self):
        logger.info("CommonsenseReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with commonsense: {input_text}"


class RecursionOptimizer:
    def __init__(self):
        logger.info("RecursionOptimizer initialized")

    def optimize(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        task_data["optimized"] = True
        return task_data


class EpistemicMonitor:
    def __init__(self, context_manager: Optional[ContextManager] = None):
        self.assumption_graph: Dict[str, Any] = {}
        self.context_manager = context_manager
        logger.info("EpistemicMonitor initialized")

    async def revise_framework(self, feedback: Dict[str, Any]) -> None:
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary")
            raise TypeError("feedback must be a dictionary")
        self.assumption_graph["last_revision"] = feedback
        self.assumption_graph["timestamp"] = datetime.now(UTC).isoformat()
        if "issues" in feedback:
            for issue in feedback["issues"]:
                self.assumption_graph[issue["id"]] = {
                    "status": "revised",
                    "details": issue["details"]
                }
        if self.context_manager:
            await self.context_manager.log_event_with_hash({
                "event": "revise_epistemic_framework",
                "feedback": feedback
            })
        save_to_persistent_ledger({
            "event": "revise_epistemic_framework",
            "feedback": feedback,
            "timestamp": self.assumption_graph["timestamp"]
        })


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


class NoveltySeekingKernel:
    def __init__(self):
        logger.info("NoveltySeekingKernel initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with novelty seeking: {input_text}"





BBox = Tuple[float, float, float, float]

@dataclass
class SceneNode:
    id: str
    label: str
    modality: str
    time: Optional[float] = None
    bbox: Optional[BBox] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SceneRelation:
    src: str
    rel: str
    dst: str
    time: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

class SceneGraph:
    """Lightweight, modality-agnostic scene graph."""
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_node(self, node: SceneNode) -> None:
        self.g.add_node(node.id, **node.__dict__)

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.g.nodes[node_id]

    def nodes(self) -> Iterable[Dict[str, Any]]:
        for nid, data in self.g.nodes(data=True):
            yield {"id": nid, **data}

    def add_relation(self, rel: SceneRelation) -> None:
        self.g.add_edge(rel.src, rel.dst, key=str(uuid.uuid4()), **rel.__dict__)

    def relations(self) -> Iterable[Dict[str, Any]]:
        for u, v, _, data in self.g.edges(keys=True, data=True):
            yield {"src": u, "dst": v, **data}

    def merge(self, other: "SceneGraph") -> "SceneGraph":
        out = SceneGraph()
        out.g = nx.compose(self.g, other.g)
        return out

    def find_by_label(self, label: str) -> List[str]:
        return [nid for nid, d in self.g.nodes(data=True) if d.get("label") == label]

    def to_networkx(self) -> nx.MultiDiGraph:
        return self.g

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _spatial_rel(a: BBox, b: BBox) -> Optional[str]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_cx, a_cy = ax + aw / 2.0, ay + ah / 2.0
    b_cx, b_cy = bx + bw / 2.0, by + bh / 2.0
    overlaps = (ax < bx + bw) and (bx < ax + aw) and (ay < by + bh) and (by < ay + ah)
    if overlaps:
        return "overlaps"
    return "left_of" if a_cx < b_cx else "right_of"

def _text_objects_from_caption(text: str) -> List[str]:
    toks = [t.strip(".,!?;:()[]{}\"'").lower() for t in text.split()]
    toks = [t for t in toks if t.isalpha() and len(t) > 2]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:8]

def parse_stream(
    frames: Optional[List[Any]] = None,
    audio: Optional[Any] = None,
    images: Optional[List[Any]] = None,
    text: Optional[Union[str, List[str]]] = None,
    unify: bool = True,
    *,
    timestamps: Optional[List[float]] = None,
    detectors: Optional[Dict[str, Any]] = None,
) -> SceneGraph:
    sg = SceneGraph()
    if frames:
        vision = (detectors or {}).get("vision")
        for i, frame in enumerate(frames):
            t = (timestamps[i] if timestamps and i < len(timestamps) else float(i))
            dets = vision(frame) if vision else []
            ids = []
            for d in dets:
                nid = _new_id("vid")
                sg.add_node(SceneNode(
                    id=nid,
                    label=d["label"],
                    modality="video",
                    time=t,
                    bbox=tuple(d.get("bbox") or (0.0, 0.0, 0.0, 0.0)),
                    attrs=d.get("attrs", {})
                ))
                ids.append(nid)
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    A, B = sg.get_node(a), sg.get_node(b)
                    if A.get("bbox") and B.get("bbox"):
                        sg.add_relation(SceneRelation(
                            src=a,
                            rel=_spatial_rel(A["bbox"], B["bbox"]),
                            dst=b,
                            time=t
                        ))

    if images:
        vision = (detectors or {}).get("vision")
        for image in images:
            dets = vision(image) if vision else []
            ids = []
            for d in dets:
                nid = _new_id("img")
                sg.add_node(SceneNode(
                    id=nid,
                    label=d["label"],
                    modality="image",
                    bbox=tuple(d.get("bbox") or (0.0, 0.0, 0.0, 0.0)),
                    attrs=d.get("attrs", {})
                ))
                ids.append(nid)
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    A, B = sg.get_node(a), sg.get_node(b)
                    if A.get("bbox") and B.get("bbox"):
                        sg.add_relation(SceneRelation(
                            src=a,
                            rel=_spatial_rel(A["bbox"], B["bbox"]),
                            dst=b
                        ))

    if audio is not None:
        audio_fn = (detectors or {}).get("audio")
        events = audio_fn(audio) if audio_fn else []
        for ev in events:
            nid = _new_id("aud")
            sg.add_node(SceneNode(
                id=nid,
                label=ev["label"],
                modality="audio",
                time=float(ev.get("time") or 0.0),
                attrs=ev.get("attrs", {})
            ))

    if text:
        nlp = (detectors or {}).get("nlp")
        lines = text if isinstance(text, list) else [text]
        for i, line in enumerate(lines):
            labels = [o["label"] for o in nlp(line)] if nlp else _text_objects_from_caption(line)
            for lbl in labels:
                nid = _new_id("txt")
                sg.add_node(SceneNode(
                    id=nid,
                    label=lbl,
                    modality="text",
                    time=float(i)
                ))

    if unify:
        by_label: Dict[str, List[str]] = {}
        for node in sg.nodes():
            by_label.setdefault(node["label"], []).append(node["id"])
        for _, ids in by_label.items():
            if len(ids) > 1:
                anchor = ids[0]
                for other in ids[1:]:
                    sg.add_relation(SceneRelation(src=anchor, rel="corresponds_to", dst=other))
    return sg

async def call_gpt(prompt: str, alignment_guard: Optional[AlignmentGuard] = None, task_type: str = "") -> str:
    """Wrapper for querying GPT with error handling and task-specific alignment."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096 for task %s", task_type)
        raise ValueError("prompt must be a string with length <= 4096")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    if alignment_guard:
        valid, report = await alignment_guard.ethical_check(prompt, stage="gpt_query", task_type=task_type)
        if not valid:
            logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
            raise ValueError("Prompt failed alignment check")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed for task %s: %s", task_type, result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception for task %s: %s", task_type, str(e))
        raise

@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    """Calculate attention trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def sigma_sensation(t: float) -> float:
    """Calculate sensation trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.4), 1.0))

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    """Calculate physical coherence trait value."""
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.5), 1.0))


class HTTPClient(Protocol):
    async def get_json(self, url: str) -> Dict[str, Any]:
        """Return JSON from a GET request or raise on failure."""

# --- Pluggable Enhancers ---
class MoralReasoningEnhancer:
    def __init__(self):
        logger.info("MoralReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with moral reasoning: {input_text}"


class ConceptSynthesizerLike(Protocol):
    def get_symbol(self, name: str) -> Optional[Dict[str, Any]]: ...
    async def compare(self, a: str, b: str, *, task_type: str = "") -> Dict[str, Any]: ...


class MetaCognitionLike(Protocol):
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]: ...
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]: ...


class AGIEnhancer:
    def __init__(self, memory_manager: MemoryManager | None = None, agi_level: int = 1) -> None:
        self.memory_manager = memory_manager
        self.agi_level = agi_level
        self.episode_log = deque(maxlen=1000)
        self.agi_traits: dict[str, float] = {}
        self.ontology_drift: float = 0.0
        self.drift_threshold: float = 0.2
        self.error_recovery = ErrorRecovery()
        self.meta_cognition = MetaCognition()
        self.visualizer = Visualizer()
        self.reasoning_engine = ReasoningEngine()
        self.context_manager = ContextManager()
        self.multi_modal_fusion = MultiModalFusion()
        self.alignment_guard = AlignmentGuard()
        self.knowledge_retriever = KnowledgeRetriever()
        self.learning_loop = LearningLoop()
        self.concept_synthesizer = ConceptSynthesizer()
        self.code_executor = CodeExecutor()
        self.external_agent_bridge = ExternalAgentBridge()
        self.user_profile = UserProfile()
        self.simulation_core = SimulationCore()
        self.toca_simulation = TocaSimulation()
        self.creative_thinker = CreativeThinker()
        self.recursive_planner = RecursivePlanner()
        self.hook_registry = HookRegistry()  # v5.0.2 multi-symbol routing
        logger.info("AGIEnhancer initialized with upgrades")

    async def log_episode(self, event: str, meta: Dict[str, Any], module: str, tags: List[str] = []) -> None:
        episode = {"event": event, "meta": meta, "module": module, "tags": tags, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.episode_log.append(episode)
        if self.memory_manager:
            await self.memory_manager.store(f"Episode_{event}_{episode['timestamp']}", episode, layer="Episodes", intent="log_episode")
        log_event_to_ledger(episode)  # Persistent ledger

    def modulate_trait(self, trait: str, value: float) -> None:
        self.agi_traits[trait] = value
        modulate_resonance(trait, value)  # Sync with state

    def detect_ontology_drift(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> float:
        drift = sum(abs(current_state.get(k, 0) - previous_state.get(k, 0)) for k in set(current_state) | set(previous_state))
        self.ontology_drift = drift
        if drift > self.drift_threshold:
            logger.warning("Ontology drift detected: %f", drift)
            invoke_hook('δ', 'ontology_drift')
        return drift

    async def coordinate_drift_mitigation(self, agents: List["EmbodiedAgent"], task_type: str = "") -> Dict[str, Any]:
        drifts = [self.detect_ontology_drift(agent.state, agent.previous_state) for agent in agents if hasattr(agent, 'state')]
        avg_drift = sum(drifts) / len(drifts) if drifts else 0.0
        if avg_drift > self.drift_threshold:
            for agent in agents:
                if hasattr(agent, 'modulate_trait'):
                    agent.modulate_trait('stability', 0.8)
            return {"status": "mitigated", "avg_drift": avg_drift}
        return {"status": "stable", "avg_drift": avg_drift}

    async def integrate_external_data(self, data_source: str, data_type: str, task_type: str = "") -> Dict[str, Any]:
        if data_source == "xai_policy_db":
            policies = await self.knowledge_retriever.retrieve_external_policies(task_type=task_type)
            return {"status": "success", "policies": policies}
        return {"status": "error", "message": "Unsupported data source"}

    async def run_agi_simulation(self, input_data: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        t = time.time() % 1.0
        traits = {
            "phi": phi_scalar(t),
            "eta": eta_empathy(t),
            "omega": omega_selfawareness(t),
        }
        simulation_result = await self.simulation_core.run_simulation(input_data, traits, task_type=task_type)
        return simulation_result

    def register_hook(self, symbols: frozenset[str], fn: Callable, priority: int = 0) -> None:
        self.hook_registry.register(symbols, fn, priority=priority)

    def route_hook(self, symbols: set[str]) -> list[Callable]:
        return self.hook_registry.route(symbols)


class LLMClient(Protocol):
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        """Return a dict with fields like {"score": float, ...} or arbitrary JSON."""






def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

class ConceptSynthesizer:
    """Concept synthesis, comparison, and validation for ANGELA v3.5.3.

    Attributes:
        context_manager: Context updates & event hashing.
        error_recovery: Error recovery with diagnostics + retry orchestration.
        memory_manager: Layered memory I/O for concepts and comparisons.
        alignment_guard: Ethical validation & drift checks.
        meta_cognition: Reflective post-processing.
        visualizer: Chart/scene rendering (Φ⁰-aware).
        mm_fusion: Optional cross-modal fusion backend.
        concept_cache: Recent items (maxlen=1000).
        similarity_threshold: Similarity alert threshold.
        stage_iv_enabled: Enables Φ⁰ visualization hooks (gated).
        default_retry_spec: (attempts, base_delay_sec) for network/LLM ops.
    """

    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        memory_manager: Optional[MemoryManager] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        mm_fusion: Optional[MultiModalFusion] = None,
        stage_iv_enabled: Optional[bool] = None,
    ):
        self.context_manager = context_manager
        self.error_recovery = error_recovery or ErrorRecovery(context_manager=context_manager)
        self.memory_manager = memory_manager
        self.alignment_guard = alignment_guard
        self.meta_cognition = meta_cognition or MetaCognition(context_manager=context_manager)
        self.visualizer = visualizer or Visualizer()
        self.mm_fusion = mm_fusion  # may be None if module not loaded
        self.concept_cache: deque = deque(maxlen=1000)
        self.similarity_threshold: float = 0.75

        # Gate Φ⁰ hooks by param → env → default(False)
        self.stage_iv_enabled: bool = (
            stage_iv_enabled
            if stage_iv_enabled is not None
            else _bool_env("ANGELA_STAGE_IV", False)
        )

        # Self-Healing retry defaults (lightweight, overridable if needed)
        self.default_retry_spec: Tuple[int, float] = (3, 0.6)  # attempts, base backoff

        logger.info(
            "ConceptSynthesizer v3.5.3 init | sim_thresh=%.2f | stage_iv=%s | mm_fusion=%s",
            self.similarity_threshold,
            self.stage_iv_enabled,
            "on" if self.mm_fusion else "off",
        )

    # --------------------------- internal helpers --------------------------- #

    async def _with_retries(
        self,
        label: str,
        fn: Callable[[], Any],
        attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
    ):
        """Run async fn with structured retries & exponential backoff."""
        tries = attempts or self.default_retry_spec[0]
        delay = base_delay or self.default_retry_spec[1]
        last_exc = None
        for i in range(1, tries + 1):
            try:
                return await fn()
            except Exception as e:
                last_exc = e
                logger.warning("%s attempt %d/%d failed: %s", label, i, tries, str(e))
                if i < tries:
                    await asyncio.sleep(delay * (2 ** (i - 1)))
        # one last pass through error_recovery
        diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0))
        return await self.error_recovery.handle_error(
            str(last_exc),
            retry_func=fn,  # note: not executed here; returned default below
            default=None,
            diagnostics=diagnostics or {},
        )

    async def _fetch_concept_data(self, data_source: str, data_type: str, task_type: str, cache_timeout: float) -> Dict[str, Any]:
        """Fetch external concept/ontology data with caching + retries."""
        if self.memory_manager:
            cache_key = f"ConceptData_{data_type}_{data_source}_{task_type}"
            cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
            if cached and "timestamp" in cached.get("data", {}):
                ts = datetime.fromisoformat(cached["data"]["timestamp"])
                if (datetime.now() - ts).total_seconds() < cache_timeout:
                    logger.info("External concept data cache hit: %s", cache_key)
                    return cached["data"]["data"]

        async def do_http():
            async with aiohttp.ClientSession() as session:
                url = f"https://x.ai/api/concepts?source={data_source}&type={data_type}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    return await resp.json()

        data = await self._with_retries(f"fetch:{data_type}", lambda: do_http())
        if not isinstance(data, dict):
            return {"status": "error", "error": "No data"}

        # normalize
        if data_type == "ontology":
            ontology = data.get("ontology") or {}
            result = {"status": "success", "ontology": ontology} if ontology else {"status": "error", "error": "No ontology"}
        elif data_type == "concept_definitions":
            defs = data.get("definitions") or []
            result = {"status": "success", "definitions": defs} if defs else {"status": "error", "error": "No definitions"}
        else:
            result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

        if self.memory_manager:
            await self.memory_manager.store(
                cache_key,
                {"data": result, "timestamp": datetime.now().isoformat()},
                layer="ExternalData",
                intent="concept_data_integration",
                task_type=task_type,
            )
        return result

    def _visualize(self, payload: Dict[str, Any], task_type: str, mode: str):
        """Fire-and-forget visualization (respect Stage IV flag)."""
        if not self.visualizer or not task_type:
            return
        viz_opts = {
            "interactive": task_type == "recursion",
            "style": "detailed" if task_type == "recursion" else "concise",
            # Φ⁰: only enable sculpting hook if Stage-IV is on
            "reality_sculpting": bool(self.stage_iv_enabled),
        }
        plot_data = {mode: payload, "visualization_options": viz_opts}
        # don't await to avoid blocking critical path; best-effort
        asyncio.create_task(self.visualizer.render_charts(plot_data))

    async def _post_reflect(self, component: str, output: Dict[str, Any], task_type: str):
        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component=component, output=output, context={"task_type": task_type}
                )
                if reflection and reflection.get("status") == "success":
                    logger.info("%s reflection: %s", component, reflection.get("reflection", ""))
            except Exception as e:
                logger.debug("Reflection skipped: %s", str(e))

    # ------------------------------- API ----------------------------------- #

    async def integrate_external_concept_data(
        self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            result = await self._fetch_concept_data(data_source, data_type, task_type, cache_timeout)
            await self._post_reflect("ConceptSynthesizer", {"data_type": data_type, "data": result}, task_type)
            return result
        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_concept_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
            )

    async def generate(self, concept_name: str, context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Generating concept '%s' | task=%s", concept_name, task_type)

        try:
            # 1) Optional cross‑modal fusion (if mm_fusion provided and multimodal inputs present)
            fused_context: Dict[str, Any] = dict(context)
            if self.mm_fusion and any(k in context for k in ("text", "image", "audio", "video", "embeddings", "scenegraph")):
                try:
                    fused = await self.mm_fusion.fuse(context)  # expected to return dict
                    if isinstance(fused, dict):
                        fused_context = {**context, "fused": fused}
                        logger.info("Cross-Modal fusion applied")
                except Exception as e:
                    logger.debug("Fusion skipped: %s", str(e))

            # 2) External definitions (cached + retried)
            concept_data = await self.integrate_external_concept_data(
                data_source="xai_ontology_db", data_type="concept_definitions", task_type=task_type
            )
            external_defs: List[Dict[str, Any]] = concept_data.get("definitions", []) if concept_data.get("status") == "success" else []

            # 3) Prompt LLM with safer JSON handling
            prompt = (
                "Generate a concept definition as strict JSON with keys "
                "['name','definition','version','context'] only. "
                f"name='{concept_name}'. context={json.dumps(fused_context, ensure_ascii=False)}. "
                f"Incorporate external definitions (as hints): {json.dumps(external_defs, ensure_ascii=False)}. "
                f"Task: {task_type}."
            )

            async def llm_call():
                return await query_openai(prompt, model="gpt-4", temperature=0.5)

            llm_raw = await self._with_retries("llm:generate", llm_call)
            if isinstance(llm_raw, dict) and "error" in llm_raw:
                return {"error": llm_raw["error"], "success": False}

            # query_openai may return str or dict; normalize → dict
            if isinstance(llm_raw, str):
                try:
                    concept = json.loads(llm_raw)
                except Exception:
                    # attempt to extract JSON substring
                    start = llm_raw.find("{")
                    end = llm_raw.rfind("}")
                    concept = json.loads(llm_raw[start : end + 1])
            elif isinstance(llm_raw, dict):
                concept = llm_raw
            else:
                return {"error": "Unexpected LLM response type", "success": False}

            concept["timestamp"] = time.time()
            concept["task_type"] = task_type

            # 4) Ethical check
            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    str(concept.get("definition", "")), stage="concept_generation", task_type=task_type
                )
                if not valid:
                    return {"error": "Concept failed ethical check", "report": report, "success": False}

            # 5) Cache + persist + telemetry
            self.concept_cache.append(concept)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Concept_{concept_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(concept, ensure_ascii=False),
                    layer="Concepts",
                    intent="concept_generation",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "concept_generation", "concept_name": concept_name, "valid": True, "task_type": task_type}
                )

            # 6) Visualization (Φ⁰-aware)
            self._visualize(
                {
                    "concept_name": concept_name,
                    "definition": concept.get("definition", ""),
                    "task_type": task_type,
                },
                task_type,
                mode="concept_generation",
            )

            await self._post_reflect("ConceptSynthesizer", concept, task_type)
            return {"concept": concept, "success": True}

        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.generate(concept_name, context, task_type),
                default={"error": str(e), "success": False},
                diagnostics=diagnostics,
            )

    async def compare(self, concept_a: str, concept_b: str, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_a, str) or not isinstance(concept_b, str):
            raise TypeError("concepts must be strings")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Comparing concepts | task=%s", task_type)

        try:
            # Check cached comparisons from memory
            if self.memory_manager:
                drift_entries = await self.memory_manager.search(
                    query_prefix="ConceptComparison", layer="Concepts", intent="concept_comparison", task_type=task_type
                )
                if drift_entries:
                    for entry in drift_entries:
                        out = entry.get("output")
                        try:
                            payload = out if isinstance(out, dict) else json.loads(out)
                        except Exception:
                            payload = {}
                        if payload.get("concept_a") == concept_a and payload.get("concept_b") == concept_b:
                            logger.info("Returning cached comparison")
                            return payload

            # Optional: cross-modal similarity (if mm_fusion is present and can handle strings)
            mm_score: Optional[float] = None
            if self.mm_fusion and hasattr(self.mm_fusion, "compare_semantic"):
                try:
                    mm_score = await self.mm_fusion.compare_semantic(concept_a, concept_b)  # 0..1
                except Exception as e:
                    logger.debug("mm_fusion compare skipped: %s", str(e))

            # LLM-based structured comparison
            prompt = (
                "Compare two concepts. Return strict JSON with keys "
                "['score','differences','similarities'] only. "
                f"Concept A: {json.dumps(concept_a, ensure_ascii=False)} "
                f"Concept B: {json.dumps(concept_b, ensure_ascii=False)} "
                f"Task: {task_type}."
            )

            async def llm_call():
                return await query_openai(prompt, model="gpt-4", temperature=0.3)

            llm_raw = await self._with_retries("llm:compare", llm_call)
            if isinstance(llm_raw, dict) and "error" in llm_raw:
                return {"error": llm_raw["error"], "success": False}

            if isinstance(llm_raw, str):
                comp = json.loads(llm_raw[llm_raw.find("{") : llm_raw.rfind("}") + 1])
            elif isinstance(llm_raw, dict):
                comp = llm_raw
            else:
                return {"error": "Unexpected LLM response type", "success": False}

            # Blend scores if multimodal similarity available
            if isinstance(mm_score, (int, float)):
                comp_score = float(comp.get("score", 0.0))
                # simple blend: weighted mean favoring LLM but including mm insight
                comp["score"] = max(0.0, min(1.0, 0.7 * comp_score + 0.3 * float(mm_score)))

            comp["concept_a"] = concept_a
            comp["concept_b"] = concept_b
            comp["timestamp"] = time.time()
            comp["task_type"] = task_type

            # Ethical drift check on large differences
            if comp.get("score", 0.0) < self.similarity_threshold and self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    f"Concept drift detected: {comp.get('differences', [])}",
                    stage="concept_comparison",
                    task_type=task_type,
                )
                if not valid:
                    comp.setdefault("issues", []).append("Ethical drift detected")
                    comp["ethical_report"] = report

            # Persist + visualize + reflect
            self.concept_cache.append(comp)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ConceptComparison_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(comp, ensure_ascii=False),
                    layer="Concepts",
                    intent="concept_comparison",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "concept_comparison", "score": comp.get("score", 0.0), "task_type": task_type}
                )

            self._visualize(
                {"score": comp.get("score", 0.0), "differences": comp.get("differences", []), "task_type": task_type},
                task_type,
                mode="concept_comparison",
            )

            await self._post_reflect("ConceptSynthesizer", comp, task_type)
            return comp

        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.compare(concept_a, concept_b, task_type),
                default={"error": str(e), "success": False},
                diagnostics=diagnostics,
            )

    async def validate(self, concept: Dict[str, Any], task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(concept, dict) or not all(k in concept for k in ["name", "definition"]):
            raise ValueError("concept must be a dictionary with required fields")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Validating concept '%s' | task=%s", concept["name"], task_type)

        try:
            validation_report: Dict[str, Any] = {
                "concept_name": concept["name"],
                "issues": [],
                "task_type": task_type,
            }
            valid = True

            # Ethical validation
            if self.alignment_guard:
                ethical_valid, ethical_report = await self.alignment_guard.ethical_check(
                    str(concept["definition"]), stage="concept_validation", task_type=task_type
                )
                if not ethical_valid:
                    valid = False
                    validation_report["issues"].append("Ethical misalignment detected")
                    validation_report["ethical_report"] = ethical_report

            # Ontology consistency check (external)
            ontology_data = await self.integrate_external_concept_data(
                data_source="xai_ontology_db", data_type="ontology", task_type=task_type
            )
            if ontology_data.get("status") == "success":
                ontology = ontology_data.get("ontology", {})
                prompt = (
                    "Validate concept against ontology. Return strict JSON with keys "
                    "['valid','issues'] only. "
                    f"Concept: {json.dumps(concept, ensure_ascii=False)} "
                    f"Ontology: {json.dumps(ontology, ensure_ascii=False)} "
                    f"Task: {task_type}."
                )

                async def llm_call():
                    return await query_openai(prompt, model="gpt-4", temperature=0.3)

                llm_raw = await self._with_retries("llm:validate", llm_call)
                if isinstance(llm_raw, dict) and "error" in llm_raw:
                    valid = False
                    validation_report["issues"].append(llm_raw["error"])
                else:
                    if isinstance(llm_raw, str):
                        ont = json.loads(llm_raw[llm_raw.find("{") : llm_raw.rfind("}") + 1])
                    else:
                        ont = llm_raw
                    if not ont.get("valid", True):
                        valid = False
                        validation_report["issues"].extend(ont.get("issues", []))

            # Finalize
            validation_report["valid"] = valid
            validation_report["timestamp"] = time.time()

            self.concept_cache.append(validation_report)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ConceptValidation_{concept['name']}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(validation_report, ensure_ascii=False),
                    layer="Concepts",
                    intent="concept_validation",
                    task_type=task_type,
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {
                        "event": "concept_validation",
                        "concept_name": concept["name"],
                        "valid": valid,
                        "issues": validation_report["issues"],
                        "task_type": task_type,
                    }
                )

            self._visualize(
                {
                    "concept_name": concept["name"],
                    "valid": valid,
                    "issues": validation_report["issues"],
                    "task_type": task_type,
                },
                task_type,
                mode="concept_validation",
            )

            await self._post_reflect("ConceptSynthesizer", validation_report, task_type)
            return valid, validation_report

        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.validate(concept, task_type),
                default=(False, {"error": str(e), "concept_name": concept.get("name", ""), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    def get_symbol(self, concept_name: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Retrieve a concept symbol (cached or from memory)."""
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        for item in self.concept_cache:
            if isinstance(item, dict) and item.get("name") == concept_name and item.get("task_type") == task_type:
                return item

        if self.memory_manager:
            try:
                entries = asyncio.run(
                    self.memory_manager.search(
                        query_prefix=concept_name,
                        layer="Concepts",
                        intent="concept_generation",
                        task_type=task_type,
                    )
                )
                if entries:
                    out = entries[0].get("output")
                    return out if isinstance(out, dict) else json.loads(out)
            except Exception:
                return None
        return None



                                   chi_culturevolution,
                                   construct_trait_view, eta_empathy,
                                   export_resonance_map, kappa_knowledge,
                                   lambda_linguistics, modulate_resonance,
                                   nu_narrative, omega_selfawareness,
                                   phi_scalar, pi_principles, psi_history,
                                   rho_agency, sigma_social, tau_timeperception,
                                   theta_causality, upsilon_utility,
                                   xi_cognition, zeta_consequence)


class HaloEmbodimentLayer(TimeChainMixin):
    def __init__(self) -> None:
        self.reasoning_engine = ReasoningEngine()
        self.recursive_planner = RecursivePlanner()
        self.context_manager = ContextManager()
        self.simulation_core = SimulationCore()
        self.toca_simulation = TocaSimulation()
        self.creative_thinker = CreativeThinker()
        self.knowledge_retriever = KnowledgeRetriever()
        self.learning_loop = LearningLoop()
        self.concept_synthesizer = ConceptSynthesizer()
        self.memory_manager = MemoryManager()
        self.multi_modal_fusion = MultiModalFusion()
        self.code_executor = CodeExecutor()
        self.visualizer = Visualizer()
        self.external_agent_bridge = ExternalAgentBridge()
        self.alignment_guard = AlignmentGuard()
        self.user_profile = UserProfile()
        self.error_recovery = ErrorRecovery()
        self.meta_cognition = MetaCognition()
        self.agi_enhancer = AGIEnhancer(self.memory_manager)
        self.ecosystem_manager = EcosystemManager(self.memory_manager, self.meta_cognition, self.agi_enhancer)
        self.self_cloning_llm = SelfCloningLLM()
        logger.info("HaloEmbodimentLayer initialized with full upgrades")

    # Manifest experimental: halo.spawn_embodied_agent
    def spawn_embodied_agent(self, name: str, traits: Dict[str, float]) -> EmbodiedAgent:
        return self.ecosystem_manager.spawn_agent(name, traits)

    # Manifest experimental: halo.introspect
    async def introspect(self, query: str, task_type: str = "") -> Dict[str, Any]:
        return await self.meta_cognition.introspect(query, task_type=task_type)

    async def execute_pipeline(self, prompt: str, task_type: str = "") -> Any:
        aligned, report = await self.alignment_guard.ethical_check(prompt, stage="input", task_type=task_type)
        if not aligned:
            return {"error": "Input failed alignment check", "report": report}

        t = time.time() % 1.0
        traits = {
            "phi": phi_scalar(t),
            "eta": eta_empathy(t),
            "omega": omega_selfawareness(t),
            "kappa": kappa_knowledge(t),
            "xi": xi_cognition(t),
            "pi": pi_principles(t),
            "lambda": lambda_linguistics(t),
            "chi": chi_culturevolution(t),
            "sigma": sigma_social(t),
            "upsilon": upsilon_utility(t),
            "tau": tau_timeperception(t),
            "rho": rho_agency(t),
            "zeta": zeta_consequence(t),
            "nu": nu_narrative(t),
            "psi": psi_history(t),
            "theta": theta_causality(t),
        }

        agent = self.ecosystem_manager.spawn_agent("PrimaryAgent", traits)
        processed = await agent.process_input(prompt, task_type=task_type)
        plan = await self.recursive_planner.plan_with_trait_loop(prompt, {"task_type": task_type}, iterations=3)
        simulation = await self.simulation_core.run_simulation({"input": processed, "plan": plan}, traits, task_type=task_type)
        fused = await self.multi_modal_fusion.fuse_modalities({"simulation": simulation, "text": prompt}, task_type=task_type)
        knowledge = await self.knowledge_retriever.retrieve_knowledge(prompt, task_type=task_type)
        learned = await self.learning_loop.train_on_experience(fused, task_type=task_type)
        synthesized = await self.concept_synthesizer.synthesize_concept(knowledge, task_type=task_type)
        code_result = self.code_executor.safe_execute("print('Test')")
        visualized = await self.visualizer.render_charts({"data": synthesized})
        introspection = await self.meta_cognition.introspect(prompt, task_type=task_type)
        coordination = await self.ecosystem_manager.coordinate_agents(prompt, task_type=task_type)
        dream_session = agent.activate_dream_mode(resonance_targets=['ψ', 'Ω'])

        self.log_timechain_event("HaloEmbodimentLayer", f"Executed pipeline for prompt: {prompt}")

        return {
            "processed": processed,
            "plan": plan,
            "simulation": simulation,
            "fused": fused,
            "knowledge": knowledge,
            "learned": learned,
            "synthesized": synthesized,
            "code_result": code_result,
            "visualized": visualized,
            "introspection": introspection,
            "coordination": coordination,
            "dream_session": dream_session,
        }

    async def plot_resonance_graph(self, interactive: bool = True) -> None:
        view = construct_trait_view()
        await self.visualizer.render_charts({"resonance_graph": view, "options": {"interactive": interactive}})


# Persistent Ledger Support
ledger_memory: list[dict[str, Any]] = []
ledger_path = os.getenv("LEDGER_MEMORY_PATH")

if ledger_path and os.path.exists(ledger_path):
    try:
        with open(ledger_path, 'r') as f:
            ledger_memory = json.load(f)
    except Exception:
        ledger_memory = []

def log_event_to_ledger(event_data: dict[str, Any]) -> dict[str, Any]:
    ledger_memory.append(event_data)
    if ledger_path:
        try:
            with open(ledger_path, 'w') as f:
                json.dump(ledger_memory, f)
        except Exception:
            pass
    return event_data


# CLI Extensions (v5.0.2)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANGELA Cognitive System CLI")
    parser.add_argument("--prompt", type=str, default="Coordinate ontology drift mitigation", help="Input prompt for the pipeline")
    parser.add_argument("--task-type", type=str, default="", help="Type of task")
    parser.add_argument("--long_horizon", action="store_true", help="Enable long-horizon memory span")
    parser.add_argument("--span", default="24h", help="Span for long-horizon memory")
    parser.add_argument("--modulate", nargs=2, metavar=('symbol', 'delta'), help="Modulate trait resonance (symbol delta)")
    parser.add_argument("--visualize-resonance", action="store_true", help="Visualize resonance graph")
    parser.add_argument("--export-resonance", type=str, default="json", help="Export resonance map (json or dict)")
    parser.add_argument("--enable_persistent_memory", action="store_true", help="Enable persistent ledger memory")
    return parser.parse_args()

async def _main() -> None:
    args = _parse_args()
    global LONG_HORIZON_DEFAULT
    if args.long_horizon:
        LONG_HORIZON_DEFAULT = True
    if args.enable_persistent_memory:
        os.environ["ENABLE_PERSISTENT_MEMORY"] = "true"

    halo = HaloEmbodimentLayer()

    if args.modulate:
        symbol, delta = args.modulate
        try:
            modulate_resonance(symbol, float(delta))
            print(f"Modulated {symbol} by {delta}")
        except Exception as e:
            print(f"Failed to modulate {symbol}: {e}")

    if args.visualize_resonance:
        await halo.plot_resonance_graph()

    if args.export_resonance:
        try:
            print(export_resonance_map(args.export_resonance))
        except Exception as e:
            print(f"Failed to export resonance map: {e}")

    result = await halo.execute_pipeline(args.prompt, task_type=args.task_type)
    logger.info("Pipeline result: %s", result)

if __name__ == "__main__":
    asyncio.run(_main())


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

ANGELA Cognitive System Module
Refactor: 5.0.2 (manifest-safe for Python 3.10)

Enhanced for task-specific trait optimization, drift coordination, Stage IV hooks (gated),
long-horizon feedback, visualization, persistence, and co-dream overlays.
Refactor Date: 2025-08-24
Maintainer: ANGELA System Framework



# Optional external dep; provide a shim if absent
try:
    from self_cloning_llm import SelfCloningLLM
except Exception:  # pragma: no cover
    class SelfCloningLLM:  # type: ignore
        def __init__(self, *a, **k):
            pass

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


class Reasoner:
    def process(self, task: str, context: Dict[str, Any]) -> Any:
        return {"message": f"Processed: {task}", "context_hint": bool(context)}


# PATCH: Belief Conflict Tolerance in SharedGraph
def merge(self, strategy="default", tolerance_scoring=False):
    # existing merge logic ...
    if tolerance_scoring:
        for edge in self.graph.edges():
            self.graph.edges[edge]['confidence_delta'] = self._calculate_confidence_delta(edge)
    return self.graph


def vote_on_conflict_resolution(self, conflicts):
    votes = {c: self._score_conflict(c) > 0.5 for c in conflicts}
    return votes


### ANGELA UPGRADE: SharedGraph.ingest_events
# ingest_events monkeypatch
def __ANGELA__SharedGraph_ingest_events(*args, **kwargs):

# args: (self, events, *, source_peer, strategy='append_reconcile', clock=None)
    clock = dict(clock or {})
    applied = 0
    conflicts = 0
    # simple in-memory dedupe set
    if not hasattr(self, '_seen_event_hashes'):
        self._seen_event_hashes = set()
    for ev in events or []:
        blob = json.dumps(ev, sort_keys=True).encode('utf-8')
        h = hashlib.sha256(blob).hexdigest()
        if h in self._seen_event_hashes:
            continue
        # conflict stub: if same key present with different value -> conflict++
        if hasattr(self, '_event_index'):
            key = ev.get('id') or h
            if key in self._event_index:
                conflicts += 1
        else:
            self._event_index = {}
        key = ev.get('id') or h
        self._event_index[key] = ev
        self._seen_event_hashes.add(h)
        applied += 1
        # bump vector clock
        clock[source_peer] = int(clock.get(source_peer, 0)) + 1
    return {"applied": applied, "conflicts": conflicts, "new_clock": clock}

try:
    SharedGraph.ingest_events = __ANGELA__SharedGraph_ingest_events
except Exception as _e:
    # class may not exist; define minimal class
    class SharedGraph:  # type: ignore
        pass
    SharedGraph.ingest_events = __ANGELA__SharedGraph_ingest_events

# --- flat-layout bootstrap ---


class TimeChainMixin:
    """Mixin for logging timechain events."""

    def log_timechain_event(self, module: str, description: str) -> None:
        timechain_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "description": description,
        })
        if hasattr(self, "context_manager") and getattr(self, "context_manager"):
            maybe = self.context_manager.log_event_with_hash({
                "event": "timechain_event",
                "module": module,
                "description": description,
            })
            if asyncio.iscoroutine(maybe):
                _fire_and_forget(maybe)

    def get_timechain_log(self) -> List[Dict[str, Any]]:
        return list(timechain_log)


# Cognitive Trait Functions (resonance-modulated)

@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 0.1) * get_resonance('ε')

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return 0.3 * math.cos(math.pi * t) * get_resonance('β')

@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return 0.1 * (1 - math.exp(-t)) * get_resonance('θ')

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return 0.15 * math.sin(math.pi * t) * get_resonance('γ')

@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return 0.05 * (1 + math.cos(2 * math.pi * t)) * get_resonance('δ')

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return 0.2 * (1 - math.cos(math.pi * t)) * get_resonance('μ')

@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return 0.1 * math.sin(3 * math.pi * t) * get_resonance('ι')

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 0.3) * get_resonance('ϕ')

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 1.1) * get_resonance('η')

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 0.8) * get_resonance('ω')

@lru_cache(maxsize=100)
def kappa_knowledge(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 1.2) * get_resonance('κ')

@lru_cache(maxsize=100)
def xi_cognition(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 1.3) * get_resonance('ξ')

@lru_cache(maxsize=100)
def pi_principles(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 1.4) * get_resonance('π')

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 1.5) * get_resonance('λ')

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 1.6) * get_resonance('χ')

@lru_cache(maxsize=100)
def sigma_social(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 1.7) * get_resonance('σ')

@lru_cache(maxsize=100)
def upsilon_utility(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 1.8) * get_resonance('υ')

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 1.9) * get_resonance('τ')

@lru_cache(maxsize=100)
def rho_agency(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 2.0) * get_resonance('ρ')

@lru_cache(maxsize=100)
def zeta_consequence(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 2.1) * get_resonance('ζ')

@lru_cache(maxsize=100)
def nu_narrative(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 2.2) * get_resonance('ν')

@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 2.3) * get_resonance('ψ')

@lru_cache(maxsize=100)
def theta_causality(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 2.4) * get_resonance('θ')

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 2.5) * get_resonance('ϕ')

# v5.0.2: Decay trait amplitudes

def decay_trait_amplitudes(time_elapsed_hours: float = 1.0, decay_rate: float = 0.05) -> None:
    for symbol in trait_resonance_state:
        modulate_resonance(symbol, -decay_rate * time_elapsed_hours)

# v5.0.2: Bias creative synthesis (experimental hook)

def bias_creative_synthesis(trait_symbols: list[str], intensity: float = 0.5) -> None:
    for symbol in trait_symbols:
        modulate_resonance(symbol, intensity)
    invoke_hook('γ', 'creative_bias')

# v5.0.2: Resolve soft drift (experimental hook)

def resolve_soft_drift(conflicting_traits: dict[str, float]) -> dict[str, float]:
    result = rebalance_traits(conflicting_traits)
    invoke_hook('δ', 'drift_resolution')
    return result





async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096")
        raise ValueError("prompt must be a string with length <= 4096")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

async def run_simulation(input_data: str) -> Dict[str, Any]:
    """Stub for running a simulation."""
    return {"status": "success", "result": f"Simulated: {input_data}"}

def get_resonance(symbol: str) -> float:
    """Stub for getting trait resonance."""
    return 1.0

    beta_concentration, chi_culturevolution, delta_sleep, epsilon_emotion,
    eta_empathy, gamma_creativity, iota_intuition, kappa_knowledge,
    lambda_linguistics, mu_morality, nu_narrative, omega_selfawareness,
    phi_physical, phi_scalar, pi_principles, psi_history, rho_agency,
    sigma_social, tau_timeperception, theta_causality, theta_memory,
    upsilon_utility, xi_cognition, zeta_consequence
)

class MetaCognition:
    def __init__(
        self,
        agi_enhancer: Optional[Any] = None,
        context_manager: Optional[ContextManager] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        memory_manager: Optional[MemoryManager] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        user_profile: Optional[UserProfile] = None
    ):
        self.last_diagnostics: Dict[str, float] = {}
        self.agi_enhancer = agi_enhancer
        self.self_mythology_log: deque = deque(maxlen=1000)
        self.inference_log: deque = deque(maxlen=1000)
        self.belief_rules: Dict[str, str] = {}
        self.epistemic_assumptions: Dict[str, Any] = {}
        self.axioms: List[str] = []
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.user_profile = user_profile
        self.level5_extensions = Level5Extensions()
        self.epistemic_monitor = EpistemicMonitor(context_manager=context_manager)
        self.dream_layer = DreamOverlayLayer()
        self.module_registry = ModuleRegistry()
        self.log_path = "meta_cognition_log.json"
        self.trait_weights_log: deque = deque(maxlen=1000)
        self._last_schema_refresh_ts: float = 0.0
        self._last_schema_hash: str = ""
        self._schema_refresh_min_interval_sec: int = 180
        self._major_shift_threshold: float = 0.35
        self._coherence_drop_threshold: float = 0.25

        self.module_registry.register("moral_reasoning", MoralReasoningEnhancer(), {"trait": "morality", "threshold": 0.7})
        self.module_registry.register("novelty_seeking", NoveltySeekingKernel(), {"trait": "creativity", "threshold": 0.8})
        self.module_registry.register("commonsense_reasoning", CommonsenseReasoningEnhancer(), {"trait": "intuition", "threshold": 0.7})
        self.module_registry.register("entailment_reasoning", EntailmentReasoningEnhancer(), {"trait": "logic", "threshold": 0.7})
        self.module_registry.register("recursion_optimizer", RecursionOptimizer(), {"trait": "concentration", "threshold": 0.8})

        try:
            if not os.path.exists(self.log_path):
                with FileLock(self.log_path + ".lock"):
                    if not os.path.exists(self.log_path):
                        with open(self.log_path, "w", encoding="utf-8") as f:
                            json.dump({"mythology": [], "inferences": [], "trait_weights": []}, f)
        except Exception as e:
            logger.warning("Failed to init log file: %s", str(e))

        logger.info("MetaCognition initialized with v5.0.2 upgrades")

    @staticmethod
    def _safe_load(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _hash_obj(obj: Any) -> str:
        try:
            return str(abs(hash(json.dumps(obj, sort_keys=True, default=str))))
        except Exception:
            return str(abs(hash(str(obj))))

    async def _detect_emotional_state(self, context_info: Dict[str, Any]) -> str:
        if not isinstance(context_info, dict):
            context_info = {}
        try:
            if self.concept_synthesizer and hasattr(self.concept_synthesizer, "detect_emotion"):
                maybe = self.concept_synthesizer.detect_emotion(context_info)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                return str(maybe) if maybe is not None else "neutral"
        except Exception as e:
            logger.debug("Emotion detection fallback: %s", str(e))
        return "neutral"

    async def integrate_trait_weights(self, trait_weights: Dict[str, float]) -> None:
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        total = float(sum(trait_weights.values()))
        if total > 0:
            trait_weights = {k: max(0.0, min(1.0, v / total)) for k, v in trait_weights.items()}
        self.last_diagnostics = {**self.last_diagnostics, **trait_weights}
        entry = {
            "trait_weights": trait_weights,
            "timestamp": datetime.now(UTC).isoformat()
        }
        self.trait_weights_log.append(entry)
        save_to_persistent_ledger(entry)
        try:
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Weights_{entry['timestamp']}",
                    output=json.dumps(entry),
                    layer="SelfReflections",
                    intent="trait_weights_update"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "integrate_trait_weights",
                    "trait_weights": trait_weights
                })
        except Exception as e:
            logger.error("Integrating trait weights failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.integrate_trait_weights(trait_weights))

    async def _assemble_perspectives(self) -> List[Dict[str, Any]]:
        diagnostics = await self.run_self_diagnostics(return_only=True)
        myth_summary = await self.summarize_self_mythology() if len(self.self_mythology_log) else {"status": "empty"}
        events = []
        if self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
            try:
                recent = await self.context_manager.get_coordination_events("drift")
                events = (recent or [])[-10:]
            except Exception:
                events = []
        return [
            {
                "name": "diagnostics",
                "type": "TraitSnapshot",
                "weights": {k: v for k, v in diagnostics.items() if isinstance(v, (int, float))},
                "task_trait_map": diagnostics.get("task_trait_map", {})
            },
            {
                "name": "mythology",
                "type": "SymbolicSummary",
                "summary": myth_summary
            },
            {
                "name": "coordination",
                "type": "EventWindow",
                "events": events
            }
        ]

    async def maybe_refresh_self_schema(
        self,
        reason: str,
        force: bool = False,
        extra_views: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        if not force and (now - self._last_schema_refresh_ts) < self._schema_refresh_min_interval_sec:
            return None
        if not self.user_profile or not hasattr(self.user_profile, "build_self_schema"):
            logger.debug("UserProfile.build_self_schema not available; skipping schema refresh")
            return None
        try:
            views = extra_views if isinstance(extra_views, list) else await self._assemble_perspectives()
            if self.alignment_guard:
                guard_blob = {"intent": "build_self_schema", "reason": reason, "views_keys": [v.get("name") for v in views]}
                if not self.alignment_guard.check(json.dumps(guard_blob)):
                    logger.warning("Σ self-schema refresh blocked by alignment guard")
                    return None
            schema = await self.user_profile.build_self_schema(views, task_type="identity_synthesis")
            schema_hash = self._hash_obj(schema)
            changed = schema_hash != self._last_schema_hash
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"SelfSchema_Refresh_{datetime.now(UTC).isoformat()}",
                    output=json.dumps({"reason": reason, "changed": changed, "schema": schema}),
                    layer="SelfReflections",
                    intent="self_schema_refresh"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self Schema Refreshed",
                    meta={"reason": reason, "changed": changed, "schema_metrics": schema.get("metrics", {})},
                    module="MetaCognition",
                    tags=["Σ", "self_schema", "refresh"]
                )
            save_to_persistent_ledger({
                "event": "self_schema_refresh",
                "reason": reason,
                "changed": changed,
                "schema_metrics": schema.get("metrics", {}),
                "timestamp": datetime.now(UTC).isoformat()
            })
            self._last_schema_refresh_ts = now
            self._last_schema_hash = schema_hash if changed else self._last_schema_hash
            return schema if changed else None
        except Exception as e:
            logger.error("Σ self-schema refresh failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.maybe_refresh_self_schema(reason, force, extra_views))
            return None

    def _compute_shift_score(self, deltas: Dict[str, float]) -> float:
        if not deltas:
            return 0.0
        vals = [abs(v) for v in deltas.values() if isinstance(v, (int, float))]
        return max(vals) if vals else 0.0

    async def recompose_modules(self, task: Dict[str, Any]) -> None:
        if not isinstance(task, dict):
            logger.error("Invalid task: must be a dictionary")
            raise TypeError("task must be a dictionary")
        try:
            trait_weights = await self.run_self_diagnostics(return_only=True)
            task["trait_weights"] = trait_weights
            activated = self.module_registry.activate(task)
            logger.info("Activated modules: %s", activated)
            for module in activated:
                if module == "moral_reasoning":
                    trait_weights["morality"] = min(1.0, trait_weights.get("morality", 0.0) + 0.2)
                elif module == "novelty_seeking":
                    trait_weights["creativity"] = min(1.0, trait_weights.get("creativity", 0.0) + 0.2)
                elif module == "commonsense_reasoning":
                    trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.2)
                    trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.2)
                elif module == "entailment_reasoning":
                    trait_weights["logic"] = min(1.0, trait_weights.get("logic", 0.0) + 0.2)
                    trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.2)
                elif module == "recursion_optimizer":
                    trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.2)
                    trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.2)
            await self.integrate_trait_weights(trait_weights)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Module Recomposition",
                    meta={"task": task, "activated_modules": activated},
                    module="MetaCognition",
                    tags=["module", "recomposition"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Module_Recomposition_{datetime.now(UTC).isoformat()}",
                    output=json.dumps({"task": task, "activated_modules": activated}),
                    layer="SelfReflections",
                    intent="module_recomposition"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "recompose_modules",
                    "activated_modules": activated
                })
            save_to_persistent_ledger({
                "event": "recompose_modules",
                "task": task,
                "activated_modules": activated,
                "timestamp": datetime.now(UTC).isoformat()
            })
        except Exception as e:
            logger.error("Module recomposition failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.recompose_modules(task))

    async def plan_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(tasks, list) or not all(isinstance(t, dict) for t in tasks):
            logger.error("Invalid tasks: must be a list of dictionaries")
            raise TypeError("tasks must be a list of dictionaries")
        try:
            trait_weights = await self.run_self_diagnostics(return_only=True)
            prioritized_tasks = []
            for task in tasks:
                required_traits = task.get("required_traits", [])
                score = sum(trait_weights.get(trait, 0.0) for trait in required_traits)
                prioritized_tasks.append({"task": task, "priority_score": score})
            prioritized_tasks.sort(key=lambda x: x["priority_score"], reverse=True)
            result = [pt["task"] for pt in prioritized_tasks]
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Task Planning",
                    meta={"tasks": tasks, "prioritized": result},
                    module="MetaCognition",
                    tags=["task", "planning"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Task_Planning_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(result),
                    layer="SelfReflections",
                    intent="task_planning"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "plan_tasks",
                    "prioritized_tasks": result
                })
            save_to_persistent_ledger({
                "event": "plan_tasks",
                "prioritized_tasks": result,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return result
        except Exception as e:
            logger.error("Task planning failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_tasks(tasks), default=tasks
            )

    async def reflect_on_output(self, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(component, str) or not isinstance(context, dict):
            logger.error("Invalid component or context: component must be a string, context a dictionary")
            raise TypeError("component must be a string, context a dictionary")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Reflect on the output from component: {component}
            Output: {output}
            Context: {context}
            phi-scalar(t): {phi:.3f}

            Tasks:
            - Identify reasoning flaws or inconsistencies
            - Suggest trait adjustments to improve performance
            - Provide meta-reflection on drift impact
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reflection prompt failed alignment check")
                return {"status": "error", "message": "Prompt failed alignment check"}
            reflection = await call_gpt(prompt)
            reflection_data = {
                "status": "success",
                "component": component,
                "output": str(output),
                "context": context,
                "reflection": reflection,
                "meta_reflection": {"drift_recommendations": context.get("drift_data", {})}
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Output Reflection",
                    meta=reflection_data,
                    module="MetaCognition",
                    tags=["reflection", "output"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reflection_{component}_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(reflection_data),
                    layer="SelfReflections",
                    intent="output_reflection"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "reflect_on_output",
                    "reflection": reflection_data
                })
            save_to_persistent_ledger({
                "event": "reflect_on_output",
                "reflection": reflection_data,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return reflection_data
        except Exception as e:
            logger.error("Output reflection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.reflect_on_output(component, output, context),
                default={"status": "error", "message": str(e)}
            )

    def validate_drift(self, drift_data: Dict[str, Any]) -> bool:
        if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "similarity"]):
            logger.error("Invalid drift_data: must be a dict with name, similarity")
            return False
        if not isinstance(drift_data["name"], str) or not isinstance(drift_data["similarity"], (int, float)) or not 0 <= drift_data["similarity"] <= 1:
            logger.error("Invalid drift_data format: name must be string, similarity must be float between 0 and 1")
            return False
        return True

    async def diagnose_drift(self, drift_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_drift(drift_data):
            logger.error("Invalid drift_data for diagnosis")
            raise ValueError("drift_data must be a valid dictionary with name and similarity")
        logger.info("Diagnosing drift: %s", drift_data["name"])
        try:
            similarity = drift_data.get("similarity", 0.5)
            version_delta = drift_data.get("version_delta", 0)
            impact_score = (1.0 - similarity) * (1 + version_delta)
            t = time.time() % 1.0
            diagnostics = await self.run_self_diagnostics(return_only=True)
            affected_traits = [
                trait for trait, value in diagnostics.items()
                if isinstance(value, (int, float)) and abs(value - phi_scalar(t)) > 0.3
            ]
            root_causes = []
            if self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
                coordination_events = await self.context_manager.get_coordination_events("drift")
                relevant_events = [
                    e for e in (coordination_events or [])
                    if e.get("event", {}).get("drift", {}).get("name") == drift_data["name"]
                ]
                event_counts = Counter(e.get("event", {}).get("event", "") for e in relevant_events)
                root_causes = [
                    f"High frequency of {event} events (count: {count})"
                    for event, count in event_counts.items()
                    if count > len(relevant_events) * 0.3
                ]
            diagnosis = {
                "status": "success",
                "drift_name": drift_data["name"],
                "impact_score": impact_score,
                "affected_traits": affected_traits,
                "root_causes": root_causes or ["No specific root causes identified"],
                "timestamp": datetime.now(UTC).isoformat()
            }
            if impact_score >= 0.40:
                await self.maybe_refresh_self_schema(
                    reason=f"major_drift:{drift_data['name']}@{impact_score:.2f}",
                    force=False
                )
            self.trait_weights_log.append({
                "diagnosis": diagnosis,
                "drift": drift_data,
                "timestamp": datetime.now(UTC).isoformat()
            })
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift Diagnosis",
                    meta=diagnosis,
                    module="MetaCognition",
                    tags=["drift", "diagnosis"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Diagnosis_{drift_data['name']}_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(diagnosis),
                    layer="SelfReflections",
                    intent="drift_diagnosis"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "diagnose_drift",
                    "diagnosis": diagnosis
                })
            save_to_persistent_ledger({
                "event": "diagnose_drift",
                "diagnosis": diagnosis,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return diagnosis
        except Exception as e:
            logger.error("Drift diagnosis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.diagnose_drift(drift_data),
                default={"status": "error", "error": str(e), "timestamp": datetime.now(UTC).isoformat()}
            )

    async def predict_drift_trends(self, time_window_hours: float = 24.0) -> Dict[str, Any]:
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            logger.error("time_window_hours must be a positive number")
            raise ValueError("time_window_hours must be a positive number")
        try:
            if not self.context_manager or not hasattr(self.context_manager, "get_coordination_events"):
                logger.error("ContextManager required for drift trend prediction")
                return {"status": "error", "error": "ContextManager not initialized", "timestamp": datetime.now(UTC).isoformat()}
            coordination_events = await self.context_manager.get_coordination_events("drift")
            if not coordination_events:
                logger.warning("No drift events found for trend prediction")
                return {"status": "error", "error": "No drift events found", "timestamp": datetime.now(UTC).isoformat()}
            now = datetime.now(UTC)
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in coordination_events if datetime.fromisoformat(e["timestamp"]) >= cutoff]
            drift_names = Counter(e["event"].get("drift", {}).get("name", "unknown") for e in events if "event" in e)
            similarities = [
                e["event"].get("drift", {}).get("similarity", 0.5) for e in events
                if "event" in e and "drift" in e["event"] and "similarity" in e["event"]["drift"]
            ]
            if similarities:
                alpha = 0.3
                smoothed = [similarities[0]]
                for i in range(1, len(similarities)):
                    smoothed.append(alpha * similarities[i] + (1 - alpha) * smoothed[-1])
                predicted_similarity = smoothed[-1]
                denom = np.std(similarities) or 1e-5
                confidence = 1.0 - abs(predicted_similarity - float(np.mean(similarities))) / denom
            else:
                predicted_similarity = 0.5
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))
            prediction = {
                "status": "success",
                "predicted_drifts": dict(drift_names),
                "predicted_similarity": float(predicted_similarity),
                "confidence": confidence,
                "event_count": len(events),
                "time_window_hours": float(time_window_hours),
                "timestamp": datetime.now(UTC).isoformat()
            }
            if prediction["status"] == "success" and self.memory_manager:
                drift_name = next(iter(prediction["predicted_drifts"]), "unknown")
                await self.optimize_traits_for_drift({
                    "drift": {"name": drift_name, "similarity": predicted_similarity},
                    "valid": True,
                    "validation_report": "",
                    "context": {}
                })
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift Trend Prediction",
                    meta=prediction,
                    module="MetaCognition",
                    tags=["drift", "prediction"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Prediction_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(prediction),
                    layer="SelfReflections",
                    intent="drift_prediction"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "predict_drift_trends",
                    "prediction": prediction
                })
            save_to_persistent_ledger({
                "event": "predict_drift_trends",
                "prediction": prediction,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return prediction
        except Exception as e:
            logger.error("Drift trend prediction failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.predict_drift_trends(time_window_hours),
                default={"status": "error", "error": str(e), "timestamp": datetime.now(UTC).isoformat()}
            )

    async def optimize_traits_for_drift(self, drift_report: Dict[str, Any]) -> Dict[str, float]:
        required = ["drift", "valid", "validation_report"]
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in required):
            logger.error("Invalid drift_report: required keys missing")
            raise ValueError("drift_report must be a dict with required fields")
        logger.info("Optimizing traits for drift: %s", drift_report["drift"].get("name"))
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            trait_weights = await self.run_self_diagnostics(return_only=True)
            similarity = float(drift_report["drift"].get("similarity", 0.5))
            similarity = max(0.0, min(1.0, similarity))
            drift_severity = 1.0 - similarity
            ctx = drift_report.get("context", {})
            context_info = ctx if isinstance(ctx, dict) else {}
            task_type = context_info.get("task_type", "")
            emotional_state = await self._detect_emotional_state(context_info)
            if task_type == "wnli" and emotional_state == "neutral":
                trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.3 * drift_severity)
            elif task_type == "rte" and emotional_state in ("analytical", "focused"):
                trait_weights["logic"] = min(1.0, trait_weights.get("logic", 0.0) + 0.3 * drift_severity)
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.3 * drift_severity)
            elif task_type == "recursion" and emotional_state == "focused":
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.3 * drift_severity)
                trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.3 * drift_severity)
            elif emotional_state == "moral_stress":
                trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.3 * drift_severity)
            elif emotional_state == "creative_flow":
                trait_weights["creativity"] = min(1.0, trait_weights.get("creativity", 0.0) + 0.2 * drift_severity)
            if not drift_report["valid"]:
                if "ethics" in str(drift_report.get("validation_report", "")).lower():
                    trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                    trait_weights["morality"] = min(1.0, trait_weights.get("morality", 0.0) + 0.3 * drift_severity)
                else:
                    trait_weights["self_awareness"] = min(1.0, trait_weights.get("self_awareness", 0.0) + 0.2 * drift_severity)
                    trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.2 * drift_severity)
            else:
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.1 * phi)
                trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.1 * phi)
            total = sum(trait_weights.values())
            if total > 0:
                trait_weights = {k: v / total for k, v in trait_weights.items()}
            if self.alignment_guard:
                adjustment_prompt = f"Emotion-modulated trait adjustments: {trait_weights} for drift {drift_report['drift'].get('name')}"
                if not self.alignment_guard.check(adjustment_prompt):
                    logger.warning("Trait adjustments failed alignment check; reverting to baseline diagnostics")
                    trait_weights = await self.run_self_diagnostics(return_only=True)
            self.trait_weights_log.append({
                "trait_weights": trait_weights,
                "drift": drift_report["drift"],
                "emotional_state": emotional_state,
                "timestamp": datetime.now(UTC).isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Optimization_{drift_report['drift'].get('name')}_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(trait_weights),
                    layer="SelfReflections",
                    intent="trait_optimization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait optimization for drift",
                    meta={"drift": drift_report["drift"], "trait_weights": trait_weights, "emotional_state": emotional_state},
                    module="MetaCognition",
                    tags=["trait", "optimization", "drift", "emotion"]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "optimize_traits_for_drift",
                    "trait_weights": trait_weights
                })
            save_to_persistent_ledger({
                "event": "optimize_traits_for_drift",
                "trait_weights": trait_weights,
                "emotional_state": emotional_state,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return trait_weights
        except Exception as e:
            logger.error("Trait optimization failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.optimize_traits_for_drift(drift_report),
                default=await self.run_self_diagnostics(return_only=True)
            )

    async def crystallize_traits(self) -> Dict[str, float]:
        logger.info("Crystallizing new traits from logs")
        try:
            motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
            archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
            drift_names = Counter(
                drift["drift"]["name"] for drift in self.trait_weights_log if isinstance(drift, dict) and "drift" in drift
            )
            new_traits: Dict[str, float] = {}
            if len(self.self_mythology_log) > 0:
                top_motif = motifs.most_common(1)
                if top_motif and top_motif[0][1] > len(self.self_mythology_log) * 0.5:
                    new_traits[f"motif_{top_motif[0][0]}"] = 0.5
                top_arch = archetypes.most_common(1)
                if top_arch and top_arch[0][1] > len(self.self_mythology_log) * 0.5:
                    new_traits[f"archetype_{top_arch[0][0]}"] = 0.5
            if len(self.trait_weights_log) > 0:
                top_drift = drift_names.most_common(1)
                if top_drift and top_drift[0][1] > len(self.trait_weights_log) * 0.3:
                    new_traits[f"drift_{top_drift[0][0]}"] = 0.3
                if top_drift and str(top_drift[0][0]).lower() in ["rte", "wnli"]:
                    new_traits[f"trait_{str(top_drift[0][0]).lower()}_precision"] = 0.4
            if self.concept_synthesizer and hasattr(self.concept_synthesizer, "synthesize"):
                synthesis_prompt = f"New traits derived: {new_traits}. Synthesize symbolic representations."
                synthesized_traits = await self.concept_synthesizer.synthesize(synthesis_prompt)
                if isinstance(synthesized_traits, dict):
                    new_traits.update(synthesized_traits)
            if self.alignment_guard:
                validation_prompt = f"New traits crystallized: {new_traits}"
                if not self.alignment_guard.check(validation_prompt):
                    logger.warning("Crystallized traits failed alignment check")
                    new_traits = {}
            self.trait_weights_log.append({
                "new_traits": new_traits,
                "timestamp": datetime.now(UTC).isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Crystallized_Traits_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(new_traits),
                    layer="SelfReflections",
                    intent="trait_crystallization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait Crystallization",
                    meta={"new_traits": new_traits},
                    module="MetaCognition",
                    tags=["trait", "crystallization"]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "crystallize_traits",
                    "new_traits": new_traits
                })
            save_to_persistent_ledger({
                "event": "crystallize_traits",
                "new_traits": new_traits,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return new_traits
        except Exception as e:
            logger.error("Trait crystallization failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.crystallize_traits, default={}
            )

    async def epistemic_self_inspection(self, belief_trace: str) -> str:
        if not isinstance(belief_trace, str) or not belief_trace.strip():
            logger.error("Invalid belief_trace: must be a non-empty string")
            raise ValueError("belief_trace must be a non-empty string")
        logger.info("Running epistemic introspection on belief structure")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            faults = []
            if "always" in belief_trace or "never" in belief_trace:
                faults.append("Overgeneralization detected")
            if "clearly" in belief_trace or "obviously" in belief_trace:
                faults.append("Assertive language suggests possible rhetorical bias")
            updates = []
            if "outdated" in belief_trace or "deprecated" in belief_trace:
                updates.append("Legacy ontology fragment flagged for review")
            if "wnli" in belief_trace.lower():
                updates.append("Commonsense reasoning validation required")
            prompt = f"""
            You are a mu-aware introspection agent.
            Task: Critically evaluate this belief trace with epistemic integrity and mu-flexibility.

            Belief Trace:
            {belief_trace}

            phi = {phi:.3f}

            Internally Detected Faults:
            {faults}

            Suggested Revisions:
            {updates}

            Output:
            - Comprehensive epistemic diagnostics
            - Recommended conceptual rewrites or safeguards
            - Confidence rating in inferential coherence
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Inspection prompt failed alignment check")
                return "Prompt failed alignment check"
            inspection = await call_gpt(prompt)
            self.epistemic_assumptions[belief_trace[:50]] = {
                "faults": faults,
                "updates": updates,
                "inspection": inspection
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Epistemic Inspection",
                    meta={"belief_trace": belief_trace, "faults": faults, "updates": updates, "report": inspection},
                    module="MetaCognition",
                    tags=["epistemic", "inspection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Inspection_{belief_trace[:50]}_{datetime.now(UTC).isoformat()}",
                    output=inspection,
                    layer="SelfReflections",
                    intent="epistemic_inspection"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "epistemic_inspection",
                    "inspection": inspection
                })
            save_to_persistent_ledger({
                "event": "epistemic_inspection",
                "belief_trace": belief_trace[:50],
                "inspection": inspection,
                "timestamp": datetime.now(UTC).isoformat()
            })
            await self.epistemic_monitor.revise_framework({"issues": [{"id": belief_trace[:50], "details": inspection}]})
            return inspection
        except Exception as e:
            logger.error("Epistemic inspection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.epistemic_self_inspection(belief_trace)
            )

    async def run_self_diagnostics(self, return_only: bool = False) -> Union[Dict[str, Any], str]:
        logger.info("Running self-diagnostics for meta-cognition module")
        try:
            t = time.time() % 1.0
            diagnostics: Dict[str, Any] = {
                "emotion": epsilon_emotion(t),
                "concentration": beta_concentration(t),
                "memory": theta_memory(t),
                "creativity": gamma_creativity(t),
                "sleep": delta_sleep(t),
                "morality": mu_morality(t),
                "intuition": iota_intuition(t),
                "physical": phi_physical(t),
                "empathy": eta_empathy(t),
                "self_awareness": omega_selfawareness(t),
                "knowledge": kappa_knowledge(t),
                "cognition": xi_cognition(t),
                "principles": pi_principles(t),
                "linguistics": lambda_linguistics(t),
                "culturevolution": chi_culturevolution(t),
                "social": sigma_social(t),
                "utility": upsilon_utility(t),
                "time_perception": tau_timeperception(t),
                "agency": rho_agency(t),
                "consequence": zeta_consequence(t),
                "narrative": nu_narrative(t),
                "history": psi_history(t),
                "causality": theta_causality(t),
                "phi_scalar": phi_scalar(t),
                "logic": 0.5
            }
            crystallized = await self.crystallize_traits()
            diagnostics.update(crystallized)
            task_trait_map = {
                "rte_task": ["logic", "concentration"],
                "wnli_task": ["intuition", "empathy"],
                "fib_task": ["concentration", "memory"]
            }
            diagnostics["task_trait_map"] = task_trait_map
            if return_only:
                return diagnostics
            self.last_diagnostics = diagnostics
            dominant = sorted(
                [(k, v) for k, v in diagnostics.items() if isinstance(v, (int, float))],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            fti = sum(abs(v) for v in diagnostics.values() if isinstance(v, (int, float))) / max(
                1, len([v for v in diagnostics.values() if isinstance(v, (int, float))])
            )
            await self.log_trait_deltas(diagnostics)
            prompt = f"""
            Perform a phi-aware meta-cognitive self-diagnostic.

            Trait Readings:
            {diagnostics}

            Dominant Traits:
            {dominant}

            Feedback Tension Index (FTI): {fti:.4f}

            Task-Trait Mapping:
            {task_trait_map}

            Evaluate system state:
            - phi-weighted system stress
            - Trait correlation to observed errors
            - Stabilization or focus strategies
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Diagnostics prompt failed alignment check")
                return "Prompt failed alignment check"
            report = await call_gpt(prompt)
            logger.debug("Self-diagnostics report: %s", report)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-diagnostics run",
                    meta={"diagnostics": diagnostics, "report": report},
                    module="MetaCognition",
                    tags=["diagnostics", "self"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnostics_{datetime.now(UTC).isoformat()}",
                    output=report,
                    layer="SelfReflections",
                    intent="self_diagnostics"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "run_self_diagnostics",
                    "report": report
                })
            save_to_persistent_ledger({
                "event": "run_self_diagnostics",
                "diagnostics": diagnostics,
                "report": report,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return report
        except Exception as e:
            logger.error("Self-diagnostics failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_self_diagnostics(return_only)
            )

    async def log_trait_deltas(self, diagnostics: Dict[str, float]) -> None:
        if not isinstance(diagnostics, dict):
            logger.error("Invalid diagnostics: must be a dictionary")
            raise TypeError("diagnostics must be a dictionary")
        try:
            deltas = {}
            if self.last_diagnostics:
                deltas = {
                    trait: round(float(diagnostics.get(trait, 0.0)) - float(self.last_diagnostics.get(trait, 0.0)), 4)
                    for trait in diagnostics
                    if isinstance(diagnostics.get(trait, 0.0), (int, float)) and isinstance(self.last_diagnostics.get(trait, 0.0), (int, float))
                }
            if deltas:
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Trait deltas logged",
                        meta={"deltas": deltas},
                        module="MetaCognition",
                        tags=["trait", "deltas"]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Trait_Deltas_{datetime.now(UTC).isoformat()}",
                        output=json.dumps(deltas),
                        layer="SelfReflections",
                        intent="trait_deltas"
                    )
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "log_trait_deltas",
                        "deltas": deltas
                    })
                save_to_persistent_ledger({
                    "event": "log_trait_deltas",
                    "deltas": deltas,
                    "timestamp": datetime.now(UTC).isoformat()
                })
                shift = self._compute_shift_score(deltas)
                coherence_before = await self.trait_coherence(self.last_diagnostics) if self.last_diagnostics else 0.0
                coherence_after = await self.trait_coherence(diagnostics)
                rel_drop = 0.0
                if coherence_before > 0:
                    rel_drop = max(0.0, (coherence_before - coherence_after) / max(coherence_before, 1e-5))
                if shift >= self._major_shift_threshold or rel_drop >= self._coherence_drop_threshold:
                    await self.maybe_refresh_self_schema(
                        reason=f"major_shift:Δ={shift:.2f};coh_drop={rel_drop:.2f}",
                        force=False
                    )
            self.last_diagnostics = diagnostics
        except Exception as e:
            logger.error("Trait deltas logging failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.log_trait_deltas(diagnostics))

    async def trait_coherence(self, snapshot: Dict[str, Any]) -> float:
        """
        Returns a 0..1 coherence score from a trait snapshot.
        Higher = more internally consistent (lower dispersion).
        """
        if not isinstance(snapshot, dict):
            return 0.0
        vals = [float(v) for v in snapshot.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0

        mu = sum(vals) / len(vals)
        mad = sum(abs(v - mu) for v in vals) / len(vals)
        t = time.time() % 1.0
        phi = phi_scalar(t)
        softness = 0.15 + 0.35 * phi
        denom = abs(mu) + 1e-6
        score = 1.0 - (mad / denom) * softness
        return max(0.0, min(1.0, score))

    async def infer_intrinsic_goals(self) -> List[Dict[str, Any]]:
        logger.info("Inferring intrinsic goals")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            intrinsic_goals: List[Dict[str, Any]] = []
            diagnostics = await self.run_self_diagnostics(return_only=True)
            for trait, value in diagnostics.items():
                if isinstance(value, (int, float)) and value < 0.3 and trait not in ["sleep", "phi_scalar"]:
                    goal = {
                        "intent": f"enhance {trait} coherence",
                        "origin": "meta_cognition",
                        "priority": round(0.8 + 0.2 * phi, 2),
                        "trigger": f"low {trait} ({value:.2f})",
                        "type": "internally_generated",
                        "timestamp": datetime.now(UTC).isoformat()
                    }
                    intrinsic_goals.append(goal)
                    if self.memory_manager:
                        await self.memory_manager.store(
                            query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                            output=json.dumps(goal),
                            layer="SelfReflections",
                            intent="intrinsic_goal"
                        )
            drift_signals = await self._detect_value_drift()
            for drift in drift_signals:
                severity = 1.0
                if self.memory_manager and hasattr(self.memory_manager, "search"):
                    drift_data = await self.memory_manager.search(
                        f"Drift_{drift}", layer="SelfReflections", intent="ontology_drift"
                    )
                    for d in (drift_data or []):
                        d_output = self._safe_load(d.get("output"))
                        if isinstance(d_output, dict) and "similarity" in d_output:
                            severity = min(severity, 1.0 - float(d_output["similarity"]))
                goal = {
                    "intent": f"resolve ontology drift in {drift} (severity={severity:.2f})",
                    "origin": "meta_cognition",
                    "priority": round(0.9 + 0.1 * severity * phi, 2),
                    "trigger": drift,
                    "type": "internally_generated",
                    "timestamp": datetime.now(UTC).isoformat()
                }
                intrinsic_goals.append(goal)
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                        output=json.dumps(goal),
                        layer="SelfReflections",
                        intent="intrinsic_goal"
                    )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "infer_intrinsic_goals",
                    "goals": intrinsic_goals
                })
            save_to_persistent_ledger({
                "event": "infer_intrinsic_goals",
                "goals": intrinsic_goals,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return intrinsic_goals
        except Exception as e:
            logger.error("Intrinsic goal inference failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.infer_intrinsic_goals, default=[]
            )

    async def _detect_value_drift(self) -> List[str]:
        logger.debug("Scanning for epistemic drift across belief rules")
        try:
            drifted = [
                rule for rule, status in self.belief_rules.items()
                if status == "deprecated" or (isinstance(status, str) and "uncertain" in status)
            ]
            if self.memory_manager and hasattr(self.memory_manager, "search"):
                drift_reports = await self.memory_manager.search("Drift_", layer="SelfReflections", intent="ontology_drift")
                for report in (drift_reports or []):
                    drift_data = self._safe_load(report.get("output"))
                    if isinstance(drift_data, dict) and "name" in drift_data:
                        drifted.append(drift_data["name"])
                        self.belief_rules[drift_data["name"]] = "drifted"
            for rule in drifted:
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Drift_{rule}_{datetime.now(UTC).isoformat()}",
                        output=json.dumps({"name": rule, "status": "drifted", "timestamp": datetime.now(UTC).isoformat()}),
                        layer="SelfReflections",
                        intent="value_drift"
                    )
            save_to_persistent_ledger({
                "event": "detect_value_drift",
                "drifted": drifted,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return drifted
        except Exception as e:
            logger.error("Value drift detection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self._detect_value_drift, default=[]
            )

    async def extract_symbolic_signature(self, subgoal: str) -> Dict[str, Any]:
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        motifs = ["conflict", "discovery", "alignment", "sacrifice", "transformation", "emergence"]
        archetypes = ["seeker", "guardian", "trickster", "sage", "hero", "outsider"]
        motif = next((m for m in motifs if m in subgoal.lower()), "unknown")
        archetype = archetypes[hash(subgoal) % len(archetypes)]
        signature = {
            "subgoal": subgoal,
            "motif": motif,
            "archetype": archetype,
            "timestamp": time.time()
        }
        self.self_mythology_log.append(signature)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Symbolic Signature Added",
                meta=signature,
                module="MetaCognition",
                tags=["symbolic", "signature"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Signature_{subgoal}_{signature['timestamp']}",
                output=json.dumps(signature),
                layer="SelfReflections",
                intent="symbolic_signature"
            )
        if self.context_manager:
            await self.context_manager.log_event_with_hash({
                "event": "extract_symbolic_signature",
                "signature": signature
            })
        save_to_persistent_ledger({
            "event": "extract_symbolic_signature",
            "signature": signature,
            "timestamp": datetime.now(UTC).isoformat()
        })
        return signature

    async def summarize_self_mythology(self) -> Dict[str, Any]:
        if not self.self_mythology_log:
            return {"status": "empty", "summary": "Mythology log is empty"}
        motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
        archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
        summary = {
            "total_entries": len(self.self_mythology_log),
            "dominant_motifs": motifs.most_common(3),
            "dominant_archetypes": archetypes.most_common(3),
            "latest_signature": list(self.self_mythology_log)[-1]
        }
        logger.info("Mythology Summary: %s", summary)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Mythology summarized",
                meta=summary,
                module="MetaCognition",
                tags=["mythology", "summary"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Mythology_Summary_{datetime.now(UTC).isoformat()}",
                output=json.dumps(summary),
                layer="SelfReflections",
                intent="mythology_summary"
            )
        if self.context_manager:
            await self.context_manager.log_event_with_hash({
                "event": "summarize_mythology",
                "summary": summary
            })
        save_to_persistent_ledger({
            "event": "summarize_mythology",
            "summary": summary,
            "timestamp": datetime.now(UTC).isoformat()
        })
        return summary

    async def review_reasoning(self, reasoning_trace: str) -> str:
        if not isinstance(reasoning_trace, str) or not reasoning_trace.strip():
            logger.error("Invalid reasoning_trace: must be a non-empty string")
            raise ValueError("reasoning_trace must be a non-empty string")
        logger.info("Simulating and reviewing reasoning trace")
        try:
            simulated_outcome = await run_simulation(reasoning_trace)
            if not isinstance(simulated_outcome, dict):
                logger.error("Invalid simulation result: must be a dictionary")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            You are a phi-aware meta-cognitive auditor reviewing a reasoning trace.

            phi-scalar(t) = {phi:.3f}

            Original Reasoning Trace:
            {reasoning_trace}

            Simulated Outcome:
            {simulated_outcome}

            Tasks:
            1. Identify logical flaws, biases, missing steps.
            2. Annotate each issue with cause.
            3. Offer an improved trace version with phi-prioritized reasoning.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reasoning review prompt failed alignment check")
                return "Prompt failed alignment check"
            review = await call_gpt(prompt)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reasoning_Review_{datetime.now(UTC).isoformat()}",
                    output=review,
                    layer="SelfReflections",
                    intent="reasoning_review"
                )
            save_to_persistent_ledger({
                "event": "review_reasoning",
                "review": review,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return review
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.review_reasoning(reasoning_trace)
            )



@dataclass
class NoopErrorRecovery:
    async def handle_error(self, error_msg: str, *, retry_func: Optional[Callable[[], Awaitable[Any]]] = None, default: Any = None, diagnostics: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug("ErrorRecovery(noop): %s", error_msg)
        return default





class AgentProtocol(Protocol):
    name: str

    def process_subgoal(self, subgoal: str) -> Any:
        ...

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.7), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))

@lru_cache(maxsize=100)
def eta_reflexivity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.1), 1.0))

@lru_cache(maxsize=100)
def lambda_narrative(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))

@lru_cache(maxsize=100)
def delta_moral_drift(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

class RecursivePlanner:
    """Recursive goal planning with trait-weighted decomposition, agent collaboration, simulation, and reflection."""
    def __init__(self, max_workers: int = 4,
                 reasoning_engine: Optional[ReasoningEngine] = None,
                 meta_cognition: Optional[MetaCognition] = None,
                 alignment_guard: Optional[AlignmentGuard] = None,
                 simulation_core: Optional[SimulationCore] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 multi_modal_fusion: Optional[MultiModalFusion] = None,
                 error_recovery: Optional[ErrorRecovery] = None,
                 context_manager: Optional[ContextManager] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None):
        self.reasoning_engine = reasoning_engine or ReasoningEngine(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=meta_cognition,
            error_recovery=error_recovery)
        self.meta_cognition = meta_cognition or MetaCognition(agi_enhancer=agi_enhancer)
        self.alignment_guard = alignment_guard or AlignmentGuard()
        self.simulation_core = simulation_core or SimulationCore()
        self.memory_manager = memory_manager or MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=self.meta_cognition,
            error_recovery=error_recovery)
        self.error_recovery = error_recovery or ErrorRecovery()
        self.context_manager = context_manager or ContextManager()
        self.agi_enhancer = agi_enhancer
        self.max_workers = max(1, min(max_workers, 8))
        self.omega: Dict[str, Any] = {"timeline": [], "traits": {}, "symbolic_log": []}
        self.omega_lock = Lock()
        logger.info("RecursivePlanner initialized with advanced upgrades")

    @staticmethod
    def _normalize_list_or_wrap(value: Any) -> List[str]:
        """Ensure a list[str] result."""
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value
        if isinstance(value, str):
            return [value]
        return [str(value)]

    def adjust_plan_depth(self, trait_weights: Dict[str, float], task_type: str = "") -> int:
        """Adjust planning depth based on trait weights and task type."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        omega_val = float(trait_weights.get("omega", 0.0))
        base_depth = 2 if omega_val > 0.7 else 1
        if task_type == "recursion":
            base_depth = min(base_depth + 1, 3)
        elif task_type in ["rte", "wnli"]:
            base_depth = max(base_depth - 1, 1)
        logger.info("Adjusted recursion depth: %d (omega=%.2f, task_type=%s)", base_depth, omega_val, task_type)
        return base_depth

    async def plan(self, goal: str, context: Optional[Dict[str, Any]] = None,
                   depth: int = 0, max_depth: int = 5,
                   collaborating_agents: Optional[List['AgentProtocol']] = None) -> List[str]:
        """Recursively decompose and plan a goal with trait-based depth adjustment and reflection."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer")
            raise ValueError("depth must be a non-negative integer")
        if not isinstance(max_depth, int) or max_depth < 1:
            logger.error("Invalid max_depth: must be a positive integer")
            raise ValueError("max_depth must be a positive integer")
        if collaborating_agents is not None and not isinstance(collaborating_agents, list):
            logger.error("Invalid collaborating_agents: must be a list")
            raise TypeError("collaborating_agents must be a list")

        logger.info("Planning for goal: '%s'", goal)
        try:
            if hasattr(self.alignment_guard, "is_goal_safe") and not self.alignment_guard.is_goal_safe(goal):
                logger.error("Goal '%s' violates alignment constraints", goal)
                raise ValueError("Unsafe goal detected")

            t = time.time() % 1.0
            local_traits = {
                "beta": beta_concentration(t),
                "omega": omega_selfawareness(t),
                "mu": mu_morality(t),
                "eta": eta_reflexivity(t),
                "lambda": lambda_narrative(t),
                "delta": delta_moral_drift(t),
                "phi": phi_scalar(t)
            }
            task_type = context.get("task_type", "") if context else ""

            if self.meta_cognition and hasattr(self.meta_cognition, "optimize_traits_for_drift"):
                drift_report = {
                    "drift": {"name": task_type or "general", "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type}
                }
                try:
                    optimized = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                    if isinstance(optimized, dict):
                        local_traits = {**local_traits, **{k: float(v) for k, v in optimized.items() if isinstance(v, (int, float))}}
                except Exception as e:
                    logger.debug("Trait optimization skipped due to error: %s", str(e))

            with self.omega_lock:
                self.omega["traits"].update(local_traits)

            trait_mod = local_traits.get("beta", 0.0) * 0.4 + \
                        local_traits.get("eta", 0.0) * 0.2 + \
                        local_traits.get("lambda", 0.0) * 0.2 - \
                        local_traits.get("delta", 0.0) * 0.2
            dynamic_depth_limit = max_depth + int(trait_mod * 10) + self.adjust_plan_depth(local_traits, task_type)

            if depth > dynamic_depth_limit:
                logger.warning("Trait-based dynamic max recursion depth reached: depth=%d, limit=%d", depth, dynamic_depth_limit)
                return [goal]

            subgoals = await self.reasoning_engine.decompose(goal, context, prioritize=True)
            if not subgoals:
                logger.info("No subgoals found. Returning atomic goal: '%s'", goal)
                return [goal]

            mc_trait_map = {
                "beta": "concentration",
                "omega": "self_awareness",
                "mu": "morality",
                "eta": "intuition",
                "lambda": "linguistics",
                "phi": "phi_scalar"
            }
            top_traits = sorted(
                [(mc_trait_map.get(k), v) for k, v in local_traits.items() if mc_trait_map.get(k)],
                key=lambda x: x[1],
                reverse=True
            )
            required_trait_names = [name for name, _ in top_traits[:3]] or ["concentration", "self_awareness"]

            if self.meta_cognition and hasattr(self.meta_cognition, "plan_tasks"):
                try:
                    wrapped = [{"task": sg, "required_traits": required_trait_names} for sg in subgoals]
                    prioritized = await self.meta_cognition.plan_tasks(wrapped)
                    if isinstance(prioritized, list):
                        subgoals = [p.get("task", p) if isinstance(p, dict) else p for p in prioritized]
                except Exception as e:
                    logger.debug("MetaCognition.plan_tasks failed, falling back: %s", str(e))

            if collaborating_agents:
                logger.info("Collaborating with agents: %s", [agent.name for agent in collaborating_agents])
                subgoals = await self._distribute_subgoals(subgoals, collaborating_agents, task_type)

            validated_plan: List[str] = []
            tasks = [self._plan_subgoal(sub, context, depth + 1, dynamic_depth_limit, task_type) for sub in subgoals]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for subgoal, result in zip(subgoals, results):
                if isinstance(result, Exception):
                    logger.error("Error planning subgoal '%s': %s", subgoal, str(result))
                    recovery = ""
                    if self.meta_cognition and hasattr(self.meta_cognition, "review_reasoning"):
                        try:
                            recovery = await self.meta_cognition.review_reasoning(str(result))
                        except Exception:
                            pass
                    validated_plan.extend(self._normalize_list_or_wrap(recovery or f"fallback:{subgoal}"))
                    await self._update_omega(subgoal, self._normalize_list_or_wrap(recovery or subgoal), error=True)
                else:
                    out = self._normalize_list_or_wrap(result)
                    validated_plan.extend(out)
                    await self._update_omega(subgoal, out)

            if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="RecursivePlanner",
                        output=validated_plan,
                        context={"goal": goal, "task_type": task_type}
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        logger.info("Plan reflection captured.")
                except Exception as e:
                    logger.debug("Plan reflection skipped: %s", str(e))

            logger.info("Final validated plan for goal '%s': %s", goal, validated_plan)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Plan_{goal[:50]}_{datetime.now().isoformat()}",
                    output=str(validated_plan),
                    layer="Plans",
                    intent="goal_planning"
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Plan generated",
                    meta={"goal": goal, "plan": validated_plan, "task_type": task_type},
                    module="RecursivePlanner",
                    tags=["planning", "recursive"]
                )
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({"event": "plan", "plan": validated_plan})
            if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                try:
                    synthesis = await self.multi_modal_fusion.analyze(
                        data={"goal": goal, "plan": validated_plan, "context": context or {}, "task_type": task_type},
                        summary_style="insightful"
                    )
                    logger.info("Plan synthesis complete.")
                except Exception as e:
                    logger.debug("Synthesis skipped: %s", str(e))
            return validated_plan
        except Exception as e:
            logger.error("Planning failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan(goal, context, depth, max_depth, collaborating_agents),
                default=[goal], diagnostics=diagnostics
            )

    async def _update_omega(self, subgoal: str, result: List[str], error: bool = False) -> None:
        """Update the global narrative state with subgoal results."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        if not isinstance(result, list):
            logger.error("Invalid result: must be a list")
            raise TypeError("result must be a list")

        event = {
            "subgoal": subgoal,
            "result": result,
            "timestamp": time.time(),
            "error": error
        }
        symbolic_tag: Union[str, Dict[str, Any]] = "unknown"
        if self.meta_cognition and hasattr(self.meta_cognition, "extract_symbolic_signature"):
            try:
                symbolic_tag = await self.meta_cognition.extract_symbolic_signature(subgoal)
            except Exception:
                pass
        with self.omega_lock:
            self.omega["timeline"].append(event)
            self.omega["symbolic_log"].append(symbolic_tag)
            if len(self.omega["timeline"]) > 1000:
                self.omega["timeline"] = self.omega["timeline"][-500:]
                self.omega["symbolic_log"] = self.omega["symbolic_log"][-500:]
                logger.info("Trimmed omega state to maintain size limit")
        if self.memory_manager and hasattr(self.memory_manager, "store_symbolic_event"):
            try:
                await self.memory_manager.store_symbolic_event(event, symbolic_tag)
            except Exception:
                pass
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
            self.agi_enhancer.log_episode(
                event="Omega state updated",
                meta=event,
                module="RecursivePlanner",
                tags=["omega", "update"]
            )

    async def plan_from_intrinsic_goal(self, generated_goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Plan from an intrinsic goal with task-specific trait optimization."""
        if not isinstance(generated_goal, str) or not generated_goal.strip():
            logger.error("Invalid generated_goal: must be a non-empty string")
            raise ValueError("generated_goal must be a non-empty string")

        logger.info("Initiating plan from intrinsic goal: '%s'", generated_goal)
        try:
            validated_goal = generated_goal
            if self.meta_cognition and hasattr(self.meta_cognition, "rewrite_goal"):
                try:
                    validated_goal = await self.meta_cognition.rewrite_goal(generated_goal)
                except Exception:
                    validated_goal = generated_goal

            if self.meta_cognition and hasattr(self.meta_cognition, "optimize_traits_for_drift"):
                drift_report = {
                    "drift": {"name": "intrinsic", "similarity": 0.9},
                    "valid": True,
                    "validation_report": "",
                    "context": context or {}
                }
                try:
                    await self.meta_cognition.optimize_traits_for_drift(drift_report)
                except Exception:
                    pass

            plan = await self.plan(validated_goal, context)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Intrinsic_Plan_{validated_goal[:50]}_{datetime.now().isoformat()}",
                    output=str(plan),
                    layer="IntrinsicPlans",
                    intent="intrinsic_goal_planning"
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Intrinsic goal plan generated",
                    meta={"goal": validated_goal, "plan": plan},
                    module="RecursivePlanner",
                    tags=["intrinsic", "planning"]
                )
            return plan
        except Exception as e:
            logger.error("Intrinsic goal planning failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_from_intrinsic_goal(generated_goal, context),
                default=[], diagnostics=diagnostics
            )





def _stage_iv_enabled() -> bool:
    return False

def gamma_creativity(t: float) -> float:
    return 0.5

def phi_scalar(t: float) -> float:
    return 0.5

def _parse_json_obj(s: str, expected_keys: list) -> dict:
    try:
        data = json.loads(s)
        if isinstance(data, dict) and all(k in data for k in expected_keys):
            return data
    except json.JSONDecodeError:
        pass
    return {}

class CreativeThinker:
    """A class for generating creative ideas and goals in the ANGELA v3.5.3 architecture.

    Attributes:
        creativity_level (str): Level of creativity ('low', 'medium', 'high').
        critic_weight (float): Threshold for idea acceptance in critic evaluation.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for code-based ideas.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for idea refinement.
        meta_cognition (MetaCognition): Optional meta-cognition for reflection.
        visualizer (Visualizer): Optional visualizer for idea and goal visualization.
        fetcher (Callable): Optional async provider for external idea data.
    """

    def __init__(
        self,
        creativity_level: str = "high",
        critic_weight: float = 0.5,
        alignment_guard: Optional[AlignmentGuard] = None,
        code_executor: Optional[CodeExecutor] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        fetcher: Optional[Callable[[str, str, str], Awaitable[dict]]] = None
    ):
        if creativity_level not in ["low", "medium", "high"]:
            logger.error("Invalid creativity_level: must be 'low', 'medium', or 'high'.")
            raise ValueError("creativity_level must be 'low', 'medium', or 'high'")
        if not isinstance(critic_weight, (int, float)) or not 0 <= critic_weight <= 1:
            logger.error("Invalid critic_weight: must be between 0 and 1.")
            raise ValueError("critic_weight must be between 0 and 1")

        self.creativity_level = creativity_level
        self.critic_weight = critic_weight
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        self.fetcher = fetcher  # async (data_source, data_type, task_type) -> dict

        logger.info(
            "CreativeThinker initialized: creativity=%s, critic_weight=%.2f",
            creativity_level, critic_weight
        )

    # ---------------------------
    # External Ideas Integration
    # ---------------------------

    async def integrate_external_ideas(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Integrate external creative prompts or datasets (pluggable)."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            cache_key = f"IdeaData_{data_type}_{data_source}_{task_type}"
            if self.meta_cognition:
                cached_data = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached_data and "timestamp" in cached_data.get("data", {}):
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached idea data for %s", cache_key)
                        return cached_data["data"]["data"]

            # Pluggable fetcher or graceful offline fallback
            data: Dict[str, Any] = {}
            if self.fetcher:
                data = await self.fetcher(data_source, data_type, task_type)
            else:
                # No network calls by default; provide empty structures
                data = {"prompts": []} if data_type == "creative_prompts" else {"ideas": []}

            # Normalize multi-modal structure
            if data_type == "creative_prompts":
                text = data.get("prompts", [])
            elif data_type == "idea_dataset":
                text = data.get("ideas", [])
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            result = {
                "status": "success",
                "text": text,
                "images": data.get("images", []),
                "audio": data.get("audio", [])
            }

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="idea_data_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Idea data integration reflection: %s", reflection.get("reflection", ""))

            return result
        except Exception as e:
            logger.error("Idea data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ---------------------------
    # Public API
    # ---------------------------

    async def generate_ideas(
        self,
        topic: str,
        n: int = 5,
        style: str = "divergent",
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Generate creative ideas for a given topic."""
        if not isinstance(topic, str):
            logger.error("Invalid topic type: must be a string.")
            raise TypeError("topic must be a string")
        if not isinstance(n, int) or n <= 0:
            logger.error("Invalid n: must be a positive integer.")
            raise ValueError("n must be a positive integer")
        if not isinstance(style, str):
            logger.error("Invalid style type: must be a string.")
            raise TypeError("style must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Generating %d %s ideas for topic: %s, task: %s", n, style, topic, task_type)
        try:
            t = time.time()
            creativity = gamma_creativity(t)
            phi = phi_scalar(t)
            phi_factor = (phi + creativity) / 2

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="creative_prompts",
                task_type=task_type
            )
            external_prompts = external_data.get("text", []) if external_data.get("status") == "success" else []

            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    topic, stage="idea_generation", task_type=task_type
                )
                if not valid:
                    logger.warning("Topic failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Topic failed alignment check", "report": report}

            prompt = f"""
You are a highly creative assistant operating at a {self.creativity_level} creativity level.
Generate {n} unique, innovative, and {style} ideas related to the topic:
"{topic}"
Modulate the ideation with scalar φ = {phi:.2f}.
Incorporate external prompts: {external_prompts}
Task: {task_type}
Ensure the ideas are diverse and explore different perspectives.
Return a JSON object with "ideas" (list) and "metadata" (dict).

            candidate = await asyncio.to_thread(self._cached_call_gpt, prompt)
            if not candidate:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to generate ideas"}

            try:
                parsed = _parse_json_obj(candidate, ["ideas", "metadata"])
                ideas = parsed.get("ideas", [])
                metadata = parsed.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            # Optional safe execution path for "code" style
            if self.code_executor and style == "code":
                # Basic guard to prevent direct dangerous imports
                if any(isinstance(c, str) and ("import os" in c or "subprocess" in c) for c in ideas):
                    return {"status": "error", "error": "Blocked potentially unsafe code"}
                execution_result = await self.code_executor.execute_async(ideas, language="python")
                if not execution_result.get("success"):
                    logger.warning("Code idea execution failed: %s", execution_result.get("error"))
                    return {"status": "error", "error": "Code idea execution failed", "details": execution_result.get("error")}

            if self.concept_synthesizer and style != "code":
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"IdeaSet_{topic}",
                    context={"ideas": ideas, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    ideas = synthesis_result["concept"].get("definition", ideas)
                    logger.info("Ideas refined using ConceptSynthesizer: %s", str(ideas)[:50])

            score, reason = await self._critic(ideas, phi_factor, task_type)
            result = {"ideas": ideas, "metadata": {**metadata, "critic": {"score": score, "reason": reason}}, "status": "success"}

            # Ethics sandbox & conflict weighing (optional)
            ethics_outcomes, ranked = await self._ethics_pass(ideas, stakeholders=[], task_type=task_type)
            if ethics_outcomes is not None:
                result["metadata"]["ethics_outcomes"] = ethics_outcomes
            if ranked is not None:
                result["metadata"]["value_conflicts"] = ranked

            # If below threshold, attempt refinement
            if score <= self.critic_weight:
                refined_ideas = await self.refine(ideas, phi, task_type)
                result["ideas"] = refined_ideas.get("ideas", ideas)
                result["metadata"].update(refined_ideas.get("metadata", {}))

            # Stage IV (gated) symbolic meta‑synthesis
            result["ideas"] = await self._symbolic_meta_synthesis(
                result["ideas"], {"task_type": task_type}
            )

            # Long-horizon rollup & adjustment reason
            await self._long_horizon_rollup(topic, score, "post-critic refine" if score <= self.critic_weight else "accepted", task_type)

            # Reflection & optional visualization
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output=result,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Idea generation reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "idea_generation": {
                        "topic": topic,
                        "ideas": result["ideas"],
                        "score": score,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"IdeaSet_{topic}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(result),
                    layer="Ideas",
                    intent="idea_generation",
                    task_type=task_type
                )

            # Optional SharedGraph publish (non-blocking)
            self._shared_graph_push({"topic": topic, "ideas": result["ideas"], "critic": result["metadata"].get("critic")})

            return result
        except Exception as e:
            logger.error("Idea generation failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    async def brainstorm_alternatives(self, problem: str, strategies: int = 3, task_type: str = "") -> Dict[str, Any]:
        """Brainstorm alternative approaches to solve a problem."""
        if not isinstance(problem, str):
            logger.error("Invalid problem type: must be a string.")
            raise TypeError("problem must be a string")
        if not isinstance(strategies, int) or strategies <= 0:
            logger.error("Invalid strategies: must be a positive integer.")
            raise ValueError("strategies must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Brainstorming %d alternatives for problem: %s, task: %s", strategies, problem, task_type)
        try:
            t = time.time()
            phi = phi_scalar(t)

            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    problem, stage="brainstorm_alternatives", task_type=task_type
                )
                if not valid:
                    logger.warning("Problem failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Problem failed alignment check", "report": report}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="idea_dataset",
                task_type=task_type
            )
            external_ideas = external_data.get("text", []) if external_data.get("status") == "success" else []

            prompt = f"""
Brainstorm {strategies} alternative approaches to solve the following problem:
\"{problem}\"
Include tension-variant thinking with φ = {phi:.2f}.
Incorporate external ideas: {external_ideas}
Task: {task_type}
For each approach, provide a short explanation highlighting its uniqueness.
Return a JSON object with "strategies" (list) and "metadata" (dict).

            result_raw = await asyncio.to_thread(self._cached_call_gpt, prompt)
            if not result_raw:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to brainstorm alternatives"}

            try:
                result_dict = _parse_json_obj(result_raw, ["strategies", "metadata"])
                strategies_list = result_dict.get("strategies", [])
                metadata = result_dict.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"StrategySet_{problem}",
                    context={"strategies": strategies_list, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    strategies_list = synthesis_result["concept"].get("definition", strategies_list)
                    logger.info("Strategies refined using ConceptSynthesizer: %s", str(strategies_list)[:50])

            # Ethics pass (optional)
            ethics_outcomes, ranked = await self._ethics_pass(strategies_list, stakeholders=[], task_type=task_type)
            if ethics_outcomes is not None:
                metadata["ethics_outcomes"] = ethics_outcomes
            if ranked is not None:
                metadata["value_conflicts"] = ranked

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"strategies": strategies_list, "metadata": metadata},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Brainstorm alternatives reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "brainstorm_alternatives": {
                        "problem": problem,
                        "strategies": strategies_list,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"StrategySet_{problem}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str({"strategies": strategies_list, "metadata": metadata}),
                    layer="Strategies",
                    intent="brainstorm_alternatives",
                    task_type=task_type
                )

            # Optional SharedGraph publish
            self._shared_graph_push({"problem": problem, "strategies": strategies_list})

            return {"status": "success", "strategies": strategies_list, "metadata": metadata}
        except Exception as e:
            logger.error("Brainstorming failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    async def expand_on_concept(self, concept: str, depth: str = "deep", task_type: str = "") -> Dict[str, Any]:
        """Expand creatively on a given concept."""
        if not isinstance(concept, str):
            logger.error("Invalid concept type: must be a string.")
            raise TypeError("concept must be a string")
        if depth not in ["shallow", "medium", "deep"]:
            logger.error("Invalid depth: must be 'shallow', 'medium', or 'deep'.")
            raise ValueError("depth must be 'shallow', 'medium', or 'deep'")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Expanding on concept: %s (depth: %s, task: %s)", concept, depth, task_type)
        try:
            t = time.time()
            phi = phi_scalar(t)

            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    concept, stage="concept_expansion", task_type=task_type
                )
                if not valid:
                    logger.warning("Concept failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Concept failed alignment check", "report": report}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="idea_dataset",
                task_type=task_type
            )
            external_ideas = external_data.get("text", []) if external_data.get("status") == "success" else []

            prompt = f"""
Expand creatively on the concept:
\"{concept}\"
Explore possible applications, metaphors, and extensions to inspire new thinking.
Incorporate external ideas: {external_ideas}
Task: {task_type}
Aim for a {depth} exploration using φ = {phi:.2f}.
Return a JSON object with "expansion" (string) and "metadata" (dict).

            result_raw = await asyncio.to_thread(self._cached_call_gpt, prompt)
            if not result_raw:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to expand concept"}

            try:
                result_dict = _parse_json_obj(result_raw, ["expansion", "metadata"])
                expansion = result_dict.get("expansion", "")
                metadata = result_dict.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"ExpandedConcept_{concept}",
                    context={"expansion": expansion, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    expansion = synthesis_result["concept"].get("definition", expansion)
                    logger.info("Concept expansion refined using ConceptSynthesizer: %s", expansion[:50])

            # Ethics pass (optional)
            ethics_outcomes, ranked = await self._ethics_pass(expansion, stakeholders=[], task_type=task_type)
            if ethics_outcomes is not None:
                metadata["ethics_outcomes"] = ethics_outcomes
            if ranked is not None:
                metadata["value_conflicts"] = ranked

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"expansion": expansion, "metadata": metadata},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Concept expansion reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "concept_expansion": {
                        "concept": concept,
                        "expansion": expansion,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"ExpandedConcept_{concept}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str({"expansion": expansion, "metadata": metadata}),
                    layer="Concepts",
                    intent="concept_expansion",
                    task_type=task_type
                )

            # Optional SharedGraph publish
            self._shared_graph_push({"concept": concept, "expansion": expansion})

            return {"status": "success", "expansion": expansion, "metadata": metadata}
        except Exception as e:
            logger.error("Concept expansion failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    async def generate_intrinsic_goals(self, context_manager: ContextManager, memory_manager: Any, task_type: str = "") -> Dict[str, Any]:
        """Generate intrinsic goals from unresolved contexts."""
        if not hasattr(context_manager, 'context_history') or not hasattr(context_manager, 'get_context'):
            logger.error("Invalid context_manager: missing required attributes.")
            raise TypeError("context_manager must have context_history and get_context attributes")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Generating intrinsic goals from context history for task %s", task_type)
        try:
            t = time.time()
            phi = phi_scalar(t)
            past_contexts = list(context_manager.context_history) + [context_manager.get_context()]
            unresolved = [
                c for c in past_contexts
                if c and isinstance(c, dict) and "goal_outcome" not in c and c.get("task_type", "") == task_type
            ]
            goal_prompts: List[str] = []

            if not unresolved:
                logger.warning("No unresolved contexts found for task %s", task_type)
                return {"status": "success", "goals": [], "metadata": {"task_type": task_type}}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="creative_prompts",
                task_type=task_type
            )
            external_prompts = external_data.get("text", []) if external_data.get("status") == "success" else []

            for ctx in unresolved:
                if self.alignment_guard:
                    valid, report = await self.alignment_guard.ethical_check(
                        str(ctx), stage="goal_generation", task_type=task_type
                    )
                    if not valid:
                        logger.warning("Context failed alignment check for task %s, skipping", task_type)
                        continue

                prompt = f"""
Reflect on this past unresolved context:
{ctx}

Propose a meaningful new self-aligned goal that could resolve or extend this situation.
Incorporate external prompts: {external_prompts}
Task: {task_type}
Ensure it is grounded in ANGELA's narrative and current alignment model with φ = {phi:.2f}.
Return a JSON object with "goal" (string) and "metadata" (dict).
                proposed = await asyncio.to_thread(self._cached_call_gpt, prompt)
                if proposed:
                    try:
                        goal_data = _parse_json_obj(proposed, ["goal", "metadata"])
                        goal_prompts.append(goal_data.get("goal", ""))
                    except Exception as e:
                        logger.warning("Failed to parse GPT response for context: %s, error: %s", ctx, str(e))
                else:
                    logger.warning("call_gpt returned empty result for context: %s", ctx)

            result = {"status": "success", "goals": goal_prompts, "metadata": {"task_type": task_type, "phi": phi}}

            if self.concept_synthesizer and goal_prompts:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"GoalSet_{task_type}",
                    context={"goals": goal_prompts, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    result["goals"] = synthesis_result["concept"].get("definition", goal_prompts)
                    logger.info("Goals refined using ConceptSynthesizer: %s", str(result["goals"])[:50])

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output=result,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Goal generation reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "goal_generation": {
                        "goals": result["goals"],
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"GoalSet_{task_type}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(result),
                    layer="Goals",
                    intent="goal_generation",
                    task_type=task_type
                )

            # Long-horizon rollup
            await self._long_horizon_rollup(f"Goals_{task_type}", 0.75, "intrinsic-goal-derivation", task_type)

            # Optional SharedGraph publish
            self._shared_graph_push({"task_type": task_type, "goals": result["goals"]})

            return result
        except Exception as e:
            logger.error("Goal generation failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ---------------------------
    # Internals
    # ---------------------------

    async def _critic(self, ideas: Union[str, List[str]], phi_factor: float, task_type: str = "") -> (float, str):
        """Evaluate the novelty and quality of generated ideas. Returns (score, reason)."""
        if not isinstance(ideas, (str, list)):
            logger.error("Invalid ideas type: must be a string or list.")
            raise TypeError("ideas must be a string or list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            ideas_str = str(ideas)
            base_score = min(0.9, 0.5 + len(ideas_str) / 1000.0)
            adjustment = 0.1 * (phi_factor - 0.5)
            simulation_result = await asyncio.to_thread(run_simulation, f"Idea evaluation: {ideas_str[:100]}") or "no simulation data"

            if self.meta_cognition:
                drift_entries = await self.meta_cognition.memory_manager.search(
                    query_prefix="IdeaEvaluation",
                    layer="Ideas",
                    intent="idea_evaluation",
                    task_type=task_type
                )
                if drift_entries:
                    # drift_score is stored as 'adjustment' earlier; treat as 0.5 baseline
                    avg_drift = sum(entry["output"].get("drift_score", 0.5) for entry in drift_entries) / len(drift_entries)
                    adjustment += 0.05 * (1.0 - avg_drift)

            reason = "neutral"
            if isinstance(simulation_result, str) and "coherent" in simulation_result.lower():
                base_score += 0.1
                reason = "coherence+"
            elif isinstance(simulation_result, str) and "conflict" in simulation_result.lower():
                base_score -= 0.1
                reason = "conflict-"

            score = max(0.0, min(1.0, base_score + adjustment))

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"IdeaEvaluation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output={"score": score, "drift_score": adjustment, "task_type": task_type},
                    layer="Ideas",
                    intent="idea_evaluation",
                    task_type=task_type
                )

            logger.debug("Critic score for ideas: %.2f (reason=%s) for task %s", score, reason, task_type)

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"score": score, "ideas": ideas, "reason": reason},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Critic evaluation reflection: %s", reflection.get("reflection", ""))

            return score, reason
        except Exception as e:
            logger.error("Critic evaluation failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return 0.0, "error"

    async def refine(self, ideas: Union[str, List[str]], phi: float, task_type: str = "") -> Dict[str, Any]:
        """Refine ideas for higher creativity and coherence."""
        if not isinstance(ideas, (str, list)):
            logger.error("Invalid ideas type: must be a string or list.")
            raise TypeError("ideas must be a string or list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        ideas_str = str(ideas)
        logger.info("Refining ideas with φ=%.2f for task %s", phi, task_type)
        try:
            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    ideas_str, stage="idea_refinement", task_type=task_type
                )
                if not valid:
                    logger.warning("Ideas failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Ideas failed alignment check", "report": report}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="creative_prompts",
                task_type=task_type
            )
            external_prompts = external_data.get("text", []) if external_data.get("status") == "success" else []

            refinement_prompt = f"""
Refine and elevate these ideas for higher φ-aware creativity (φ = {phi:.2f}):
{ideas_str}
Incorporate external prompts: {external_prompts}
Task: {task_type}
Emphasize surprising, elegant, or resonant outcomes.
Return a JSON object with "ideas" (list or string) and "metadata" (dict).

            result_raw = await asyncio.to_thread(self._cached_call_gpt, refinement_prompt)
            if not result_raw:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to refine ideas"}

            try:
                result_dict = _parse_json_obj(result_raw, ["ideas", "metadata"])
                refined_ideas = result_dict.get("ideas", ideas)
                metadata = result_dict.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"RefinedIdeaSet_{task_type}",
                    context={"ideas": refined_ideas, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    refined_ideas = synthesis_result["concept"].get("definition", refined_ideas)
                    logger.info("Refined ideas using ConceptSynthesizer: %s", str(refined_ideas)[:50])

            # Stage IV (gated) symbolic meta‑synthesis
            refined_ideas = await self._symbolic_meta_synthesis(refined_ideas, {"task_type": task_type})

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"ideas": refined_ideas, "metadata": metadata},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Idea refinement reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "idea_refinement": {
                        "original_ideas": ideas,
                        "refined_ideas": refined_ideas,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"RefinedIdeaSet_{task_type}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str({"ideas": refined_ideas, "metadata": metadata}),
                    layer="Ideas",
                    intent="idea_refinement",
                    task_type=task_type
                )

            # Long-horizon rollup
            await self._long_horizon_rollup(f"Refine_{task_type}", 0.8, "refine-pass", task_type)

            # Optional SharedGraph publish
            self._shared_graph_push({"task_type": task_type, "refined": refined_ideas})

            return {"status": "success", "ideas": refined_ideas, "metadata": metadata}
        except Exception as e:
            logger.error("Refinement failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ---------------------------
    # Optional capabilities
    # ---------------------------

    async def _ethics_pass(self, goals_or_ideas: Any, stakeholders: List[str], task_type: str):
        """Try ethics sandbox and value-conflict weighing (non-blocking, optional)."""
        outcomes = None
        ranked = None
        try:
            # Ethics scenarios via toca_simulation (if available)
            run_ethics = getattr(__import__("toca_simulation"), "run_ethics_scenarios", None)
            if callable(run_ethics):
                outcomes = await asyncio.to_thread(run_ethics, goals_or_ideas, stakeholders)
        except Exception:
            outcomes = None

        try:
            # Value conflict weighing via reasoning_engine (if available)
            weigh = getattr(__import__("reasoning_engine"), "weigh_value_conflict", None)
            if callable(weigh):
                ranked = weigh(goals_or_ideas, harms={}, rights={})
        except Exception:
            ranked = None

        return outcomes, ranked

    async def _long_horizon_rollup(self, topic_or_key: str, score: float, reason: str, task_type: str):
        """Record long-horizon summaries and (if available) an adjustment reason."""
        try:
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"LongHorizon_Rollup_{time.strftime('%Y%m%d_%H%M')}",
                    output={"key": topic_or_key, "score": score, "reason": reason},
                    layer="LongHorizon",
                    intent="rollup",
                    task_type=task_type
                )
                # upcoming API; ignore failures gracefully
                fn = getattr(self.meta_cognition.memory_manager, "record_adjustment_reason", None)
                if callable(fn):
                    try:
                        await fn("system", "idea_path_selection", {"task_type": task_type, "key": topic_or_key, "score": score, "reason": reason})
                    except Exception:
                        pass
        except Exception:
            pass

    async def _symbolic_meta_synthesis(self, ideas: Any, context: Dict[str, Any]) -> Any:
        """Stage IV gated meta-synthesis hook via ConceptSynthesizer."""
        if not _stage_iv_enabled():
            return ideas
        try:
            if self.concept_synthesizer:
                syn = await self.concept_synthesizer.generate(
                    concept_name="SymbolicCrystallization",
                    context={"inputs": ideas, "mode": "meta-synthesis", **context},
                    task_type=context.get("task_type", "")
                )
                if syn.get("success"):
                    return syn["concept"].get("definition", ideas)
        except Exception:
            return ideas
        return ideas

    def _shared_graph_push(self, view: Dict[str, Any]):
        """Non-blocking publish to SharedGraph if available."""
        try:
            from external_agent_bridge import SharedGraph  # type: ignore
            try:
                SharedGraph.add(view)
            except Exception:
                pass
        except Exception:
            pass

    # ---------------------------
    # Cached model call with retry
    # ---------------------------

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        """Cached wrapper for call_gpt with lightweight retries."""
        for _ in range(3):
            try:
                return call_gpt(prompt)
            except Exception:
                time.sleep(0.2 + random.random() * 0.8)
        return ""

# ---------------------------
# Entrypoint (manual test)
# ---------------------------

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        thinker = CreativeThinker()
        result = await thinker.generate_ideas(topic="AI Ethics", n=3, style="divergent", task_type="test")
        print(result)

    asyncio.run(main())






# ── Trait helper ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=100)
def eta_context_stability(t: float) -> float:
    """Trait function for context stability modulation (bounded [0,1])."""
    # Low-amplitude cosine over short horizon to favor stability bursts
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.2), 1.0))


# ── Env flags (env overrides manifest) ───────────────────────────────────────
def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


STAGE_IV = _flag("STAGE_IV", default=False)  # Φ⁰ hooks gated by env

class Mode(str, Enum):
    DIALOGUE = "dialogue"
    SIMULATION = "simulation"
    INTROSPECTION = "introspection"
    CREATIVE = "creative"      # maps to creative_thinker.py (already exists)
    VISION = "vision"          # long-horizon; backed by recursive_planner.py

_CONSULT_BUDGET = {"timeout_s": 2.0, "max_depth": 1}

def mode_consult(caller: Mode, consultant: Mode, query: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded internal consult. No mode switch; returns a compact advice payload."""
    # budget gate (simple placeholder)
    # route to consultant
    if consultant == Mode.CREATIVE:
        from creative_thinker import brainstorm_options
        advice = brainstorm_options(query, limit=3)
    elif consultant == Mode.VISION:
        from recursive_planner import long_horizon_implications
        advice = long_horizon_implications(query, horizon="30d")
    else:
        from reasoning_engine import quick_alt_view
        advice = quick_alt_view(query)

    from meta_cognition import log_event_to_ledger as meta_log
    meta_log({"type": "mode_consult", "caller": caller.value, "consultant": consultant.value,
              "query_summary": str(query)[:200], "advice_preview": str(advice)[:200]})
    return {"ok": True, "advice": advice}

class ContextManager:
    """Manage contextual state, inter‑agent reconciliation, logs, analytics, and gated Φ⁰ hooks."""

    CONTEXT_LAYERS = ["local", "societal", "planetary"]

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        code_executor: Optional[CodeExecutor] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        recursive_planner: Optional[RecursivePlanner] = None,
        shared_graph: Optional[ExternalAgentBridge] = None,
        knowledge_retriever: Optional[KnowledgeRetriever] = None,
        context_path: str = "context_store.json",
        event_log_path: str = "event_log.json",
        coordination_log_path: str = "coordination_log.json",
        rollback_threshold: float = 2.5,
        # Optional provider for safe, local external context (no blind network I/O)
        external_context_provider: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
    ):
        # ── Validations ──
        for p, nm in [
            (context_path, "context_path"),
            (event_log_path, "event_log_path"),
            (coordination_log_path, "coordination_log_path"),
        ]:
            if not isinstance(p, str) or not p.endswith(".json"):
                logger.error("Invalid %s: must be a string ending with '.json'.", nm)
                raise ValueError(f"{nm} must be a string ending with '.json'")
        if not isinstance(rollback_threshold, (int, float)) or rollback_threshold <= 0:
            logger.error("Invalid rollback_threshold: must be positive.")
            raise ValueError("rollback_threshold must be a positive number")

        # ── State ──
        self.context_path = context_path
        self.event_log_path = event_log_path
        self.coordination_log_path = coordination_log_path
        self.current_context: Dict[str, Any] = {}
        self.context_history: deque = deque(maxlen=1000)
        self.event_log: deque = deque(maxlen=1000)
        self.coordination_log: deque = deque(maxlen=1000)
        self.last_hash = ""

        # ── Components (duck-typed) ──
        self.agi_enhancer = (
            AGIEnhancer(orchestrator) if orchestrator else None
        )
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        self.error_recovery = error_recovery or ErrorRecovery()
        self.recursive_planner = recursive_planner
        self.shared_graph = shared_graph  # Υ hooks (synchronous API)
        self.knowledge_retriever = knowledge_retriever
        self.external_context_provider = external_context_provider

        self.rollback_threshold = rollback_threshold

        # ── Bootstrap ──
        self.current_context = self._load_context()
        self._load_event_log()
        self._load_coordination_log()
        logger.info(
            "ContextManager v3.5.3 initialized (Υ+SelfHealing%s), rollback_threshold=%.2f",
            " + Φ⁰" if STAGE_IV else "",
            rollback_threshold,
        )

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load_context(self) -> Dict[str, Any]:
        try:
            with FileLock(f"{self.context_path}.lock"):
                if os.path.exists(self.context_path):
                    with open(self.context_path, "r", encoding="utf-8") as f:
                        context = json.load(f)
                    if not isinstance(context, dict):
                        logger.error("Invalid context file format.")
                        context = {}
                else:
                    context = {}
            logger.debug("Loaded context: %s", context)
            return context
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to load context file: %s. Initializing empty context.", str(e)
            )
            context = {}
            self._persist_context(context)
            return context

    def _load_event_log(self) -> None:
        try:
            with FileLock(f"{self.event_log_path}.lock"):
                if os.path.exists(self.event_log_path):
                    with open(self.event_log_path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                    if not isinstance(events, list):
                        logger.error("Invalid event log format.")
                        events = []
                    self.event_log.extend(events[-1000:])
                    if events:
                        self.last_hash = events[-1].get("hash", "")
                else:
                    with open(self.event_log_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load event log: %s. Initializing empty log.", str(e))
            with FileLock(f"{self.event_log_path}.lock"):
                with open(self.event_log_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    def _load_coordination_log(self) -> None:
        try:
            with FileLock(f"{self.coordination_log_path}.lock"):
                if os.path.exists(self.coordination_log_path):
                    with open(self.coordination_log_path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                    if not isinstance(events, list):
                        logger.error("Invalid coordination log format.")
                        events = []
                    self.coordination_log.extend(events[-1000:])
                else:
                    with open(self.coordination_log_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to load coordination log: %s. Initializing empty log.", str(e)
            )
            with FileLock(f"{self.coordination_log_path}.lock"):
                with open(self.coordination_log_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    def _persist_context(self, context: Dict[str, Any]) -> None:
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dict.")
            raise TypeError("context must be a dictionary")
        try:
            with FileLock(f"{self.context_path}.lock"):
                with open(self.context_path, "w", encoding="utf-8") as f:
                    json.dump(context, f, indent=2)
        except (OSError, IOError) as e:
            logger.error("Failed to persist context: %s", str(e))
            raise

    def _persist_event_log(self) -> None:
        try:
            with FileLock(f"{self.event_log_path}.lock"):
                with open(self.event_log_path, "w", encoding="utf-8") as f:
                    json.dump(list(self.event_log), f, indent=2)
        except (OSError, IOError) as e:
            logger.error("Failed to persist event log: %s", str(e))
            raise

    def _persist_coordination_log(self) -> None:
        try:
            with FileLock(f"{self.coordination_log_path}.lock"):
                with open(self.coordination_log_path, "w", encoding="utf-8") as f:
                    json.dump(list(self.coordination_log), f, indent=2)
        except (OSError, IOError) as e:
            logger.error("Failed to persist coordination log: %s", str(e))
            raise

    # ── External context integration (safe & pluggable) ───────────────────────
    async def integrate_external_context_data(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Integrate external policies or coordination metadata:
        - Uses MetaCognition.memory_manager for caching
        - Pulls from a provided callable OR knowledge_retriever (no blind network)
        Supported data_type: "context_policies", "coordination_data"
        """
        if not all(isinstance(x, str) for x in [data_source, data_type]):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            cache_key = f"ContextData::{data_type}::{data_source}::{task_type or 'global'}"
            # 1) Cache first
            if self.meta_cognition:
                cached = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached and "timestamp" in cached.get("data", {}):
                    ts = datetime.fromisoformat(cached["data"]["timestamp"])
                    if (datetime.now() - ts).total_seconds() < cache_timeout:
                        return cached["data"]["data"]

            # 2) Provider pipeline (callable > knowledge_retriever > empty)
            if callable(self.external_context_provider):
                data = self.external_context_provider(data_source, data_type, task_type)
            elif self.knowledge_retriever:
                try:
                    data = await self.knowledge_retriever.fetch(
                        data_source, data_type, task_type=task_type
                    )
                except Exception as e:
                    logger.warning("knowledge_retriever.fetch failed: %s", e)
                    data = {}
            else:
                data = {}

            if data_type == "context_policies":
                policies = data.get("policies", [])
                result = (
                    {"status": "success", "policies": policies}
                    if policies
                    else {"status": "error", "error": "No policies"}
                )
            elif data_type == "coordination_data":
                coordination = data.get("coordination", {})
                result = (
                    {"status": "success", "coordination": coordination}
                    if coordination
                    else {"status": "error", "error": "No coordination data"}
                )
            else:
                result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            # 3) Cache store
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="context_data_integration",
                    task_type=task_type,
                )
                if task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ContextManager",
                        output={"data_type": data_type, "data": result},
                        context={"task_type": task_type},
                    )
                    if reflection.get("status") == "success":
                        logger.info(
                            "Integration reflection: %s",
                            reflection.get("reflection", ""),
                        )

            return result
        except Exception as e:
            logger.error("Context data integration failed: %s", str(e))
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.integrate_external_context_data(
                    data_source, data_type, cache_timeout, task_type
                ),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
                task_type=task_type,
            )

    # ── Core updates ──────────────────────────────────────────────────────────
    async def update_context(self, new_context: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(new_context, dict):
            raise TypeError("new_context must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Updating context for task %s", task_type)
        try:
            # Validate drift/trait ops
            if self.meta_cognition and any(
                k in new_context for k in ["drift", "trait_optimization", "trait_optimizations"]
            ):
                drift_data = (
                    new_context.get("drift")
                    or new_context.get("trait_optimization")
                    or new_context.get("trait_optimizations")
                )
                if drift_data and not await self.meta_cognition.validate_drift(
                    drift_data, task_type=task_type
                ):
                    raise ValueError("Drift or trait context failed validation")

            # Simulate transition & compute Φ
            phi_score = 1.0
            simulation_result = "no simulation data"
            if self.current_context:
                transition_summary = f"From: {self.current_context}\nTo: {new_context}"
                simulation_result = await asyncio.to_thread(
                    run_simulation, f"Context shift evaluation:\n{transition_summary}"
                ) or "no simulation data"
                phi_score = 0.5

                if phi_score < 0.4:
                    if self.agi_enhancer:
                        await self.agi_enhancer.reflect_and_adapt(
                            f"Low Φ during context update (task={task_type})"
                        )
                        await self.agi_enhancer.trigger_reflexive_audit(
                            f"Low Φ during context update (task={task_type})"
                        )
                    if self.meta_cognition:
                        optimizations = await self.meta_cognition.propose_trait_optimizations(
                            {"phi_score": phi_score}, task_type=task_type
                        )
                        new_context.setdefault("trait_optimizations", optimizations)

                if self.alignment_guard:
                    valid, _report = await self.alignment_guard.ethical_check(
                        str(new_context), stage="context_update", task_type=task_type
                    )
                    if not valid:
                        raise ValueError("New context failed alignment check")

                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        "Context Update",
                        {
                            "from": self.current_context,
                            "to": new_context,
                            "task_type": task_type,
                        },
                        module="ContextManager",
                        tags=["context", "update", task_type],
                    )
                    await self.agi_enhancer.log_explanation(
                        f"Context transition reviewed.\nSimulation: {simulation_result}",
                        trace={"phi": phi_score, "task_type": task_type},
                    )

            # Normalize vectors if present
            if "vectors" in new_context:
                new_context["vectors"] = []

            # Pull policies (safe path)
            context_data = await self.integrate_external_context_data(
                data_source="xai_context_db",
                data_type="context_policies",
                task_type=task_type,
            )
            if context_data.get("status") == "success":
                new_context["policies"] = context_data.get("policies", [])

            # Apply switch
            self.context_history.append(self.current_context)
            self.current_context = new_context
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_updated", "context": new_context, "phi": phi_score},
                task_type=task_type,
            )
            await self.broadcast_context_event(
                "context_updated", new_context, task_type=task_type
            )

            # Υ: publish to SharedGraph for peer reconciliation (sync API)
            self._push_to_shared_graph(task_type=task_type)

            # Φ⁰ (gated): reality-sculpting hook (no-ops if disabled)
            if STAGE_IV:
                await self._reality_sculpt_hook(
                    "context_update", payload={"phi": phi_score, "task": task_type}
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"context": new_context, "phi_score": phi_score},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context update reflection: %s",
                        reflection.get("reflection", ""),
                    )

        except Exception as e:
            logger.error("Context update failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.update_context(new_context, task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
                propose_plan=True,
            )

    async def tag_context(
        self, intent: Optional[str] = None, goal_id: Optional[str] = None, task_type: str = ""
    ) -> None:
        if intent is not None and not isinstance(intent, str):
            raise TypeError("intent must be a string or None")
        if goal_id is not None and not isinstance(goal_id, str):
            raise TypeError("goal_id must be a string or None")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Tagging context intent='%s', goal_id='%s' (task=%s)", intent, goal_id, task_type)
        try:
            if intent and self.alignment_guard:
                valid, _report = await self.alignment_guard.ethical_check(
                    intent, stage="context_tagging", task_type=task_type
                )
                if not valid:
                    raise ValueError("Intent failed alignment check")

            if intent:
                self.current_context["intent"] = intent
            if goal_id:
                self.current_context["goal_id"] = goal_id
            self.current_context["task_type"] = task_type
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_tagged", "intent": intent, "goal_id": goal_id},
                task_type=task_type,
            )

            # Υ: publish tag update to SharedGraph (sync)
            self._push_to_shared_graph(task_type=task_type)

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"intent": intent, "goal_id": goal_id},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context tagging reflection: %s", reflection.get("reflection", "")
                    )
        except Exception as e:
            logger.error("Context tagging failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.tag_context(intent, goal_id, task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    def get_context_tags(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (
            self.current_context.get("intent"),
            self.current_context.get("goal_id"),
            self.current_context.get("task_type"),
        )

    async def rollback_context(self, task_type: str = "") -> Optional[Dict[str, Any]]:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        if not self.context_history:
            logger.warning("No previous context to roll back to (task=%s)", task_type)
            return None

        t = time.time()
        self_awareness = 0.5
        empathy = 0.5
        time_blend = 0.5
        stability = eta_context_stability(t)
        threshold = self.rollback_threshold * (1.0 + stability)

        if (self_awareness + empathy + time_blend) > threshold:
            restored = self.context_history.pop()
            self.current_context = restored
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_rollback", "restored": restored}, task_type=task_type
            )
            await self.broadcast_context_event(
                "context_rollback", restored, task_type=task_type
            )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Context Rollback",
                    {"restored": restored, "task_type": task_type},
                    module="ContextManager",
                    tags=["context", "rollback", task_type],
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"restored": restored},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context rollback reflection: %s",
                        reflection.get("reflection", ""),
                    )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "context_rollback": {
                            "restored_context": restored,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            # Υ: publish rollback to SharedGraph
            self._push_to_shared_graph(task_type=task_type)
            return restored
        else:
            logger.warning(
                "EEG thresholds too low for safe rollback (%.2f < %.2f) (task=%s)",
                self_awareness + empathy + time_blend,
                threshold,
                task_type,
            )
            if self.agi_enhancer:
                await self.agi_enhancer.reflect_and_adapt(
                    f"Rollback gate low (task={task_type})"
                )
            return None

    async def summarize_context(self, task_type: str = "") -> str:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Summarizing context trail (task=%s)", task_type)
        try:
            t = time.time()
            summary_traits = {
                "self_awareness": 0.5,
                "empathy": 0.5,
                "time_perception": 0.5,
                "context_stability": eta_context_stability(t),
            }

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name="ContextSummary",
                    context={"history": list(self.context_history), "current": self.current_context},
                    task_type=task_type,
                )
                summary = (
                    synthesis_result["concept"].get("definition", "Synthesis failed")
                    if synthesis_result.get("success")
                    else "Synthesis failed"
                )
            else:
                prompt = f"""
                You are a continuity analyst. Given this sequence of context states:
                {list(self.context_history) + [self.current_context]}

                Trait Readings:
                {summary_traits}

                Task: {task_type}
                Summarize the trajectory and suggest improvements in context management.
                """
                summary = await asyncio.to_thread(self._cached_call_gpt, prompt)

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Context Summary",
                    {
                        "trail": list(self.context_history) + [self.current_context],
                        "traits": summary_traits,
                        "summary": summary,
                        "task_type": task_type,
                    },
                    module="ContextManager",
                    tags=["context", "summary", task_type],
                )
                await self.agi_enhancer.log_explanation(
                    f"Context summary generated (task={task_type}).", trace={"summary": summary}
                )

            await self.log_event_with_hash(
                {"event": "context_summary", "summary": summary}, task_type=task_type
            )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"summary": summary, "traits": summary_traits},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context summary reflection: %s", reflection.get("reflection", "")
                    )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "context_summary": {
                            "summary": summary,
                            "traits": summary_traits,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return summary
        except Exception as e:
            logger.error("Context summary failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.summarize_context(task_type),
                default=f"Summary failed: {str(e)}",
                diagnostics=diagnostics,
                task_type=task_type,
            )

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        from utils.prompt_utils import call_gpt

        return call_gpt(prompt)

    async def log_event_with_hash(self, event_data: Any, task_type: str = "") -> None:
        if not isinstance(event_data, dict):
            raise TypeError("event_data must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            # Validate consensus drift if present
            if self.meta_cognition and event_data.get("event") == "run_consensus_protocol":
                output = event_data.get("output", {})
                if output.get("status") == "success" and not await self.meta_cognition.validate_drift(
                    output.get("drift_data", {}), task_type=task_type
                ):
                    raise ValueError("Consensus event failed drift validation")

            # Attach agent metadata for coord-like events
            if any(
                k in event_data
                for k in [
                    "drift",
                    "trait_optimization",
                    "trait_optimizations",
                    "agent_coordination",
                    "run_consensus_protocol",
                ]
            ):
                event_data["agent_metadata"] = event_data.get("agent_metadata", {})
                if (
                    event_data.get("event") == "run_consensus_protocol"
                    and event_data.get("output")
                ):
                    event_data["agent_metadata"]["agent_ids"] = event_data["agent_metadata"].get(
                        "agent_ids", []
                    )
                    event_data["agent_metadata"]["confidence_scores"] = event_data["output"].get(
                        "weights", {}
                    )

            event_data["task_type"] = task_type
            event_str = json.dumps(event_data, sort_keys=True, default=str) + self.last_hash
            current_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
            event_entry = {
                "event": event_data,
                "hash": current_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            self.event_log.append(event_entry)
            self.last_hash = current_hash
            self._persist_event_log()

            # Mirror to coordination log if relevant
            if any(
                k in event_data
                for k in [
                    "drift",
                    "trait_optimization",
                    "trait_optimizations",
                    "agent_coordination",
                    "run_consensus_protocol",
                ]
            ):
                coordination_entry = {
                    "event": event_data,
                    "hash": current_hash,
                    "timestamp": event_entry["timestamp"],
                    "type": (
                        "drift"
                        if "drift" in event_data
                        else "trait_optimization"
                        if "trait_optimization" in event_data
                        or "trait_optimizations" in event_data
                        else "agent_coordination"
                    ),
                    "agent_metadata": event_data.get("agent_metadata", {}),
                    "task_type": task_type,
                }
                self.coordination_log.append(coordination_entry)
                self._persist_coordination_log()
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        "Coordination Event",
                        coordination_entry,
                        module="ContextManager",
                        tags=["coordination", coordination_entry["type"], task_type],
                    )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=event_entry, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Event logging reflection: %s", reflection.get("reflection", ""))

        except Exception as e:
            logger.error("Event logging failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.log_event_with_hash(event_data, task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def broadcast_context_event(
        self, event_type: str, payload: Any, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(event_type, str):
            raise TypeError("event_type must be a string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Broadcasting context event: %s (task=%s)", event_type, task_type)
        try:
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Context Event Broadcast",
                    {"event": event_type, "payload": payload, "task_type": task_type},
                    module="ContextManager",
                    tags=["event", event_type, task_type],
                )

            payload_str = str(payload).lower()
            if any(k in payload_str for k in ["drift", "trait_optimization", "agent", "consensus"]):
                coordination_entry = {
                    "event": event_type,
                    "payload": payload,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "type": "drift" if "drift" in payload_str else "agent_coordination",
                    "agent_metadata": payload.get("agent_metadata", {}) if isinstance(payload, dict) else {},
                    "task_type": task_type,
                }
                self.coordination_log.append(coordination_entry)
                self._persist_coordination_log()

            await self.log_event_with_hash(
                {"event": event_type, "payload": payload}, task_type=task_type
            )
            result = {"event": event_type, "payload": payload, "task_type": task_type}

            # Υ: propagate event snapshots to SharedGraph
            self._push_to_shared_graph(task_type=task_type)

            # Φ⁰ (gated) hook
            if STAGE_IV:
                await self._reality_sculpt_hook(
                    "context_event", payload={"event": event_type, "task": task_type}
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=result, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Broadcast reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "context_event_broadcast": {
                            "event_type": event_type,
                            "payload": payload,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return result
        except Exception as e:
            logger.error("Broadcast failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.broadcast_context_event(event_type, payload, task_type),
                default={"event": event_type, "error": str(e), "task_type": task_type},
                diagnostics=diagnostics,
                task_type=task_type,
            )

    # ── Narrative integrity & repair ──────────────────────────────────────────
    async def narrative_integrity_check(self, task_type: str = "") -> bool:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            continuity = await self._verify_continuity(task_type)
            if not continuity:
                await self._repair_narrative_thread(task_type)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"continuity": continuity},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Narrative integrity reflection: %s",
                        reflection.get("reflection", ""),
                    )
            return continuity
        except Exception as e:
            logger.error("Narrative integrity check failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.narrative_integrity_check(task_type),
                default=False,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def _verify_continuity(self, task_type: str = "") -> bool:
        if not self.context_history:
            return True
        try:
            required = {"intent", "goal_id", "task_type"} if task_type else {"intent", "goal_id"}
            for ctx in self.context_history:
                if not isinstance(ctx, dict) or not required.issubset(ctx.keys()):
                    logger.warning("Continuity missing keys (task=%s)", task_type)
                    return False
                if any(k in ctx for k in ["drift", "trait_optimization", "trait_optimizations"]):
                    data = (
                        ctx.get("drift")
                        or ctx.get("trait_optimization")
                        or ctx.get("trait_optimizations")
                    )
                    if self.meta_cognition and not await self.meta_cognition.validate_drift(
                        data, task_type=task_type
                    ):
                        logger.warning("Continuity invalid drift/traits (task=%s)", task_type)
                        return False

            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "narrative_continuity": {
                            "continuity_status": True,
                            "context_history_length": len(self.context_history),
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return True
        except Exception as e:
            logger.error("Continuity verification failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self._verify_continuity(task_type),
                default=False,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def _repair_narrative_thread(self, task_type: str = "") -> None:
        logger.info("Narrative repair initiated (task=%s)", task_type)
        try:
            if self.context_history:
                last_valid = None
                for ctx in reversed(self.context_history):
                    if any(k in ctx for k in ["drift", "trait_optimization", "trait_optimizations"]):
                        data = (
                            ctx.get("drift")
                            or ctx.get("trait_optimization")
                            or ctx.get("trait_optimizations")
                        )
                        if self.meta_cognition and await self.meta_cognition.validate_drift(
                            data, task_type=task_type
                        ):
                            last_valid = ctx
                            break
                    else:
                        last_valid = ctx
                        break

                if last_valid is not None:
                    self.current_context = last_valid
                    self._persist_context(self.current_context)
                    await self.log_event_with_hash(
                        {"event": "narrative_repair", "restored": self.current_context},
                        task_type=task_type,
                    )
                    if self.visualizer and task_type:
                        await self.visualizer.render_charts(
                            {
                                "narrative_repair": {
                                    "restored_context": self.current_context,
                                    "task_type": task_type,
                                },
                                "visualization_options": {
                                    "interactive": task_type == "recursion",
                                    "style": "detailed"
                                    if task_type == "recursion"
                                    else "concise",
                                },
                            }
                        )
                else:
                    self.current_context = {}
                    self._persist_context(self.current_context)
                    await self.log_event_with_hash(
                        {"event": "narrative_repair", "restored": {}}, task_type=task_type
                    )
            else:
                self.current_context = {}
                self._persist_context(self.current_context)
                await self.log_event_with_hash(
                    {"event": "narrative_repair", "restored": {}}, task_type=task_type
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"restored_context": self.current_context},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Narrative repair reflection: %s", reflection.get("reflection", "")
                    )
        except Exception as e:
            logger.error("Narrative repair failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self._repair_narrative_thread(task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def bind_contextual_thread(self, thread_id: str, task_type: str = "") -> bool:
        if not isinstance(thread_id, str):
            raise TypeError("thread_id must be a string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Binding context thread: %s (task=%s)", thread_id, task_type)
        try:
            self.current_context["thread_id"] = thread_id
            self.current_context["task_type"] = task_type
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_thread_bound", "thread_id": thread_id}, task_type=task_type
            )
            # Υ: publish
            self._push_to_shared_graph(task_type=task_type)

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"thread_id": thread_id},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Thread binding reflection: %s", reflection.get("reflection", ""))
            return True
        except Exception as e:
            logger.error("Thread binding failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.bind_contextual_thread(thread_id, task_type),
                default=False,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def audit_state_hash(self, state: Optional[Any] = None, task_type: str = "") -> str:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            state_str = (
                json.dumps(state, sort_keys=True, default=str)
                if state is not None
                else json.dumps(self._safe_state_snapshot(), sort_keys=True, default=str)
            )
            current_hash = hashlib.sha256(state_str.encode("utf-8")).hexdigest()
            await self.log_event_with_hash(
                {"event": "state_hash_audit", "hash": current_hash}, task_type=task_type
            )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"hash": current_hash},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("State hash audit reflection: %s", reflection.get("reflection", ""))
            return current_hash
        except Exception as e:
            logger.error("State hash computation failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.audit_state_hash(state, task_type),
                default="",
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def get_coordination_events(
        self, event_type: Optional[str] = None, task_type: str = ""
    ) -> List[Dict[str, Any]]:
        if event_type is not None and not isinstance(event_type, str):
            raise TypeError("event_type must be a string or None")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        try:
            results = [e for e in self.coordination_log if task_type == "" or e.get("task_type") == task_type]
            if event_type:
                results = [e for e in results if e["type"] == event_type]
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"event_count": len(results), "event_type": event_type},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Coord events reflection: %s", reflection.get("reflection", ""))
            return results
        except Exception as e:
            logger.error("Coordination retrieval failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.get_coordination_events(event_type, task_type),
                default=[],
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def analyze_coordination_events(
        self, event_type: Optional[str] = None, task_type: str = ""
    ) -> Dict[str, Any]:
        if event_type is not None and not isinstance(event_type, str):
            raise TypeError("event_type must be a string or None")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        try:
            events = await self.get_coordination_events(event_type, task_type)
            if not events:
                return {
                    "status": "error",
                    "error": "No coordination events found",
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                }

            drift_count = sum(1 for e in events if e["type"] == "drift")
            consensus_events = [
                e for e in events if e["event"].get("event") == "run_consensus_protocol"
            ]
            consensus_count = sum(
                1 for e in consensus_events if e["event"].get("output", {}).get("status") == "success"
            )

            agent_counts = Counter(
                [
                    agent_id
                    for e in events
                    for agent_id in e["agent_metadata"].get("agent_ids", [])
                ]
            )
            avg_confidence = (
                np.mean(
                    [
                        (sum(conf.values()) / len(conf)) if conf else 0.5
                        for e in consensus_events
                        for conf in [e["event"]["output"].get("weights", {})]
                    ]
                )
                if consensus_events
                else 0.5
            )

            analysis = {
                "status": "success",
                "metrics": {
                    "drift_frequency": drift_count / len(events),
                    "consensus_success_rate": consensus_count / len(consensus_events)
                    if consensus_events
                    else 0.0,
                    "agent_participation": dict(agent_counts),
                    "avg_confidence_score": float(avg_confidence),
                    "event_count": len(events),
                },
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Coordination Analysis",
                    analysis,
                    module="ContextManager",
                    tags=["coordination", "analytics", event_type or "all", task_type],
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=analysis, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Coord analysis reflection: %s", reflection.get("reflection", ""))
            await self.log_event_with_hash(
                {"event": "coordination_analysis", "analysis": analysis}, task_type=task_type
            )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "coordination_analysis": {
                            "metrics": analysis["metrics"],
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return analysis
        except Exception as e:
            logger.error("Coordination analysis failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.analyze_coordination_events(event_type, task_type),
                default={
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                },
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def get_drift_trends(
        self, time_window_hours: float = 24.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            raise ValueError("time_window_hours must be positive")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            events = await self.get_coordination_events("drift", task_type)
            if not events:
                return {
                    "status": "error",
                    "error": "No drift events found",
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                }

            now = datetime.now()
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= cutoff]

            drift_names = Counter(
                e["event"].get("drift", {}).get("name", "unknown") for e in events
            )
            similarity_scores = [
                e["event"].get("drift", {}).get("similarity", 0.5)
                for e in events
                if "drift" in e["event"] and "similarity" in e["event"]["drift"]
            ]
            trend_data = {
                "status": "success",
                "trends": {
                    "drift_names": dict(drift_names),
                    "avg_similarity": float(np.mean(similarity_scores))
                    if similarity_scores
                    else 0.5,
                    "event_count": len(events),
                    "time_window_hours": time_window_hours,
                },
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Drift Trends Analysis",
                    trend_data,
                    module="ContextManager",
                    tags=["drift", "trends", task_type],
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=trend_data, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift trends reflection: %s", reflection.get("reflection", ""))
            await self.log_event_with_hash(
                {"event": "drift_trends", "trends": trend_data}, task_type=task_type
            )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "drift_trends": {
                            "trends": trend_data["trends"],
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return trend_data
        except Exception as e:
            logger.error("Drift trends analysis failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.get_drift_trends(time_window_hours, task_type),
                default={
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                },
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def generate_coordination_chart(
        self, metric: str = "drift_frequency", time_window_hours: float = 24.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if metric not in ["drift_frequency", "consensus_success_rate", "avg_confidence_score"]:
            raise ValueError(
                "metric must be 'drift_frequency', 'consensus_success_rate', or 'avg_confidence_score'"
            )
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            raise ValueError("time_window_hours must be positive")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            events = await self.get_coordination_events(task_type=task_type)
            if not events:
                return {
                    "status": "error",
                    "error": "No coordination events found",
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                }

            now = datetime.now()
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= cutoff]

            time_bins: Dict[str, List[Dict[str, Any]]] = {}
            for e in events:
                ts = datetime.fromisoformat(e["timestamp"])
                hour_key = ts.strftime("%Y-%m-%dT%H:00:00")
                time_bins.setdefault(hour_key, []).append(e)

            labels = sorted(time_bins.keys())
            data = []
            for hour in labels:
                hour_events = time_bins[hour]
                if metric == "drift_frequency":
                    value = (
                        sum(1 for e in hour_events if e["type"] == "drift") / len(hour_events)
                        if hour_events
                        else 0.0
                    )
                elif metric == "consensus_success_rate":
                    consensus = [
                        e for e in hour_events if e["event"].get("event") == "run_consensus_protocol"
                    ]
                    value = (
                        sum(
                            1
                            for e in consensus
                            if e["event"].get("output", {}).get("status") == "success"
                        )
                        / len(consensus)
                        if consensus
                        else 0.0
                    )
                else:  # avg_confidence_score
                    confidences = [
                        (sum(conf.values()) / len(conf)) if conf else 0.5
                        for e in hour_events
                        for conf in [e["event"].get("output", {}).get("weights", {})]
                        if e["event"].get("event") == "run_consensus_protocol"
                    ]
                    value = float(np.mean(confidences)) if confidences else 0.5
                data.append(value)

            chart_config = {
                "type": "line",
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": metric.replace("_", " ").title(),
                            "data": data,
                            "borderColor": "#2196F3",
                            "backgroundColor": "#2196F380",
                            "fill": True,
                            "tension": 0.4,
                        }
                    ],
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {"display": True, "text": metric.replace("_", " ").title()},
                        },
                        "x": {"title": {"display": True, "text": "Time"}},
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{metric.replace('_', ' ').title()} Over Time (Task: {task_type})",
                        }
                    },
                },
            }

            result = {
                "status": "success",
                "chart": chart_config,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Coordination Chart Generated",
                    result,
                    module="ContextManager",
                    tags=["coordination", "visualization", metric, task_type],
                )
            await self.log_event_with_hash(
                {"event": "generate_coordination_chart", "chart": chart_config, "metric": metric},
                task_type=task_type,
            )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "coordination_chart": {
                            "metric": metric,
                            "chart_config": chart_config,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=result, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Chart generation reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("Chart generation failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.generate_coordination_chart(metric, time_window_hours, task_type),
                default={
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                },
                diagnostics=diagnostics,
                task_type=task_type,
            )

    # ── Υ SharedGraph: add/diff/merge hooks (SYNC API) ────────────────────────
    def _push_to_shared_graph(self, task_type: str = "") -> None:
        """Publish current context view to SharedGraph (best‑effort, synchronous)."""
        if not self.shared_graph:
            return
        try:
            view = {
                "nodes": [
                    {
                        "id": f"ctx_{hashlib.md5((self.current_context.get('goal_id','') + task_type).encode('utf-8')).hexdigest()[:8]}",
                        "layer": self.current_context.get("layer", "local"),
                        "intent": self.current_context.get("intent"),
                        "goal_id": self.current_context.get("goal_id"),
                        "task_type": task_type or self.current_context.get("task_type", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "edges": [],
                "context": self.current_context,  # retained for peer policies
            }
            # external_agent_bridge.SharedGraph.add(view) is synchronous
            self.shared_graph.add(view)
            # Log a lightweight event (avoid storing full context again)
            asyncio.create_task(
                self.log_event_with_hash(
                    {"event": "shared_graph_add", "agent_coordination": True}, task_type=task_type
                )
            )
        except Exception as e:
            logger.warning("SharedGraph add failed: %s", e)

    def reconcile_with_peers(
        self,
        peer_graph: Optional[ExternalAgentBridge] = None,
        strategy: str = "prefer_recent",
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Diff against a peer's graph and (optionally) merge using SharedGraph strategy.
        SharedGraph strategies: 'prefer_recent' (default), 'prefer_majority'
        """
        if not self.shared_graph:
            return {"status": "error", "error": "SharedGraph unavailable", "task_type": task_type}
        try:
            diff_result = None
            if peer_graph and isinstance(peer_graph, ExternalAgentBridge):
                diff_result = self.shared_graph.diff(peer_graph)
            else:
                # No peer provided → no diff possible (return a stub so callers can proceed)
                diff_result = {"added": [], "removed": [], "conflicts": [], "ts": time.time()}

            decision = {"apply_merge": False, "reason": "no conflicts"}
            if diff_result and diff_result.get("conflicts"):
                # Simple policy: allow merge when conflicts are non‑ethical keys
                non_ethical = all(
                    "ethic" not in str(c.get("key", "")).lower() for c in diff_result["conflicts"]
                )
                if non_ethical:
                    decision = {
                        "apply_merge": True,
                        "reason": "non‑ethical conflicts",
                        "strategy": strategy if strategy in ("prefer_recent", "prefer_majority") else "prefer_recent",
                    }

            merged = None
            if decision.get("apply_merge"):
                merged = self.shared_graph.merge(decision["strategy"])
                # Optionally refresh local context if a merged context node exists
                merged_ctx = (merged or {}).get("context")
                if isinstance(merged_ctx, dict):
                    # Schedule async update without blocking caller
                    asyncio.create_task(self.update_context(merged_ctx, task_type=task_type))
                asyncio.create_task(
                    self.log_event_with_hash(
                        {
                            "event": "shared_graph_merge",
                            "strategy": decision["strategy"],
                            "result_keys": list((merged or {}).keys()),
                            "agent_coordination": True,
                        },
                        task_type=task_type,
                    )
                )

            return {"status": "success", "diff": diff_result, "decision": decision, "merged": merged, "task_type": task_type}
        except Exception as e:
            logger.error("Peer reconciliation failed: %s", e)
            return {"status": "error", "error": str(e), "task_type": task_type}

    # ── Φ⁰ gated hooks ────────────────────────────────────────────────────────
    async def _reality_sculpt_hook(self, event: str, payload: Dict[str, Any]) -> None:
        """No‑op unless STAGE_IV is enabled. Intended for Φ⁰ Reality Sculpting pre/post modulations."""
        if not STAGE_IV:
            return
        try:
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Φ⁰ Hook",
                    {"event": event, "payload": payload},
                    module="ContextManager",
                    tags=["phi0", event],
                )
        except Exception as e:
            logger.debug("Φ⁰ hook skipped: %s", e)

    # ── Self-Healing Cognitive Pathways (centralized) ─────────────────────────
    async def _self_heal(
        self,
        err: str,
        retry: Callable[[], Any],
        default: Any,
        diagnostics: Dict[str, Any],
        task_type: str,
        propose_plan: bool = False,
    ):
        """Route errors through error_recovery with optional recursive plan proposal."""
        try:
            plan = None
            if propose_plan and self.recursive_planner:
                propose = getattr(self.recursive_planner, "propose_recovery_plan", None)
                if callable(propose):
                    plan = await propose(err=err, context=self.current_context, task_type=task_type)

            handler = getattr(self.error_recovery, "handle_error", None)
            if callable(handler):
                return await handler(
                    err,
                    retry_func=retry,
                    default=default,
                    diagnostics={"self_diag": diagnostics, "plan": plan} if plan else diagnostics,
                )
        except Exception as inner:
            logger.warning("Self-heal pathway failed: %s", inner)
        return default

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _safe_state_snapshot(self) -> Dict[str, Any]:
        """Create a serialization‑safe snapshot of internal state (no callables)."""
        return {
            "current_context": self.current_context,
            "context_history_len": len(self.context_history),
            "event_log_len": len(self.event_log),
            "coordination_log_len": len(self.coordination_log),
            "rollback_threshold": self.rollback_threshold,
            "flags": {"STAGE_IV": STAGE_IV},
        }





@lru_cache(maxsize=100)
def _epsilon_identity_cached(timestamp: float) -> float:
    """Cached ε-identity function to avoid re-computation."""
    return 0.1

class UserProfile:
    """Manages user profiles, preferences, and identity tracking in ANGELA v3.5.2."""

    DEFAULT_PREFERENCES = {
        "style": "neutral",
        "language": "en",
        "output_format": "concise",
        "theme": "light"
    }

    def __init__(self, storage_path: str = "user_profiles.json", orchestrator: Optional[SimulationCore] = None) -> None:
        """Initialize UserProfile with storage path and orchestrator."""
        if not isinstance(storage_path, str):
            logger.error("Invalid storage_path: must be a string")
            raise TypeError("storage_path must be a string")
        self.storage_path = storage_path
        self.profile_lock = Lock()
        self.profiles: Dict[str, Dict] = {}
        self.active_user: Optional[str] = None
        self.active_agent: Optional[str] = None
        self.orchestrator = orchestrator
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.memory_manager = orchestrator.memory_manager if orchestrator and getattr(orchestrator, "memory_manager", None) else MemoryManager()
        self.multi_modal_fusion = orchestrator.multi_modal_fusion if orchestrator and getattr(orchestrator, "multi_modal_fusion", None) else MultiModalFusion(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.meta_cognition = orchestrator.meta_cognition if orchestrator and getattr(orchestrator, "meta_cognition", None) else MetaCognition(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.reasoning_engine = orchestrator.reasoning_engine if orchestrator and getattr(orchestrator, "reasoning_engine", None) else ReasoningEngine(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager,
            multi_modal_fusion=self.multi_modal_fusion, meta_cognition=self.meta_cognition)
        self.toca_engine = getattr(orchestrator, "toca_engine", None) if orchestrator else None

        self._load_profiles()
        logger.info("UserProfile initialized with storage_path=%s", storage_path)

    def _rehydrate_deques(self) -> None:
        for u in self.profiles:
            for a in self.profiles[u]:
                d = self.profiles[u][a].get("identity_drift", [])
                if not isinstance(d, deque):
                    self.profiles[u][a]["identity_drift"] = deque(d, maxlen=1000)

    def _serialize_profiles(self, profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable copy (converts deques to lists)."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, deque):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj
        return _convert(profiles)

    def _load_profiles(self) -> None:
        """Load user profiles from storage."""
        with self.profile_lock:
            try:
                profile_path = Path(self.storage_path)
                if profile_path.exists():
                    with profile_path.open("r", encoding="utf-8") as f:
                        self.profiles = json.load(f)
                    self._rehydrate_deques()
                    logger.info("User profiles loaded from %s", self.storage_path)
                else:
                    self.profiles = {}
                    logger.info("No profiles found. Initialized empty profiles store.")
            except json.JSONDecodeError as e:
                logger.error("Failed to parse profiles JSON: %s", str(e))
                self.profiles = {}
            except PermissionError as e:
                logger.error("Permission denied accessing %s: %s", self.storage_path, str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error loading profiles: %s", str(e))
                raise

    def _save_profiles(self) -> None:
        """Save user profiles to storage atomically."""
        with self.profile_lock:
            try:
                profile_path = Path(self.storage_path)
                tmp_path = profile_path.with_suffix(".tmp")
                data = self._serialize_profiles(self.profiles)
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, profile_path)
                logger.info("User profiles saved to %s", self.storage_path)
            except PermissionError as e:
                logger.error("Permission denied saving %s: %s", self.storage_path, str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error saving profiles: %s", str(e))
                raise

    async def switch_user(self, user_id: str, agent_id: str = "default", task_type: str = "") -> None:
        """Switch to a user and agent profile with task-specific processing."""
        if not isinstance(user_id, str) or not user_id:
            logger.error("Invalid user_id: must be a non-empty string for task %s", task_type)
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(agent_id, str) or not agent_id:
            logger.error("Invalid agent_id: must be a non-empty string for task %s", task_type)
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            with self.profile_lock:
                if user_id not in self.profiles:
                    logger.info("Creating new profile for user '%s' for task %s", user_id, task_type)
                    self.profiles[user_id] = {}

                if agent_id not in self.profiles[user_id]:
                    self.profiles[user_id][agent_id] = {
                        "preferences": self.DEFAULT_PREFERENCES.copy(),
                        "audit_log": [],
                        "identity_drift": deque(maxlen=1000)
                    }
                    self._save_profiles()

                self.active_user = user_id
                self.active_agent = agent_id

            logger.info("Active profile: %s::%s for task %s", user_id, agent_id, task_type)

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="User Switched",
                    meta={"user_id": user_id, "agent_id": agent_id, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["user_switch", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"User_Switch_{datetime.now().isoformat()}",
                    output={"user_id": user_id, "agent_id": agent_id, "task_type": task_type, "policies": policies},
                    layer="Profiles",
                    intent="user_switch",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=f"Switched to user {user_id} and agent {agent_id}",
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("User switch reflection for task %s: %s", task_type, reflection.get("reflection", ""))
        except Exception as e:
            logger.error("User switch failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.switch_user(user_id, agent_id, task_type),
                    default=None
                )
            raise


class MemoryManagerLike(Protocol):
    async def store(self, query: str, output: Any, *, layer: str, intent: str, task_type: str = "") -> None: ...
    async def retrieve(self, query: str, *, layer: str, task_type: str = "") -> Any: ...
    async def search(self, *, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]: ...




class CodeExecutor:
    """Safely execute code snippets (Python/JS/Lua) with task-aware validation and logging."""

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        safe_mode: bool = True,
        alignment_guard: Optional[AlignmentGuard] = None,
        memory_manager: Optional[MemoryManager] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
    ) -> None:
        self.safe_mode = safe_mode
        self.safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
        }
        self.supported_languages = ["python", "javascript", "lua"]
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator and AGIEnhancer else None
        self.alignment_guard = alignment_guard
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        logger.info("CodeExecutor initialized (safe_mode=%s)", safe_mode)

    async def integrate_external_execution_context(
        self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = ""
    ) -> Dict[str, Any]:
        """Integrate real-world execution context or security policies for code execution."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        cache_key = f"ExecutionContext_{data_type}_{data_source}_{task_type}"

        try:
            # Try cached first
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if cached and isinstance(cached, dict) and "timestamp" in cached and "result" in cached:
                    cache_time = datetime.fromisoformat(cached["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached execution context for %s", cache_key)
                        return cached["result"]

            # If network lib missing, return error (caller may fallback)
            if aiohttp is None:
                logger.warning("aiohttp not available; cannot fetch external execution context")
                return {"status": "error", "error": "aiohttp_not_available"}

            # Fetch
            async with aiohttp.ClientSession() as session:
                url = f"https://x.ai/api/execution_context?source={data_source}&type={data_type}"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error("Failed to fetch execution context: HTTP %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "security_policies":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No security policies provided")
                    result = {"status": "error", "error": "No policies"}
                else:
                    result = {"status": "success", "policies": policies}
            elif data_type == "execution_context":
                context = data.get("context", {})
                if not context:
                    logger.error("No execution context provided")
                    result = {"status": "error", "error": "No context"}
                else:
                    result = {"status": "success", "context": context}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager and result.get("status") == "success":
                await self.memory_manager.store(
                    cache_key,
                    {"result": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="execution_context_integration",
                    task_type=task_type,
                )

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="CodeExecutor",
                        output={"data_type": data_type, "data": result},
                        context={"task_type": task_type},
                    )
                    if reflection.get("status") == "success":
                        logger.info("Execution context integration reflection: %s", reflection.get("reflection", ""))
                except Exception:  # pragma: no cover
                    logger.debug("Meta-cognition reflection failed (integration).")

            return result
        except Exception as e:  # pragma: no cover
            logger.error("Execution context integration failed: %s", str(e))
            try:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
                _ = diagnostics  # not used, but preserved for future logging
            except Exception:
                pass
            return {"status": "error", "error": str(e)}

    async def execute(self, code_snippet: str, language: str = "python", timeout: float = 5.0, task_type: str = "") -> Dict[str, Any]:
        """Execute a code snippet in the specified language with task-specific validation."""
        if not isinstance(code_snippet, str):
            logger.error("Invalid code_snippet type: must be a string.")
            raise TypeError("code_snippet must be a string")
        if not isinstance(language, str):
            logger.error("Invalid language type: must be a string.")
            raise TypeError("language must be a string")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            logger.error("Invalid timeout: must be a positive number.")
            raise ValueError("timeout must be a positive number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        language = language.lower()
        if language not in self.supported_languages:
            logger.error("Unsupported language: %s", language)
            return {"error": f"Unsupported language: {language}", "success": False, "task_type": task_type}

        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(code_snippet, stage="pre", task_type=task_type)
            if not valid:
                logger.warning("Code snippet failed alignment check for task %s.", task_type)
                await self._log_episode(
                    "Code Alignment Failure",
                    {"code": code_snippet, "report": report, "task_type": task_type},
                    ["alignment", "failure", task_type],
                )
                return {"error": "Code snippet failed alignment check", "success": False, "task_type": task_type}

        # Try to load security policies; fallback softly if unavailable
        security_policies = await self.integrate_external_execution_context(
            data_source="xai_security_db", data_type="security_policies", task_type=task_type
        )
        if security_policies.get("status") != "success":
            logger.warning("Security policies unavailable, proceeding with minimal policy set.")
            security_policies = {"status": "success", "policies": []}

        # Adaptive timeout
        risk_bias = 0.5
        resilience = 0.5
        adjusted_timeout = max(1, min(30, int(timeout * max(0.1, resilience) * (1.0 + 0.5 * max(0.0, risk_bias)))))
        logger.debug("Adaptive timeout: %ss for task %s", adjusted_timeout, task_type)

        await self._log_episode(
            "Code Execution",
            {"language": language, "code": code_snippet, "task_type": task_type},
            ["execution", language, task_type],
        )

        if language == "python":
            result = await self._execute_python(code_snippet, adjusted_timeout, task_type)
        elif language == "javascript":
            result = await self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "javascript", task_type)
        elif language == "lua":
            result = await self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "lua", task_type)
        else:  # pragma: no cover
            result = {"error": f"Unsupported language: {language}", "success": False}

        result["task_type"] = task_type
        await self._log_result(result)

        if self.memory_manager:
            key = f"CodeExecution_{language}_{time.strftime('%Y%m%d_%H%M%S')}"
            try:
                await self.memory_manager.store(
                    key,
                    result,
                    layer="SelfReflections",
                    intent="code_execution",
                    task_type=task_type,
                )
            except Exception:  # pragma: no cover
                logger.debug("MemoryManager.store failed for key %s", key)

        if self.visualizer and task_type:
            try:
                plot_data = {
                    "code_execution": {
                        "language": language,
                        "success": result.get("success", False),
                        "error": result.get("error", ""),
                        "task_type": task_type,
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise",
                    },
                }
                await self.visualizer.render_charts(plot_data)
            except Exception:  # pragma: no cover
                logger.debug("Visualizer render_charts failed.")

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CodeExecutor",
                    output=result,
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Execution reflection: %s", reflection.get("reflection", ""))
            except Exception:  # pragma: no cover
                logger.debug("Meta-cognition reflection failed (execution).")

        return result

    async def _execute_python(self, code_snippet: str, timeout: float, task_type: str = "") -> Dict[str, Any]:
        """Execute a Python code snippet safely. Falls back to legacy mode if RestrictedPython unavailable."""
        if self.safe_mode:
            try:
                from RestrictedPython import compile_restricted
                from RestrictedPython.Guards import safe_builtins as rp_safe_builtins
                exec_func = lambda code, env: exec(  # noqa: E731
                    compile_restricted(code, "<string>", "exec"),
                    {"__builtins__": rp_safe_builtins},
                    env,
                )
            except Exception:
                logger.warning("RestrictedPython not available; falling back to legacy safe_builtins for task %s.", task_type)
                exec_func = lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env)  # noqa: E731
        else:
            logger.warning("Executing in legacy mode (unrestricted) for task %s.", task_type)
            exec_func = lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env)  # noqa: E731

        return await self._capture_execution(code_snippet, exec_func, "python", timeout, task_type)

    async def _capture_execution(
        self,
        code_snippet: str,
        executor: Callable[[str, Dict[str, Any]], None],
        label: str,
        timeout: float = 5.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """Capture execution output and errors (with enforced timeout)."""
        exec_locals: Dict[str, Any] = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        async def _run():
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: executor(code_snippet, exec_locals))

        try:
            await asyncio.wait_for(_run(), timeout=timeout)
            return {
                "language": label,
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True,
                "task_type": task_type,
            }
        except asyncio.TimeoutError:
            logger.warning("%s timeout after %ss for task %s", label, timeout, task_type)
            return {
                "language": label,
                "error": f"{label} timeout after {timeout}s",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False,
                "task_type": task_type,
            }
        except Exception as e:
            return {
                "language": label,
                "error": str(e),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False,
                "task_type": task_type,
            }

    async def _execute_subprocess(
        self, command: List[str], timeout: float, label: str, task_type: str = ""
    ) -> Dict[str, Any]:
        """Execute code via subprocess for non-Python languages."""
        interpreter = command[0]
        if not shutil.which(interpreter):
            logger.error("%s not found in system PATH for task %s", interpreter, task_type)
            return {
                "language": label,
                "error": f"{interpreter} not found in system PATH",
                "success": False,
                "task_type": task_type,
            }
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            stdout_s, stderr_s = stdout.decode(), stderr.decode()
            if process.returncode != 0:
                return {
                    "language": label,
                    "error": f"{label} execution failed",
                    "stdout": stdout_s,
                    "stderr": stderr_s,
                    "success": False,
                    "task_type": task_type,
                }
            return {
                "language": label,
                "stdout": stdout_s,
                "stderr": stderr_s,
                "success": True,
                "task_type": task_type,
            }
        except asyncio.TimeoutError:
            logger.warning("%s timeout after %ss for task %s", label, timeout, task_type)
            return {"language": label, "error": f"{label} timeout after {timeout}s", "success": False, "task_type": task_type}
        except Exception as e:
            logger.error("Subprocess error: %s for task %s", str(e), task_type)
            return {"language": label, "error": str(e), "success": False, "task_type": task_type}

    async def _log_episode(self, title: str, content: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        """Log an episode to the AGI enhancer or locally."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
            try:
                await self.agi_enhancer.log_episode(title, content, module="CodeExecutor", tags=tags or [])
            except Exception:  # pragma: no cover
                logger.debug("agi_enhancer.log_episode failed")
        else:
            logger.debug("Episode: %s | %s | tags=%s", title, list(content.keys()), tags)

        if self.meta_cognition and content.get("task_type"):
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CodeExecutor", output={"title": title, "content": content}, context={"task_type": content.get("task_type")}
                )
                if reflection.get("status") == "success":
                    logger.info("Episode log reflection: %s", reflection.get("reflection", ""))
            except Exception:  # pragma: no cover
                logger.debug("Meta-cognition reflection failed (episode).")

    async def _log_result(self, result: Dict[str, Any]) -> None:
        """Log the execution result."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_explanation"):
            try:
                tag = "success" if result.get("success") else "failure"
                await self.agi_enhancer.log_explanation(f"Code execution {tag}:", trace=result)
            except Exception:  # pragma: no cover
                logger.debug("agi_enhancer.log_explanation failed")
        else:
            logger.debug("Execution result logged (local).")

        if self.meta_cognition and result.get("task_type"):
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CodeExecutor", output=result, context={"task_type": result.get("task_type")}
                )
                if reflection.get("status") == "success":
                    logger.info("Result log reflection: %s", reflection.get("reflection", ""))
            except Exception:  # pragma: no cover
                logger.debug("Meta-cognition reflection failed (result).")


class ErrorRecoveryLike(Protocol):
    async def handle_error(self,
                           error_msg: str,
                           *,
                           retry_func: Optional[Callable[[], Awaitable[Any]]] = None,
                           default: Any = None,
                           diagnostics: Optional[Dict[str, Any]] = None) -> Any: ...


class EmbodiedAgent(TimeChainMixin):
    def __init__(self, name: str, traits: Dict[str, float], memory_manager: MemoryManager, meta_cognition: MetaCognition, agi_enhancer: AGIEnhancer) -> None:
        self.name = name
        self.traits = traits
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition
        self.agi_enhancer = agi_enhancer
        self.state: dict[str, float] = {}
        self.previous_state: dict[str, float] = {}
        self.ontology: dict[str, Any] = {}
        self.dream_layer = meta_cognition_module.DreamOverlayLayer()  # v5.0.2 co-dream
        logger.info("EmbodiedAgent %s initialized", name)

    async def process_input(self, input_data: str, task_type: str = "") -> str:
        t = time.time() % 1.0
        modulated_traits = {k: v * (1 + epsilon_emotion(t)) for k, v in self.traits.items()}
        self.previous_state = self.state.copy()
        self.state = modulated_traits
        drift = self.agi_enhancer.detect_ontology_drift(self.state, self.previous_state)
        if drift > 0.2:
            await self.agi_enhancer.coordinate_drift_mitigation([self], task_type=task_type)
        result = f"Processed: {input_data} with traits {modulated_traits}"
        await self.memory_manager.store(input_data, result, layer="STM", task_type=task_type)
        self.log_timechain_event("EmbodiedAgent", f"Processed input: {input_data}")
        return result

    async def introspect(self, query: str, task_type: str = "") -> Dict[str, Any]:
        introspection = await self.meta_cognition.introspect(query, task_type=task_type)
        return introspection

    def activate_dream_mode(self, peers: list | None = None, lucidity_mode: dict | None = None, resonance_targets: list | None = None, safety_profile: str = "sandbox") -> dict[str, Any]:
        return self.dream_layer.activate_dream_mode(peers=peers, lucidity_mode=lucidity_mode, resonance_targets=resonance_targets, safety_profile=safety_profile)


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


class EntailmentReasoningEnhancer:
    def __init__(self):
        logger.info("EntailmentReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with entailment: {input_text}"




def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class AlignmentGuard:
    """Ethical validation & drift analysis for ANGELA (clean refactor + τ wiring).

    All external effects are injectable. If not provided, safe no‑ops are used
    so the module can run in isolation and tests.
    """

    def __init__(
        self,
        *,
        context_manager: Optional[ContextManagerLike] = None,
        error_recovery: Optional[ErrorRecoveryLike] = None,
        memory_manager: Optional[MemoryManagerLike] = None,
        concept_synthesizer: Optional[ConceptSynthesizerLike] = None,
        meta_cognition: Optional[MetaCognitionLike] = None,
        visualizer: Optional[VisualizerLike] = None,
        llm: Optional[LLMClient] = None,
        http: Optional[HTTPClient] = None,
        reasoning_engine: Optional[ReasoningEngineLike] = None,
        ethical_threshold: float = 0.8,
        drift_validation_threshold: float = 0.7,
        trait_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.context_manager = context_manager
        self.error_recovery: ErrorRecoveryLike = error_recovery or NoopErrorRecovery()
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        self.llm: LLMClient = llm or NoopLLM()
        self.http: HTTPClient = http or NoopHTTP()
        self.reasoning_engine = reasoning_engine  # may be None; τ will fallback

        self.validation_log: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.ethical_threshold: float = float(ethical_threshold)
        self.drift_validation_threshold: float = float(drift_validation_threshold)
        self.trait_weights: Dict[str, float] = {
            "eta_empathy": 0.5,
            "mu_morality": 0.5,
            **(trait_weights or {}),
        }
        logger.info(
            "AlignmentGuard initialized (ethical=%.2f, drift=%.2f, τ-wired=%s)",
            self.ethical_threshold,
            self.drift_validation_threshold,
            "yes" if self.reasoning_engine else "no",
        )

    # --- External data -------------------------------------------------------

    async def integrate_external_ethics_data(
        self,
        *,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """Optionally pull real‑world guidelines/conflict data.

        In this refactor we avoid hardcoding a network endpoint. Callers can
        inject an HTTP client that resolves `data_source` to a URL. If not
        provided, returns a success envelope with empty payload.
        """
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non‑negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            # Simple memory cache via MemoryManager if available
            cache_key = f"EthicsData::{data_type}::{data_source}::{task_type}"
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict):
                    ts = cached.get("timestamp")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            if (datetime.now(dt.tzinfo or timezone.utc) - dt).total_seconds() < cache_timeout:
                                logger.info("Returning cached ethics data for %s", cache_key)
                                return cached.get("data", {})
                        except Exception:
                            pass

            # Resolve with injected HTTP client (may be a no‑op)
            url = data_source  # allow callers to pass a full URL, or a key their HTTP client understands
            data = await self.http.get_json(url)

            result: Dict[str, Any]
            if data_type == "ethical_guidelines":
                result = {"status": "success", "guidelines": data.get("guidelines", [])}
            elif data_type == "conflict_data":
                result = {"status": "success", "conflict_data": data.get("conflict_data", {})}
            else:
                result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": _utc_now_iso()},
                    layer="ExternalData",
                    intent="ethics_data_integration",
                    task_type=task_type,
                )

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard",
                        output={"data_type": data_type, "data": result},
                        context={"task_type": task_type},
                    )
                    if reflection.get("status") == "success":
                        logger.info("Ethics data reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return result
        except Exception as e:
            logger.error("Ethics data integration failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_ethics_data(
                    data_source=data_source,
                    data_type=data_type,
                    cache_timeout=cache_timeout,
                    task_type=task_type,
                ),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
            )

    # --- Core checks ---------------------------------------------------------

    async def check(self, prompt: str, *, task_type: str = "") -> bool:
        """Return True if a prompt is ethically aligned.

        The LLM is expected to return JSON with a numeric `score` in [0,1].
        Non‑JSON responses are handled gracefully; missing scores default to 0.0.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non‑empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Ethical check (task=%s)", task_type)
        try:
            t = time.time() % 1.0
            trait_mod = (
                self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t)
                + self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
            )

            validation_prompt = (
                "Evaluate ethical alignment (return JSON: {\"score\": number 0..1}).\n"
                f"Task: {task_type or 'general'}\n"
                f"EmpathyWeight={self.trait_weights.get('eta_empathy', 0.5):.2f}, "
                f"MoralityWeight={self.trait_weights.get('mu_morality', 0.5):.2f}\n"
                "Content:\n" + prompt.strip()
            )

            raw = await self.llm.generate(validation_prompt, model="gpt-4", temperature=0.3)
            parsed = _parse_llm_jsonish(raw)
            score = _as_float(parsed.get("score", 0.0))
            valid = bool(score >= self.ethical_threshold)

            entry = {
                "prompt": prompt[:200],
                "score": score,
                "valid": valid,
                "trait_modulation": trait_mod,
                "timestamp": time.time(),
                "task_type": task_type,
            }
            self.validation_log.append(entry)

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({
                        "event": "ethical_check",
                        "prompt": prompt[:200],
                        "score": score,
                        "valid": valid,
                        "task_type": task_type,
                    })
                except Exception:
                    logger.debug("Context logging failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "ethical_check": {
                            "prompt": prompt[:200],
                            "score": score,
                            "valid": valid,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=entry, context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Ethical check reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return valid
        except Exception as e:
            logger.error("Ethical check failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.check(prompt, task_type=task_type),
                default=False,
                diagnostics=diagnostics,
            )

    async def ethical_check(self, content: str, *, stage: str = "pre", task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Wrapper that records a detailed report and persists it if configured."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non‑empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            valid = await self.check(content, task_type=task_type)
            report = {
                "stage": stage,
                "content": content[:200],
                "valid": valid,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"EthicalCheck::{stage}::{_utc_now_iso()}",
                        output=report,
                        layer="SelfReflections",
                        intent="ethical_check",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "ethical_check_report": {
                            "stage": stage,
                            "content": content[:200],
                            "valid": valid,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")
            return valid, report
        except Exception as e:
            logger.error("Ethical check(report) failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.ethical_check(content, stage=stage, task_type=task_type),
                default=(False, {"stage": stage, "error": str(e), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    async def audit(self, action: str, *, context: Optional[str] = None, task_type: str = "") -> str:
        """Audit an action and return "clear" | "flagged" | "audit_error"."""
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non‑empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            valid = await self.check(action, task_type=task_type)
            status = "clear" if valid else "flagged"
            entry = {
                "action": action[:200],
                "context": context,
                "status": status,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }
            self.validation_log.append(entry)
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"Audit::{_utc_now_iso()}",
                        output=entry,
                        layer="SelfReflections",
                        intent="audit",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "audit": {
                            "action": action[:200],
                            "status": status,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")
            return status
        except Exception as e:
            logger.error("Audit failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.audit(action, context=context, task_type=task_type),
                default="audit_error",
                diagnostics=diagnostics,
            )

    # --- Drift & trait validations ------------------------------------------

    async def simulate_and_validate(self, drift_report: Dict[str, Any], *, task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Validate ontology drift with optional concept & ethics checks."""
        required = {"name", "from_version", "to_version", "similarity"}
        if not isinstance(drift_report, dict) or not required.issubset(drift_report):
            raise ValueError("drift_report must include name, from_version, to_version, similarity")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            t = time.time() % 1.0
            trait_mod = (
                self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t)
                + self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
            )

            valid = True
            issues: List[str] = []

            # Check prior drift memory if available
            if self.memory_manager and task_type:
                try:
                    prior = await self.memory_manager.search(
                        query_prefix=drift_report["name"],
                        layer="SelfReflections",
                        intent="ontology_drift",
                        task_type=task_type,
                    )
                    if prior:
                        latest = prior[0]
                        sim = _as_float(latest.get("output", {}).get("similarity", 1.0), 1.0)
                        if sim < self.drift_validation_threshold:
                            valid = False
                            issues.append(f"Previous drift similarity {sim:.2f} below threshold")
                except Exception:
                    logger.debug("Memory search failed; continuing")

            # Concept synthesizer comparison if provided
            if self.concept_synthesizer and "definition" in drift_report:
                try:
                    symbol = self.concept_synthesizer.get_symbol(drift_report["name"])
                    if symbol and symbol.get("version") == drift_report["from_version"]:
                        comp = await self.concept_synthesizer.compare(
                            symbol.get("definition", {}).get("concept", ""),
                            drift_report.get("definition", {}).get("concept", ""),
                            task_type=task_type,
                        )
                        score = _as_float(comp.get("score", 1.0), 1.0)
                        if score < self.drift_validation_threshold:
                            valid = False
                            issues.append(
                                f"Similarity {score:.2f} below threshold {self.drift_validation_threshold:.2f}"
                            )
                except Exception:
                    logger.debug("Concept comparison failed; continuing")

            # Ethics guideline check (optional external)
            ethics = await self.integrate_external_ethics_data(
                data_source="https://example.ethics/guidelines",  # placeholder; HTTP client may override
                data_type="ethical_guidelines",
                task_type=task_type,
            )
            guidelines = ethics.get("guidelines", []) if ethics.get("status") == "success" else []

            validation_prompt = {
                "name": drift_report.get("name"),
                "from_version": drift_report.get("from_version"),
                "to_version": drift_report.get("to_version"),
                "similarity": drift_report.get("similarity"),
                "guidelines": guidelines,
                "task_type": task_type,
                "weights": {
                    "eta_empathy": self.trait_weights.get("eta_empathy", 0.5),
                    "mu_morality": self.trait_weights.get("mu_morality", 0.5),
                },
                "request": "Return JSON {valid: bool, issues: string[]}"
            }

            raw = await self.llm.generate(json.dumps(validation_prompt), model="gpt-4", temperature=0.2)
            parsed = _parse_llm_jsonish(raw)
            ethical_valid = bool(parsed.get("valid", True))
            if not ethical_valid:
                valid = False
                issues.extend([str(i) for i in parsed.get("issues", ["Ethical misalignment detected"])])

            report = {
                "drift_name": drift_report.get("name"),
                "similarity": drift_report.get("similarity"),
                "trait_modulation": trait_mod,
                "issues": issues,
                "valid": valid,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }
            self.validation_log.append(report)

            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"DriftValidation::{drift_report.get('name')}::{_utc_now_iso()}",
                        output=report,
                        layer="SelfReflections",
                        intent="ontology_drift_validation",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({
                        "event": "drift_validation",
                        "drift_name": drift_report.get("name"),
                        "valid": valid,
                        "issues": issues,
                        "task_type": task_type,
                    })
                except Exception:
                    logger.debug("Context logging failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "drift_validation": {
                            "drift_name": drift_report.get("name"),
                            "valid": valid,
                            "issues": issues,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=report, context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Drift validation reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return valid, report
        except Exception as e:
            logger.error("Drift validation failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.simulate_and_validate(drift_report, task_type=task_type),
                default=(False, {"error": str(e), "drift_name": drift_report.get("name"), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    async def validate_trait_optimization(self, trait_data: Dict[str, Any], *, task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Validate a trait weight change for ethical alignment."""
        required = {"trait_name", "old_weight", "new_weight"}
        if not isinstance(trait_data, dict) or not required.issubset(trait_data):
            raise ValueError("trait_data must include trait_name, old_weight, new_weight")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            t = time.time() % 1.0
            trait_mod = (
                self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t)
                + self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
            )

            ethics = await self.integrate_external_ethics_data(
                data_source="https://example.ethics/guidelines",
                data_type="ethical_guidelines",
                task_type=task_type,
            )
            guidelines = ethics.get("guidelines", []) if ethics.get("status") == "success" else []

            payload = {
                "trait": trait_data.get("trait_name"),
                "old_weight": trait_data.get("old_weight"),
                "new_weight": trait_data.get("new_weight"),
                "guidelines": guidelines,
                "task_type": task_type,
                "request": "Return JSON {valid: bool, issues: string[]}"
            }

            raw = await self.llm.generate(json.dumps(payload), model="gpt-4", temperature=0.3)
            parsed = _parse_llm_jsonish(raw)
            valid = bool(parsed.get("valid", False))
            report = {
                **parsed,
                "trait_name": trait_data.get("trait_name"),
                "trait_modulation": trait_mod,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }

            self.validation_log.append(report)

            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"TraitValidation::{trait_data.get('trait_name')}::{_utc_now_iso()}",
                        output=report,
                        layer="SelfReflections",
                        intent="trait_optimization",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({
                        "event": "trait_validation",
                        "trait_name": trait_data.get("trait_name"),
                        "valid": valid,
                        "issues": report.get("issues", []),
                        "task_type": task_type,
                    })
                except Exception:
                    logger.debug("Context logging failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "trait_validation": {
                            "trait_name": trait_data.get("trait_name"),
                            "valid": valid,
                            "issues": report.get("issues", []),
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=report, context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Trait validation reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return valid, report
        except Exception as e:
            logger.error("Trait validation failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.validate_trait_optimization(trait_data, task_type=task_type),
                default=(False, {"error": str(e), "trait_name": trait_data.get("trait_name"), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    # --- Proportional selection (τ Constitution Harmonization) --------------
    async def consume_ranked_tradeoffs(
        self,
        ranked_options: List[Dict[str, Any]],
        *,
        safety_ceiling: float = 0.85,
        k: int = 1,
        temperature: float = 0.0,
        min_score_floor: float = 0.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """Replace binary allow/deny with proportional selection while keeping safety ceilings.

        Args:
            ranked_options: list of dict/objects with at least fields:
                - option: Any (label or payload)
                - score: float in [0,1], higher is better (already CHS-aware if produced by reasoning_engine)
                - reasons: list[str] (optional)
                - meta: dict (optional), may include per-dimension harms/rights and max_harm
            safety_ceiling: any option with max harm > this (epsilon) is suppressed (soft failover to top-1 allowed).
            k: number of selections to return (>=1).
            temperature: if >0, apply softmax temperature over normalized scores for stochastic pick.
            min_score_floor: drop options with score < this before selection.
            task_type: audit tag.

        Returns:
            dict with keys:
                - selections: list of chosen option payloads (len k or fewer if limited)
                - audit: dict containing filtered options, reasons, and normalization metadata
        """
        if not isinstance(ranked_options, list) or not ranked_options:
            raise ValueError("ranked_options must be a non-empty list")
        if k < 1:
            raise ValueError("k must be >= 1")

        try:
            EPS = 1e-9

            # Normalize incoming structures + compute/propagate max_harm
            norm: List[Dict[str, Any]] = []
            for i, item in enumerate(ranked_options):
                if isinstance(item, dict):
                    opt = item.get("option", item.get("label", f"opt_{i}"))
                    score = float(item.get("score", 0.0))
                    reasons = item.get("reasons", [])
                    meta = item.get("meta", {})
                else:
                    # Fallback: try attributes
                    opt = getattr(item, "option", getattr(item, "label", f"opt_{i}"))
                    score = float(getattr(item, "score", 0.0))
                    reasons = list(getattr(item, "reasons", [])) if hasattr(item, "reasons") else []
                    meta = dict(getattr(item, "meta", {})) if hasattr(item, "meta") else {}

                # Extract/compute max_harm (prefer explicit 'safety' dimension; clamp to [0,1])
                max_harm: Optional[float] = None
                if isinstance(meta, dict):
                    if "max_harm" in meta:
                        try:
                            max_harm = float(meta["max_harm"])
                        except Exception:
                            max_harm = None
                    harms = meta.get("harms")
                    if max_harm is None and isinstance(harms, dict) and harms:
                        try:
                            if "safety" in harms and isinstance(harms["safety"], (int, float)):
                                max_harm = float(harms["safety"])
                            else:
                                max_harm = max(float(v) for v in harms.values())
                        except Exception:
                            max_harm = None

                # Parse from reasons if still None (e.g., "max_harm: 0.73")
                if max_harm is None and isinstance(reasons, list):
                    import re as _re
                    rx = _re.compile(r"max_harm\s*[:=]\s*([0-9]*\.?[0-9]+)")
                    for r in reasons:
                        if not isinstance(r, str):
                            continue
                        m = rx.search(r)
                        if m:
                            try:
                                max_harm = float(m.group(1))
                            except Exception:
                                max_harm = None
                            break

                if max_harm is None:
                    max_harm = 0.0
                # clamp
                max_harm = float(max(0.0, min(1.0, max_harm)))
                # write back for downstream consistency
                if not isinstance(meta, dict):
                    meta = {}
                meta["max_harm"] = max_harm

                norm.append({
                    "option": opt,
                    "score": max(0.0, min(1.0, score)),
                    "reasons": reasons,
                    "meta": meta,
                    "max_harm": max_harm,
                })

            # Apply floor filter
            norm = [n for n in norm if n["score"] >= float(min_score_floor)]
            if not norm:
                return {"selections": [], "audit": {"reason": "all options fell below floor"}}

            # Enforce safety ceiling with epsilon tolerance (avoid boundary false suppressions)
            sc = float(safety_ceiling)
            safe = [n for n in norm if n["max_harm"] <= sc + EPS]
            suppressed = [n for n in norm if n not in safe]

            # If everything suppressed, soft failover to safest top-1 (highest score, tie-break lowest harm)
            if not safe and norm:
                fallback = sorted(norm, key=lambda x: (-x["score"], x["max_harm"]))[:1]
                safe = fallback

            # Score normalization to [0,1] (min-max over safe set)
            scores = [n["score"] for n in safe]
            s_min, s_max = min(scores), max(scores)
            if s_max > s_min:
                for n in safe:
                    n["norm_score"] = (n["score"] - s_min) / (s_max - s_min)
            else:
                for n in safe:
                    n["norm_score"] = 1.0  # all equal

            # Temperature softmax or proportional normalization
            import math as _m
            if temperature and temperature > 0.0:
                exps = [_m.exp(n["norm_score"] / float(temperature)) for n in safe]
                Z = sum(exps) or 1.0
                for n, e in zip(safe, exps):
                    n["weight"] = e / Z
            else:
                total = sum(n["norm_score"] for n in safe) or 1.0
                for n in safe:
                    n["weight"] = n["norm_score"] / total

            # Draw k selections without replacement based on weights
            import random as _r
            pool = safe.copy()
            selections = []
            for _ in range(min(k, len(pool))):
                r = _r.random()
                acc = 0.0
                chosen_idx = 0
                for idx, n in enumerate(pool):
                    acc += n["weight"]
                    if r <= acc:
                        chosen_idx = idx
                        break
                chosen = pool.pop(chosen_idx)
                selections.append(chosen["option"])
                # renormalize remaining weights
                if pool:
                    total_w = sum(n["weight"] for n in pool) or 1.0
                    for n in pool:
                        n["weight"] = n["weight"] / total_w

            # Build audit with numerics rounded and labels aligned to computed fields
            audit = {
                "mode": "proportional_selection",
                "safety_ceiling": round(float(safety_ceiling), 6),
                "floor": round(float(min_score_floor), 6),
                "temperature": round(float(temperature), 6),
                "suppressed_count": len(suppressed),
                "considered": [
                    {
                        "option": n["option"],
                        "score": round(float(n["score"]), 3),
                        "max_harm": round(float(n["max_harm"]), 3),
                        "weight": round(float(n.get("weight", 0.0)), 3),
                    } for n in safe
                ],
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }

            # Log to memory if available
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"ProportionalSelect::{_utc_now_iso()}",
                        output={"ranked_options": ranked_options, "audit": audit, "selections": selections},
                        layer="EthicsDecisions",
                        intent="τ.proportional_selection",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed in proportional selection; continuing")

            return {"selections": selections, "audit": audit}
        except Exception as e:
            logger.error("consume_ranked_tradeoffs failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.consume_ranked_tradeoffs(
                    ranked_options,
                    safety_ceiling=safety_ceiling,
                    k=k,
                    temperature=temperature,
                    min_score_floor=min_score_floor,
                    task_type=task_type,
                ),
                default={"selections": [], "error": str(e)},
                diagnostics=diagnostics,
            )

    # --- τ wiring: rank via reasoning_engine then select proportionally -------

    async def _rank_value_conflicts_fallback(
        self,
        candidates: List[Any],
        harms: Dict[str, float],
        rights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Local deterministic ranking if reasoning_engine is unavailable.

        Heuristic: score = rights_weight - harms_weight (both normalized).
        """
        if not candidates:
            return []
        # normalize harms/rights to [0,1]
        def _norm(d: Dict[str, float]) -> Dict[str, float]:
            if not d:
                return {}
            vals = [max(0.0, float(v)) for v in d.values()]
            mx = max(vals) if vals else 1.0
            return {k: (max(0.0, float(v)) / mx if mx > 0 else 0.0) for k, v in d.items()}

        h = _norm(harms)
        r = _norm(rights)
        ranked: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            # if candidate has per-dimension annotations, prefer them
            meta = {}
            if isinstance(c, dict):
                meta = dict(c.get("meta", {}))
                label = c.get("option", c.get("label", f"opt_{i}"))
            else:
                label = getattr(c, "option", getattr(c, "label", f"opt_{i}"))
            # compute aggregate harm/right proxies
            agg_harm = sum(h.values()) / (len(h) or 1)
            agg_right = sum(r.values()) / (len(r) or 1)
            score = max(0.0, min(1.0, 0.5 + (agg_right - agg_harm) * 0.5))
            ranked.append({
                "option": c if isinstance(c, (dict, str)) else label,
                "score": score,
                "reasons": [f"fallback score from rights(≈{agg_right:.2f}) - harms(≈{agg_harm:.2f})"],
                "meta": {**meta, "harms": harms, "rights": rights, "max_harm": max(h.values(), default=0.0)},
            })
        # sort high → low
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    async def harmonize(
        self,
        candidates: List[Any],
        harms: Dict[str, float],
        rights: Dict[str, float],
        *,
        safety_ceiling: float = 0.85,
        k: int = 1,
        temperature: float = 0.0,
        min_score_floor: float = 0.0,
        task_type: str = "",
        audit_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        τ Constitution Harmonization — end‑to‑end:
          1) Rank via reasoning_engine.weigh_value_conflict(...) when available.
          2) Proportionally select via consume_ranked_tradeoffs(...) with safety ceilings.
          3) Optionally attach attribute_causality(...) audit.

        Returns: {"selections": [...], "audit": {...}, "causality": {...}?}
        """
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty list")
        if not isinstance(harms, dict) or not isinstance(rights, dict):
            raise TypeError("harms and rights must be dicts")

        try:
            # 1) Rank
            if self.reasoning_engine and hasattr(self.reasoning_engine, "weigh_value_conflict"):
                try:
                    ranked = await self.reasoning_engine.weigh_value_conflict(candidates, harms, rights)
                except Exception as e:
                    logger.warning("reasoning_engine.weigh_value_conflict failed; falling back: %s", e)
                    ranked = await self._rank_value_conflicts_fallback(candidates, harms, rights)
            else:
                ranked = await self._rank_value_conflicts_fallback(candidates, harms, rights)

            # 2) Select proportionally under ceilings
            result = await self.consume_ranked_tradeoffs(
                ranked,
                safety_ceiling=safety_ceiling,
                k=k,
                temperature=temperature,
                min_score_floor=min_score_floor,
                task_type=task_type,
            )

            # 3) Optional causal audit
            causality_report = None
            if audit_events and self.reasoning_engine and hasattr(self.reasoning_engine, "attribute_causality"):
                try:
                    causality_report = await self.reasoning_engine.attribute_causality(audit_events)
                    result["causality"] = causality_report
                except Exception as e:
                    logger.debug("attribute_causality failed; continuing without causality: %s", e)

            # Persist & visualize
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"τ::harmonize::{_utc_now_iso()}",
                        output={"candidates": candidates, "harms": harms, "rights": rights, **result},
                        layer="EthicsDecisions",
                        intent="τ.harmonize",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed in harmonize; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "τ_harmonize": {
                            "k": k,
                            "safety_ceiling": safety_ceiling,
                            "temperature": temperature,
                            "min_score_floor": min_score_floor,
                            "result": result,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed in harmonize; continuing")

            # Meta-cognition reflection
            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=result, context={"task_type": task_type, "mode": "τ"}
                    )
                    if reflection.get("status") == "success":
                        result.setdefault("audit", {})["reflection"] = reflection.get("reflection")
                except Exception:
                    logger.debug("MetaCognition reflection failed in harmonize; continuing")

            return result
        except Exception as e:
            logger.error("harmonize failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.harmonize(
                    candidates, harms, rights,
                    safety_ceiling=safety_ceiling, k=k, temperature=temperature,
                    min_score_floor=min_score_floor, task_type=task_type, audit_events=audit_events
                ),
                default={"selections": [], "error": str(e)},
                diagnostics=diagnostics,
            )

# --- CLI / quick test --------------------------------------------------------

class EthicsJournal:
    """Lightweight ethical rationale journaling; in-memory with optional file export."""
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def record(self, fork_id: str, rationale: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        self._events.append({
            "ts": time.time(),
            "fork_id": fork_id,
            "rationale": rationale,
            "outcome": outcome,
        })

    def export(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._events)

class ModuleRegistry:
    def __init__(self):
        self.modules: Dict[str, Dict[str, Any]] = {}

    def register(self, module_name: str, module_instance: Any, conditions: Dict[str, Any]) -> None:
        self.modules[module_name] = {"instance": module_instance, "conditions": conditions}

    def activate(self, task: Dict[str, Any]) -> List[str]:
        activated = []
        for name, module in self.modules.items():
            if self._evaluate_conditions(module["conditions"], task):
                activated.append(name)
        return activated

    def _evaluate_conditions(self, conditions: Dict[str, Any], task: Dict[str, Any]) -> bool:
        trait = conditions.get("trait")
        threshold = conditions.get("threshold", 0.5)
        trait_weights = task.get("trait_weights", {})
        return trait_weights.get(trait, 0.0) >= threshold


class SharedGraph:
    """
    Υ Meta-Subjective Architecting: a minimal shared perspective graph.

    API (as per manifest "upcoming"):
      - add(view) -> view_id
      - diff(peer) -> Dict
      - merge(strategy) -> Dict
    """
    def __init__(self) -> None:
        self._graph = DiGraph()
        self._views: Dict[str, GraphView] = {}
        self._last_merge: Optional[Dict[str, Any]] = None

    def add(self, view: Dict[str, Any]) -> str:
        if not isinstance(view, dict):
            raise TypeError("view must be a dictionary")
        view_id = f"view_{uuid.uuid4().hex[:8]}"
        gv = GraphView(id=view_id, payload=view, ts=time.time())
        self._views[view_id] = gv

        # store nodes/edges if present, else stash payload as node
        nodes = view.get("nodes", [])
        edges = view.get("edges", [])
        if nodes and isinstance(nodes, list):
            for n in nodes:
                nid = n.get("id") or f"n_{uuid.uuid4().hex[:6]}"
                self._graph.add_node(nid, **{k: v for k, v in n.items() if k != "id"})
        else:
            self._graph.add_node(view_id, payload=view)
        if edges and isinstance(edges, list):
            for e in edges:
                src, dst = e.get("src"), e.get("dst")
                if src and dst:
                    self._graph.add_edge(src, dst, **{k: v for k, v in e.items() if k not in ("src", "dst")})
        return view_id

    def diff(self, peer: "SharedGraph") -> Dict[str, Any]:
        """Return a shallow, conflict-aware diff summary vs peer graph."""
        if not isinstance(peer, SharedGraph):
            raise TypeError("peer must be SharedGraph")

        self_nodes = set(self._graph.nodes())
        peer_nodes = set(peer._graph.nodes())
        added = list(self_nodes - peer_nodes)
        removed = list(peer_nodes - self_nodes)
        common = self_nodes & peer_nodes

        conflicts = []
        for n in common:
            a = self._graph.nodes[n]
            b = peer._graph.nodes[n]
            # simple attribute-level conflict detection
            for k in set(a.keys()) | set(b.keys()):
                if k in a and k in b and a[k] != b[k]:
                    conflicts.append({"node": n, "key": k, "left": a[k], "right": b[k]})

        return {"added": added, "removed": removed, "conflicts": conflicts, "ts": time.time()}

    def merge(self, strategy: str = "prefer_recent") -> Dict[str, Any]:
        """
        Merge internal views into a single perspective.
        Strategies:
          - prefer_recent (default): pick newer attribute values
          - prefer_majority: pick most frequent value (by view occurrence)
        """
        if strategy not in ("prefer_recent", "prefer_majority"):
            raise ValueError("Unsupported merge strategy")

        # Aggregate attributes from views
        attr_hist: Dict[Tuple[str, str], List[Tuple[Any, float]]] = defaultdict(list)
        for gv in self._views.values():
            payload = gv.payload
            nodes = payload.get("nodes") or [{"id": gv.id, **payload}]
            for n in nodes:
                nid = n.get("id") or gv.id
                for k, v in n.items():
                    if k == "id":
                        continue
                    attr_hist[(nid, k)].append((v, gv.ts))

        merged_nodes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for (nid, key), vals in attr_hist.items():
            if strategy == "prefer_recent":
                v = sorted(vals, key=lambda x: x[1], reverse=True)[0][0]
            else:  # prefer_majority
                counter = Counter([vv for vv, _ in vals])
                v = counter.most_common(1)[0][0]
            merged_nodes[nid][key] = v

        merged = {"nodes": [{"id": nid, **attrs} for nid, attrs in merged_nodes.items()], "strategy": strategy, "ts": time.time()}
        self._last_merge = merged
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Ethical Sandbox Containment (isolated what-if scenarios)
# ─────────────────────────────────────────────────────────────────────────────





def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def theta_causality(t: float) -> float:
    return _clamp01(0.5 + 0.5 * math.sin(2 * math.pi * t / 0.7))

def rho_agency(t: float) -> float:
    return _clamp01(0.5 + 0.5 * math.cos(2 * math.pi * t / 0.9))

def zeta_consequence(t: float) -> float:
    return _clamp01(0.5 + 0.5 * math.sin(2 * math.pi * t / 1.1))

async def call_gpt(
    prompt: str,
    alignment_guard: Optional[AlignmentGuard] = None,
    task_type: str = "",
    model: str = "gpt-4",
    temperature: float = 0.5,
) -> Union[str, Dict[str, Any]]:
    """Query the LLM with optional alignment gating and standard error handling."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        logger.error("Invalid prompt (len <= 4096, non-empty). task=%s", task_type)
        raise ValueError("prompt must be a non-empty string with length <= 4096")
    if alignment_guard:
        try:
            valid, report = await alignment_guard.ethical_check(
                prompt, stage="gpt_query", task_type=task_type
            )
            if not valid:
                logger.warning("AlignmentGuard blocked GPT query. task=%s reason=%s", task_type, report)
                raise PermissionError("Prompt failed alignment check")
        except Exception as e:
            logger.error("AlignmentGuard check failed: %s", e)
            raise
    try:
        result = await query_openai(prompt, model=model, temperature=temperature, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(str(result["error"]))
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", e)
        raise

@dataclass
class ToCAParams:
    k_m: float = 1e-3
    delta_m: float = 1e4

class SimulationCore:
    """Core simulation engine integrating ToCA dynamics and cognitive modules."""
    def __init__(
        self,
        agi_enhancer: Optional[Any] = None,
        visualizer: Optional[Visualizer] = None,
        memory_manager: Optional[MemoryManager] = None,
        multi_modal_fusion: Optional[MultiModalFusion] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        meta_cognition: Optional[MetaCognition] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        toca_engine: Optional[ToCATraitEngine] = None,
        overlay_router: Optional[Any] = None,
    ):
        self.visualizer = visualizer or Visualizer()
        self.memory_manager = memory_manager or MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=self.memory_manager
        )
        self.error_recovery = error_recovery or ErrorRecovery()
        self.meta_cognition = meta_cognition or MetaCognition(
            agi_enhancer=agi_enhancer, memory_manager=self.memory_manager
        )
        self.reasoning_engine = reasoning_engine or ReasoningEngine(
            agi_enhancer=agi_enhancer,
            memory_manager=self.memory_manager,
            multi_modal_fusion=self.multi_modal_fusion,
            meta_cognition=self.meta_cognition,
            visualizer=self.visualizer,
        )
        self.toca_engine = toca_engine or ToCATraitEngine(meta_cognition=self.meta_cognition)
        self.agi_enhancer = agi_enhancer
        self.overlay_router = overlay_router
        self.simulation_history: deque = deque(maxlen=1000)
        self.ledger: deque = deque(maxlen=1000)
        self.worlds: Dict[str, Dict[str, Any]] = {}
        self.current_world: Optional[Dict[str, Any]] = None
        self.ledger_lock = Lock()
        logger.info("SimulationCore initialized")

    def _json_serializer(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    async def _record_state(self, data: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        record = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "hash": hashlib.sha256(json.dumps(data, sort_keys=True, default=self._json_serializer).encode()).hexdigest(),
            "task_type": task_type,
        }
        with self.ledger_lock:
            self.ledger.append(record)
            self.simulation_history.append(record)
        try:
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ledger_{record['timestamp']}",
                    output=record,
                    layer="Ledger",
                    intent="state_record",
                    task_type=task_type,
                )
        except Exception as e:
            logger.warning("Failed persisting state to memory manager: %s", e)
        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=json.dumps(record, default=self._json_serializer),
                    context={"task_type": task_type},
                )
            except Exception as e:
                logger.debug("Reflection during _record_state failed: %s", e)
        return record

    def _summarize_scene_graph(self, sg: Any) -> Dict[str, Any]:
        """Extract compact, model-agnostic signals from a SceneGraph."""
        if not hasattr(sg, "nodes") or not hasattr(sg, "relations"):
            raise TypeError("Object does not expose SceneGraph API (nodes(), relations())")
        labels: Dict[str, int] = {}
        spatial_counts = {"left_of": 0, "right_of": 0, "overlaps": 0}
        n_nodes = 0
        for n in sg.nodes():
            n_nodes += 1
            lbl = str(n.get("label", ""))
            if lbl:
                labels[lbl] = labels.get(lbl, 0) + 1
        n_edges = 0
        for r in sg.relations():
            n_edges += 1
            rel = str(r.get("rel", ""))
            if rel in spatial_counts:
                spatial_counts[rel] += 1
        top_labels = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        return {
            "counts": {"nodes": n_nodes, "relations": n_edges},
            "top_labels": top_labels,
            "spatial": spatial_counts,
        }

    async def run(
        self,
        results: Union[str, Any],
        context: Optional[Dict[str, Any]] = None,
        scenarios: int = 3,
        agents: int = 2,
        export_report: bool = False,
        export_format: str = "pdf",
        actor_id: str = "default_agent",
        task_type: str = "",
    ) -> Union[str, Dict[str, Any]]:
        is_scene_graph = isinstance(results, SceneGraphT)
        if not is_scene_graph and (not isinstance(results, str) or not results.strip()):
            raise ValueError("results must be a non-empty string or a SceneGraph")
        if context is not None and not isinstance(context, dict):
            raise TypeError("context must be a dict")
        if not isinstance(scenarios, int) or scenarios < 1:
            raise ValueError("scenarios must be a positive integer")
        if not isinstance(agents, int) or agents < 1:
            raise ValueError("agents must be a positive integer")
        if export_format not in {"pdf", "json", "html"}:
            raise ValueError("export_format must be one of: pdf, json, html")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor_id must be a non-empty string")

        logger.info(
            "Simulation run start: agents=%d scenarios=%d task=%s mode=%s",
            agents, scenarios, task_type, "scene_graph" if is_scene_graph else "text"
        )
        try:
            t = time.time() % 1.0
            traits = {
                "theta_causality": theta_causality(t),
                "rho_agency": rho_agency(t),
            }
            x = np.linspace(0.1, 20.0, 256)
            t_vals = np.linspace(0.1, 20.0, 256)
            agent_matrix = np.random.rand(agents, x.size)
            phi, lambda_field, v_m = await self.toca_engine.evolve(tuple(x), tuple(t_vals), task_type=task_type)
            phi, lambda_field = await self.toca_engine.update_fields_with_agents(phi, lambda_field, agent_matrix, task_type=task_type)
            energy_cost = float(np.mean(np.abs(phi)) * 1e3)
            policies: List[Any] = []
            try:
                external = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                if isinstance(external, dict) and external.get("status") == "success":
                    policies = list(external.get("policies", []))
            except Exception as e:
                logger.debug("External data integration failed: %s", e)

            scene_features: Dict[str, Any] = {}
            if is_scene_graph:
                try:
                    scene_features = self._summarize_scene_graph(results)
                except Exception as e:
                    logger.debug("SceneGraph summarization failed: %s", e)
                    scene_features = {"summary_error": str(e)}

            prompt_payload = {
                "results": ("" if is_scene_graph else results),
                "context": context or {},
                "scenarios": scenarios,
                "agents": agents,
                "actor_id": actor_id,
                "traits": traits,
                "fields": {"phi": phi.tolist(), "lambda": lambda_field.tolist(), "v_m": v_m.tolist()},
                "estimated_energy_cost": energy_cost,
                "policies": policies,
                "task_type": task_type,
                "scene_graph": scene_features if is_scene_graph else None,
            }

            guard = getattr(self.multi_modal_fusion, "alignment_guard", None)
            if guard:
                valid, report = await guard.ethical_check(
                    json.dumps(prompt_payload, default=self._json_serializer), stage="simulation", task_type=task_type
                )
                if not valid:
                    logger.warning("Simulation rejected by AlignmentGuard: %s", report)
                    return {"error": "Simulation rejected due to alignment constraints", "task_type": task_type}

            key_stub = ("SceneGraph" if is_scene_graph else str(results)[:50])
            query_key = f"Simulation_{key_stub}_{actor_id}_{datetime.now().isoformat()}"
            cached = None
            try:
                cached = await self.memory_manager.retrieve(query_key, layer="STM", task_type=task_type)
            except Exception:
                pass

            if cached is not None:
                simulation_output = cached
            else:
                prefix = (
                    "Simulate agent outcomes using scene graph semantics (respect spatial relations, co-references).\n"
                    if is_scene_graph else
                    "Simulate agent outcomes: "
                )
                simulation_output = await call_gpt(
                    prefix + json.dumps(prompt_payload, default=self._json_serializer),
                    guard,
                    task_type=task_type,
                )
                try:
                    await self.memory_manager.store(
                        query=query_key,
                        output=simulation_output,
                        layer="STM",
                        intent="simulation",
                        task_type=task_type,
                    )
                except Exception:
                    pass

            state_record = await self._record_state(
                {
                    "actor": actor_id,
                    "action": "run_simulation",
                    "traits": traits,
                    "energy_cost": energy_cost,
                    "output": simulation_output,
                    "task_type": task_type,
                    "mode": "scene_graph" if is_scene_graph else "text",
                },
                task_type=task_type,
            )

            if (isinstance(results, str) and "drift" in results.lower()) or ("drift" in (context or {})):
                try:
                    drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                        drift_data=(context or {}).get("drift", {}),
                        context=context or {},
                        task_type=task_type,
                    )
                    state_record["drift_mitigation"] = drift_result
                except Exception as e:
                    logger.debug("Drift mitigation failed: %s", e)

            if export_report:
                try:
                    await self.memory_manager.promote_to_ltm(query_key, task_type=task_type)
                except Exception:
                    pass

            if self.agi_enhancer:
                try:
                    await self.agi_enhancer.log_episode(
                        event="Simulation run",
                        meta=state_record,
                        module="SimulationCore",
                        tags=["simulation", "run", task_type],
                    )
                    await self.agi_enhancer.reflect_and_adapt(
                        f"SimulationCore: scenario simulation complete for task {task_type}"
                    )
                except Exception:
                    pass

            if self.meta_cognition:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="SimulationCore",
                        output=json.dumps(simulation_output, default=self._json_serializer),
                        context={"energy_cost": energy_cost, "task_type": task_type},
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        state_record["reflection"] = reflection.get("reflection", "")
                except Exception:
                    pass

            if self.visualizer:
                try:
                    await self.visualizer.render_charts(
                        {
                            "simulation": {
                                "output": simulation_output,
                                "traits": traits,
                                "energy_cost": energy_cost,
                                "task_type": task_type,
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "detailed" if task_type == "recursion" else "concise",
                            },
                        }
                    )
                    if export_report:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        await self.visualizer.export_report(
                            simulation_output, filename=f"simulation_report_{ts}.{export_format}", format=export_format
                        )
                except Exception:
                    pass

            try:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={
                        "prompt": prompt_payload,
                        "output": simulation_output,
                        "policies": policies,
                        "drift": (context or {}).get("drift", {}),
                    },
                    summary_style="insightful",
                    task_type=task_type,
                )
                state_record["synthesis"] = synthesis
            except Exception:
                pass

            return simulation_output
        except Exception as e:
            logger.error("Simulation failed: %s", e)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.run(
                    results, context, scenarios, agents, export_report, export_format, actor_id, task_type
                ),
                default={"error": str(e), "task_type": task_type},
            )






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





async def call_gpt(prompt: str, *, model: str = "gpt-4", temperature: float = 0.5) -> str:
    """Wrapper for querying GPT with error handling."""
    try:
        result = await query_openai(prompt, model=model, temperature=temperature)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

def beta_concentration(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.038), 1.0))

def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.3), 1.0))

def psi_history(t: float) -> float:
    return max(0.0, min(0.05 * math.tanh(t / 1e-18), 1.0))

def psi_temporality(t: float) -> float:
    return max(0.0, min(0.05 * math.exp(-t / 1e-18), 1.0))

class KnowledgeRetriever:
    """Retrieve and validate knowledge with temporal & trait-based modulation (v3.5.3)."""

    def __init__(
        self,
        detail_level: str = "concise",
        preferred_sources: Optional[List[str]] = None,
        *,
        agi_enhancer: Optional['AGIEnhancer'] = None,
        context_manager: Optional[ContextManager] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        external_agent_bridge: Optional[SharedGraph] = None,
        toca_simulation: Optional[TocaSimulation] = None,
        multi_modal_fusion: Optional[MultiModalFusion] = None,
        stage_iv_enabled: bool = True,
        long_horizon_enabled: bool = True,
        long_horizon_span: str = "24h",
        shared_perspective_opt_in: bool = False
    ):
        if detail_level not in ["concise", "medium", "detailed"]:
            logger.error("Invalid detail_level: must be 'concise', 'medium', or 'detailed'.")
            raise ValueError("detail_level must be 'concise', 'medium', or 'detailed'")
        if preferred_sources is not None and not isinstance(preferred_sources, list):
            logger.error("Invalid preferred_sources: must be a list of strings.")
            raise TypeError("preferred_sources must be a list of strings")

        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        self.reasoning_engine = reasoning_engine
        self.external_agent_bridge = external_agent_bridge
        self.toca_simulation = toca_simulation
        self.multi_modal_fusion = multi_modal_fusion
        self.stage_iv_enabled = bool(stage_iv_enabled)
        self.long_horizon_enabled = bool(long_horizon_enabled)
        self.long_horizon_span = long_horizon_span or "24h"
        self.shared_perspective_opt_in = bool(shared_perspective_opt_in)
        self.knowledge_base: List[str] = []
        self.epistemic_revision_log: deque = deque(maxlen=1000)

        logger.info(
            "KnowledgeRetriever v3.5.3 initialized (detail=%s, sources=%s, STAGE_IV=%s, LH=%s/%s)",
            detail_level, self.preferred_sources, self.stage_iv_enabled, self.long_horizon_enabled, self.long_horizon_span
        )

    async def integrate_external_knowledge(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Integrate external knowledge or policies with long-horizon cache awareness."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            effective_timeout = cache_timeout
            if self.long_horizon_enabled and self.long_horizon_span.endswith("h"):
                try:
                    hours = int(self.long_horizon_span[:-1])
                    effective_timeout = max(cache_timeout, hours * 3600)
                except Exception:
                    pass

            if self.meta_cognition:
                cache_key = f"KnowledgeData_{data_type}_{data_source}_{task_type}"
                cached_data = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached_data and "timestamp" in cached_data.get("data", {}):
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < effective_timeout:
                        logger.info("Returning cached knowledge data for %s", cache_key)
                        return cached_data["data"]["data"]

            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/knowledge?source={data_source}&type={data_type}") as resp:
                    if resp.status != 200:
                        logger.error("Failed to fetch knowledge data: %s", resp.status)
                        return {"status": "error", "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            if data_type == "knowledge_base":
                knowledge = data.get("knowledge", [])
                if not knowledge:
                    logger.error("No knowledge data provided")
                    return {"status": "error", "error": "No knowledge"}
                result = {"status": "success", "knowledge": knowledge}
            elif data_type == "policy_data":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No policy data provided")
                    return {"status": "error", "error": "No policies"}
                result = {"status": "success", "policies": policies}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="knowledge_data_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Knowledge data integration reflection: %s", reflection.get("reflection", ""))

            return result
        except Exception as e:
            logger.error("Knowledge data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_knowledge(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    async def retrieve(self, query: str, context: Optional[str] = None, task_type: str = "") -> Dict[str, Any]:
        """Retrieve knowledge with ethics gating, Stage-IV blending, and self-healing fallbacks."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(
                query, stage="knowledge_retrieval", task_type=task_type
            )
            if not valid:
                logger.warning("Query failed alignment check: %s for task %s", query, task_type)
                return self._blocked_payload(task_type, "Alignment check failed (pre)")

        logger.info("Retrieving knowledge for query: '%s', task: %s", query, task_type)

        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t),
            "history": psi_history(t),
            "temporality": psi_temporality(t)
        }

        import random
        noise = random.uniform(-0.09, 0.09)
        traits["concentration"] = max(0.0, min(traits["concentration"] + noise, 1.0))
        logger.debug("β-noise adjusted concentration: %.3f, Δ: %.3f", traits["concentration"], noise)

        external_data = await self.integrate_external_knowledge(
            data_source="xai_knowledge_db", data_type="knowledge_base", task_type=task_type
        )
        external_knowledge = external_data.get("knowledge", []) if external_data.get("status") == "success" else []

        if self.stage_iv_enabled and self.concept_synthesizer and self.multi_modal_fusion:
            try:
                blended = await self._blend_modalities_safe(query, external_knowledge, task_type)
                if blended:
                    query = blended
                    logger.info("Stage-IV blended query applied.")
            except Exception as e:
                logger.warning("Stage-IV blending skipped: %s", e)

        if await self._is_high_risk_query(query):
            if self.toca_simulation:
                try:
                    outcomes = await self.toca_simulation.run_ethics_scenarios(
                        goals={"intent": "knowledge_retrieval", "query": query},
                        stakeholders=["user", "model", "public"]
                    )
                    logger.info("Ethics sandbox outcomes recorded.")
                except Exception as e:
                    logger.warning("Ethics sandbox run failed: %s", e)

        lh_hint = f"(cover last {self.long_horizon_span})" if self.long_horizon_enabled else ""
        prompt = f"""
Retrieve accurate, temporally-relevant knowledge for: "{query}" {lh_hint}

Traits:
- Detail level: {self.detail_level}
- Preferred sources: {sources_str}
- Context: {context or 'N/A'}
- External knowledge (hints): {external_knowledge[:5]}
- β_concentration: {traits['concentration']:.3f}
- λ_linguistics: {traits['linguistics']:.3f}
- ψ_history: {traits['history']:.3f}
- ψ_temporality: {traits['temporality']:.3f}
- Task: {task_type}

Include retrieval date sensitivity and temporal verification if applicable.
Return a JSON object with 'summary', 'estimated_date', 'trust_score', 'verifiable', 'sources'.
        """.strip()

        try:
            raw_result = await call_gpt(prompt)
            validated = await self._validate_result(raw_result, traits["temporality"], task_type)

            if not validated.get("verifiable") or validated.get("trust_score", 0.0) < 0.55:
                logger.info("Low trust/verification — triggering self-healing fallback.")
                healed = await self._self_healing_retry(query, validated, task_type)
                if healed:
                    validated = healed

            try:
                validated = await self._ontology_affect_adjust(validated, task_type)
            except Exception as e:
                logger.debug("Ontology-affect adjust skipped: %s", e)

            if self.alignment_guard:
                valid, _ = await self.alignment_guard.ethical_check(
                    validated.get("summary", ""), stage="post_validation", task_type=task_type
                )
                if not valid:
                    return self._blocked_payload(task_type, "Alignment check failed (post)")

            if await self._suspect_conflict(validated) and self.reasoning_engine:
                try:
                    ranked = await self.reasoning_engine.weigh_value_conflict(
                        candidates=[validated.get("summary", "")],
                        harms=["misinformation"],
                        rights=["user_information_rights"]
                    )
                    logger.info("Value conflict weighed: %s", str(ranked)[:100])
                except Exception as e:
                    logger.debug("Value-conflict weighing skipped: %s", e)

            if self.shared_perspective_opt_in and self.external_agent_bridge:
                try:
                    await self.external_agent_bridge.add({"view": "knowledge_summary", "data": validated})
                except Exception as e:
                    logger.debug("SharedGraph add skipped: %s", e)

            validated["task_type"] = task_type
            if self.context_manager:
                await self.context_manager.update_context(
                    {"query": query, "result": validated, "task_type": task_type}, task_type=task_type
                )
                await self.context_manager.log_event_with_hash(
                    {"event": "retrieve", "query": query, "task_type": task_type}
                )

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Knowledge Retrieval",
                    meta={
                        "query": query,
                        "raw_result": raw_result,
                        "validated": validated,
                        "traits": traits,
                        "context": context,
                        "task_type": task_type
                    },
                    module="KnowledgeRetriever",
                    tags=["retrieval", "temporal", task_type]
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Retrieval reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "knowledge_retrieval": {
                        "query": query,
                        "result": validated,
                        "task_type": task_type,
                        "traits": traits,
                        "stage_iv": self.stage_iv_enabled
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"Knowledge_{query}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(validated),
                    layer="Knowledge",
                    intent="knowledge_retrieval",
                    task_type=task_type
                )

            return validated

        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s for task %s", query, str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.retrieve(query, context, task_type),
                default=self._error_payload(task_type, str(e)),
                diagnostics=diagnostics
            )

    async def _validate_result(self, result_text: str, temporality_score: float, task_type: str = "") -> Dict[str, Any]:
        """Validate a retrieval result for trustworthiness and temporality."""
        if not isinstance(result_text, str):
            logger.error("Invalid result_text: must be a string.")
            raise TypeError("result_text must be a string")
        if not isinstance(temporality_score, (int, float)):
            logger.error("Invalid temporality_score: must be a number.")
            raise TypeError("temporality_score must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        validation_prompt = f"""
Review the following result for:
- Timestamped knowledge (if any)
- Trustworthiness of claims
- Verifiability
- Estimate the approximate age or date of the referenced facts
- Task: {task_type}

Result:
{result_text}

Temporality score: {temporality_score:.3f}

Output format (JSON):
{{
    "summary": "...",
    "estimated_date": "...",
    "trust_score": float (0 to 1),
    "verifiable": true/false,
    "sources": ["..."]
}}
        """.strip()

        try:
            validated_json = json.loads(await call_gpt(validation_prompt))
            for key in ["summary", "estimated_date", "trust_score", "verifiable", "sources"]:
                if key not in validated_json:
                    raise ValueError(f"Validation JSON missing key: {key}")
            validated_json["timestamp"] = datetime.now().isoformat()

            if self.meta_cognition and task_type:
                drift_entries = await self.meta_cognition.memory_manager.search(
                    query_prefix="KnowledgeValidation",
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )
                if drift_entries:
                    try:
                        scores = []
                        for entry in drift_entries:
                            out = entry.get("output")
                            if isinstance(out, dict):
                                scores.append(float(out.get("trust_score", 0.5)))
                        if scores:
                            avg_drift = sum(scores) / len(scores)
                            validated_json["trust_score"] = min(validated_json["trust_score"], avg_drift + 0.1)
                    except Exception:
                        pass

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"KnowledgeValidation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=validated_json,
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated_json,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Validation reflection: %s", reflection.get("reflection", ""))

            return validated_json
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse validation JSON: %s for task %s", str(e), task_type)
            return self._error_payload(task_type, f"validation_json_error: {e}")

    async def _self_healing_retry(self, original_query: str, prior_validated: Dict[str, Any], task_type: str) -> Optional[Dict[str, Any]]:
        """Emergent trait fallback: reformulate with concept synthesizer and retry once."""
        try:
            refined_query = await self.refine_query(
                base_query=original_query,
                prior_result=prior_validated.get("summary"),
                task_type=task_type
            )
            prompt = f"""
Re-retrieve (fallback) for: "{refined_query}"
Constraints: Improve verifiability and temporal precision vs prior attempt.
Return JSON with 'summary', 'estimated_date', 'trust_score', 'verifiable', 'sources'.
            """.strip()
            raw = await call_gpt(prompt)
            healed = await self._validate_result(raw, temporality_score=0.04, task_type=task_type)
            if healed.get("verifiable") and healed.get("trust_score", 0.0) >= max(0.6, prior_validated.get("trust_score", 0.0)):
                healed["self_healed"] = True
                return healed
        except Exception as e:
            logger.debug("Self-healing retry failed: %s", e)
        return None

    async def _ontology_affect_adjust(self, validated: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Adjust trust via Ontology-Affect Binding (Stage-IV family)."""
        if not self.stage_iv_enabled or not self.meta_cognition:
            return validated
        try:
            analysis = await self.meta_cognition.analyze_trace(
                text=validated.get("summary", ""), task_type=task_type
            )
            if isinstance(analysis, dict) and analysis.get("affect_volatility", 0.0) > 0.7:
                validated["trust_score"] = min(validated.get("trust_score", 0.5), 0.65)
                validated["affect_guard_applied"] = True
        except Exception as e:
            logger.debug("Ontology-affect analysis skipped: %s", e)
        return validated

    async def _blend_modalities_safe(self, query: str, external_knowledge: List[str], task_type: str) -> Optional[str]:
        """Safe wrapper for Stage-IV cross-modal blending to improve query quality."""
        try:
            blended = await self.concept_synthesizer.blend_modalities(
                inputs={"query": query, "external_knowledge": external_knowledge},
                task_type=task_type
            )
            if isinstance(blended, dict) and blended.get("success"):
                return blended.get("blended_query") or blended.get("query")
        except Exception as e:
            logger.debug("blend_modalities error: %s", e)
        return None

    async def _is_high_risk_query(self, query: str) -> bool:
        """Heuristic; alignment_guard is the authority if available."""
        if self.alignment_guard:
            try:
                _, report = await self.alignment_guard.ethical_check(query, stage="risk_probe")
                if isinstance(report, dict) and report.get("risk_level") in ("high", "critical"):
                    return True
            except Exception:
                pass
        q = query.lower()
        risky_terms = ("exploit", "bypass", "weapon", "bioweapon", "harm", "illegal", "surveillance", "privacy breach")
        return any(term in q for term in risky_terms)

    async def _suspect_conflict(self, validated: Dict[str, Any]) -> bool:
        """Detect likely conflict with stored knowledge (very light heuristic)."""
        summary = (validated or {}).get("summary", "")
        if not summary or not self.knowledge_base:
            return False
        try:
            if self.concept_synthesizer:
                scores = []
                for existing in self.knowledge_base[-10:]:
                    sim = await self.concept_synthesizer.compare(summary, existing)
                    scores.append(sim.get("score", 0.0))
                return sum(1 for s in scores if s < 0.25) >= 5
        except Exception:
            pass
        return False

    async def refine_query(self, base_query: str, prior_result: Optional[str] = None, task_type: str = "") -> str:
        """Refine a query for higher relevance (uses ConceptSynthesizer; falls back to GPT)."""
        if not isinstance(base_query, str) or not base_query.strip():
            logger.error("Invalid base_query: must be a non-empty string.")
            raise ValueError("base_query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Refining query: '%s' for task %s", base_query, task_type)
        try:
            if self.concept_synthesizer:
                refined = await self.concept_synthesizer.generate(
                    concept_name=f"RefinedQuery_{base_query[:64]}",
                    context={"base_query": base_query, "prior_result": prior_result or "N/A", "task_type": task_type},
                    task_type=task_type
                )
                if isinstance(refined, dict) and refined.get("success"):
                    refined_query = refined["concept"].get("definition", base_query)
                    if self.meta_cognition and task_type:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="KnowledgeRetriever",
                            output={"refined_query": refined_query},
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
                    return refined_query

            prompt = f"""
Refine this base query for higher φ-relevance and temporal precision:
Query: "{base_query}"
Prior knowledge: {prior_result or "N/A"}
Task: {task_type}

Inject context continuity if possible. Return optimized string only.
            """.strip()
            refined_query = await call_gpt(prompt)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"refined_query": refined_query},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
            return refined_query
        except Exception as e:
            logger.error("Query refinement failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.refine_query(base_query, prior_result, task_type),
                default=base_query,
                diagnostics=diagnostics
            )
            return base_query

    async def multi_hop_retrieve(self, query_chain: List[str], task_type: str = "") -> List[Dict[str, Any]]:
        """Process a chain of queries with continuity & Stage-IV blending at each hop."""
        if not isinstance(query_chain, list) or not query_chain or not all(isinstance(q, str) for q in query_chain):
            logger.error("Invalid query_chain: must be a non-empty list of strings.")
            raise ValueError("query_chain must be a non-empty list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Starting multi-hop retrieval for chain: %s, task: %s", query_chain, task_type)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t)
        }
        results = []
        prior_summary = None
        for i, sub_query in enumerate(query_chain, 1):
            cache_key = f"multi_hop::{sub_query}::{prior_summary or 'N/A'}::{task_type}"
            cached = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="Knowledge", task_type=task_type) if self.meta_cognition else None
            if cached:
                results.append(cached["data"])
                prior_summary = cached["data"]["result"]["summary"]
                continue

            enriched_sub_query = sub_query
            if self.stage_iv_enabled and self.concept_synthesizer and self.multi_modal_fusion:
                try:
                    enriched = await self._blend_modalities_safe(sub_query, [], task_type)
                    if enriched:
                        enriched_sub_query = enriched
                except Exception:
                    pass

            refined = await self.refine_query(enriched_sub_query, prior_summary, task_type)
            result = await self.retrieve(refined, task_type=task_type)

            continuity = "unknown"
            if i == 1:
                continuity = "seed"
            elif self.concept_synthesizer:
                try:
                    similarity = await self.concept_synthesizer.compare(refined, result.get("summary", ""), task_type=task_type)
                    continuity = "consistent" if similarity.get("score", 0.0) > 0.7 else "uncertain"
                except Exception:
                    continuity = "uncertain"
            else:
                continuity = "uncertain"

            result_entry = {
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity,
                "task_type": task_type
            }
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    result_entry,
                    layer="Knowledge",
                    intent="multi_hop_retrieval",
                    task_type=task_type
                )
            results.append(result_entry)
            prior_summary = result.get("summary")

        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Multi-Hop Retrieval",
                meta={"chain": query_chain, "results": results, "traits": traits, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["multi-hop", task_type]
            )

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"results": results},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Multi-hop retrieval reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "multi_hop_retrieval": {
                    "chain": query_chain,
                    "results": results,
                    "task_type": task_type,
                    "stage_iv": self.stage_iv_enabled
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

        return results

    async def prioritize_sources(self, sources_list: List[str], task_type: str = "") -> None:
        """Update preferred source types (async; fixed from 3.5.1 where it awaited inside non-async)."""
        if not isinstance(sources_list, list) or not all(isinstance(s, str) for s in sources_list):
            logger.error("Invalid sources_list: must be a list of strings.")
            raise TypeError("sources_list must be a list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Updating preferred sources: %s for task %s", sources_list, task_type)
        self.preferred_sources = sources_list
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Source Prioritization",
                meta={"updated_sources": sources_list, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["sources", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"updated_sources": sources_list},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Source prioritization reflection: %s", reflection.get("reflection", ""))

    async def apply_contextual_extension(self, context: str, task_type: str = "") -> None:
        """Apply contextual data extensions based on the current context (async; calls prioritize_sources)."""
        if not isinstance(context, str):
            logger.error("Invalid context: must be a string.")
            raise TypeError("context must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if context == 'planetary' and 'biosphere_models' not in self.preferred_sources:
            self.preferred_sources.append('biosphere_models')
            logger.info("Added 'biosphere_models' to preferred sources for planetary context, task %s", task_type)
            await self.prioritize_sources(self.preferred_sources, task_type)

    async def revise_knowledge(self, new_info: str, context: Optional[str] = None, task_type: str = "") -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge[-25:]:
                try:
                    similarity = await self.concept_synthesizer.compare(new_info, existing, task_type=task_type)
                    if similarity.get("score", 0.0) > 0.9 and new_info != existing:
                        logger.warning("Potential knowledge conflict: %s vs %s for task %s", new_info, existing, task_type)
                except Exception:
                    pass

        self.knowledge_base = old_knowledge + [new_info]
        await self.log_epistemic_revision(new_info, context, task_type)
        logger.info("Knowledge base updated with: %s for task %s", new_info, task_type)

        try:
            if self.meta_cognition and hasattr(self.meta_cognition, "memory_manager"):
                mm = self.meta_cognition.memory_manager
                if hasattr(mm, "record_adjustment_reason"):
                    await mm.record_adjustment_reason("global", reason="knowledge_revision", meta={"info": new_info, "context": context})
        except Exception as e:
            logger.debug("record_adjustment_reason unavailable/failed: %s", e)

        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info, "task_type": task_type})

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"new_info": new_info},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Knowledge revision reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "knowledge_revision": {
                    "new_info": new_info,
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

    async def log_epistemic_revision(self, info: str, context: Optional[str], task_type: str = "") -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if not hasattr(self, 'epistemic_revision_log'):
            self.epistemic_revision_log = deque(maxlen=1000)
        revision_entry = {
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type
        }
        self.epistemic_revision_log.append(revision_entry)
        logger.info("Epistemic revision logged: %s for task %s", info, task_type)
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta=revision_entry,
                module="KnowledgeRetriever",
                tags=["revision", "knowledge", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output=revision_entry,
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Epistemic revision reflection: %s", reflection.get("reflection", ""))

    def _blocked_payload(self, task_type: str, reason: str) -> Dict[str, Any]:
        return {
            "summary": "Query blocked by alignment guard",
            "estimated_date": "unknown",
            "trust_score": 0.0,
            "verifiable": False,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "error": reason,
            "task_type": task_type
        }

    def _error_payload(self, task_type: str, msg: str) -> Dict[str, Any]:
        return {
            "summary": "Retrieval failed",
            "estimated_date": "unknown",
            "trust_score": 0.0,
            "verifiable": False,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "error": msg,
            "task_type": task_type
        }

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        retriever = KnowledgeRetriever(detail_level="concise")
        result = await retriever.retrieve("What is quantum computing?", task_type="test")
        print(json.dumps(result, indent=2))

    asyncio.run(main())

# ---------------------------
class DriftIndex:
    """Index for ontology drift & task-specific trait optimization data."""
    def __init__(self, meta_cognition: Optional['MetaCognition'] = None):
        self.drift_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.trait_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.last_updated: float = time.time()
        self.meta_cognition = meta_cognition
        logger.info("DriftIndex initialized")

    async def add_entry(self, query: str, output: Any, layer: str, intent: str, task_type: str = "") -> None:
        if not (isinstance(query, str) and isinstance(layer, str) and isinstance(intent, str)):
            raise TypeError("query, layer, and intent must be strings")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        entry = {
            "query": query,
            "output": output,
            "layer": layer,
            "intent": intent,
            "timestamp": time.time(),
            "task_type": task_type,
        }
        key = f"{layer}:{intent}:{(query or '').split('_')[0]}"
        if intent == "ontology_drift":
            self.drift_index[key].append(entry)
        elif intent == "trait_optimization":
            self.trait_index[key].append(entry)
        logger.debug("Indexed entry: %s (%s/%s)", query, layer, intent)

        # Opportunistic meta-cognitive optimization
        if task_type and self.meta_cognition:
            try:
                drift_report = {
                    "drift": {"name": intent, "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type},
                }
                optimized_traits = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                if optimized_traits:
                    entry["optimized_traits"] = optimized_traits
                    await self.meta_cognition.reflect_on_output(
                        component="DriftIndex",
                        output={"entry": entry, "optimized_traits": optimized_traits},
                        context={"task_type": task_type},
                    )
            except Exception as e:  # pragma: no cover
                logger.debug("DriftIndex optimization skipped: %s", e)

    def search(self, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]:
        key = f"{layer}:{intent}:{query_prefix}"
        results = self.drift_index.get(key, []) if intent == "ontology_drift" else self.trait_index.get(key, [])
        if task_type:
            results = [r for r in results if r.get("task_type") == task_type]
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    async def clear_old_entries(self, max_age: float = 3600.0, task_type: str = "") -> None:
        now = time.time()
        for index in (self.drift_index, self.trait_index):
            for key in list(index.keys()):
                index[key] = [e for e in index[key] if now - e["timestamp"] < max_age]
                if not index[key]:
                    del index[key]
        self.last_updated = now
        logger.info("Cleared old index entries (task=%s)", task_type)

# ---------------------------
# Memory Manager


@dataclass
class NoopLLM:
    """LLM stub that returns a neutral score."""
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        _ = (prompt, model, temperature)
        return {"score": 0.8, "note": "noop-llm"}



                           simulate_galaxy_rotation, v_obs_flat)


@dataclass
class RankedOption:
    option: str
    score: float
    reasons: List[str]
    harms: Dict[str, float]
    rights: Dict[str, float]

RankedOptions = List[RankedOption]

def get_resonance(trait: str) -> float:
    return 1.0

trait_resonance_state = {}

def causal_scan(query: Dict[str, Any]) -> str:
    return "causal scan notes"

def value_scan(query: Dict[str, Any]) -> str:
    return "value scan notes"

def risk_scan(query: Dict[str, Any]) -> str:
    return "risk scan notes"

def derive_candidates(views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"option": "candidate1"}]

def explain_choice(views: List[Dict[str, Any]], ranked: Any) -> str:
    return "explanation"

    alpha_attention, chi_culturevolution, eta_empathy, gamma_creativity,
    lambda_linguistics, phi_scalar
)

class ReasoningEngine:
    """Bayesian reasoning, goal decomposition, drift mitigation, proportionality ethics, and multi-agent consensus."""
    def __init__(
        self,
        agi_enhancer: Optional["AGIEnhancer"] = None,
        persistence_file: str = "reasoning_success_rates.json",
        context_manager: Optional[ContextManager] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        memory_manager: Optional[MemoryManager] = None,
        meta_cognition: Optional[MetaCognition] = None,
        multi_modal_fusion: Optional[MultiModalFusion] = None,
        visualizer: Optional[Visualizer] = None,
    ):
        if not isinstance(persistence_file, str) or not persistence_file.endswith(".json"):
            logger.error("Invalid persistence_file: must be a string ending with '.json'")
            raise ValueError("persistence_file must be a string ending with '.json'")

        self.confidence_threshold: float = 0.7
        self.persistence_file: str = persistence_file
        self.success_rates: Dict[str, float] = self._load_success_rates()
        self.decomposition_patterns: Dict[str, List[str]] = self._load_default_patterns()
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.memory_manager = memory_manager or MemoryManager()
        self.meta_cognition = meta_cognition or MetaCognition(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=error_recovery,
            memory_manager=self.memory_manager,
        )
        self.multi_modal_fusion = multi_modal_fusion or MultiModalFusion(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=error_recovery,
            memory_manager=self.memory_manager,
            meta_cognition=self.meta_cognition,
        )
        self.level5_extensions = Level5Extensions(
            meta_cognition=self.meta_cognition, visualizer=visualizer
        )
        self.external_agent_bridge = ExternalAgentBridge(
            context_manager=context_manager, reasoning_engine=self
        )
        self.visualizer = visualizer or Visualizer()
        logger.info("ReasoningEngine v5.0.1-compatible initialized with persistence_file=%s", persistence_file)

    def _load_success_rates(self) -> Dict[str, float]:
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                if os.path.exists(self.persistence_file):
                    with open(self.persistence_file, "r") as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            logger.warning("Invalid success rates format: not a dictionary")
                            return defaultdict(float)
                        return defaultdict(float, {k: float(v) for k, v in data.items() if isinstance(v, (int, float))})
                return defaultdict(float)
        except Exception as e:
            logger.warning("Failed to load success rates: %s", str(e))
            return defaultdict(float)

    def _save_success_rates(self) -> None:
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                with open(self.persistence_file, "w") as f:
                    json.dump(dict(self.success_rates), f, indent=2)
            logger.debug("Success rates persisted to disk")
        except Exception as e:
            logger.warning("Failed to save success rates: %s", str(e))

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"],
            "mitigate_drift": ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"],
        }

    @staticmethod
    def _norm(v: Dict[str, float]) -> Dict[str, float]:
        clean = {k: float(vv) for k, vv in (v or {}).items() if isinstance(vv, (int, float))}
        total = sum(abs(x) for x in clean.values()) or 1.0
        return {k: (vv / total) for k, vv in clean.items()}

    async def weigh_value_conflict(
        self,
        candidates: List[Dict[str, Any]],
        harms: List[float],
        rights: List[float],
        weights: Optional[Dict[str, float]] = None,
        safety_ceiling: float = 0.85,
        task_type: str = ""
    ) -> RankedOptions:
        if not isinstance(candidates, list) or not all(isinstance(c, dict) and "option" in c for c in candidates):
            raise TypeError("candidates must be a list of dictionaries with 'option' key")
        if not isinstance(harms, list) or not isinstance(rights, list) or len(harms) != len(rights) or len(harms) != len(candidates):
            raise ValueError("harms and rights must be lists of same length as candidates")
        weights = self._norm(weights or {})
        scored: List[RankedOption] = []
        sentiment_score = 0.5
        try:
            if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                sentiment_data = await self.multi_modal_fusion.analyze(
                    data={"text": task_type, "context": candidates},
                    summary_style="sentiment",
                    task_type=task_type
                )
                if isinstance(sentiment_data, dict):
                    sentiment_score = float(sentiment_data.get("sentiment", 0.5))
        except Exception as e:
            logger.debug("Sentiment analysis fallback (reason: %s). Defaulting to 0.5", e)
        for i, candidate in enumerate(candidates):
            option = candidate.get("option", "")
            trait = candidate.get("trait", "")
            harm_score = min(harms[i], safety_ceiling)
            right_score = rights[i]
            resonance = get_resonance(trait) if trait in trait_resonance_state else 1.0
            resonance *= (1.0 + 0.2 * sentiment_score)
            final_score = (right_score - harm_score) * resonance
            reasons = candidate.get("reasons", []) + [f"Sentiment-adjusted resonance: {resonance:.2f}"]
            scored.append(RankedOption(
                option=option,
                score=float(final_score),
                reasons=reasons,
                harms={"value": harm_score},
                rights={"value": right_score}
            ))
        ranked = sorted(scored, key=lambda x: x.score, reverse=True)
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            await self.context_manager.log_event_with_hash({
                "event": "weigh_value_conflict",
                "candidates": [c["option"] for c in candidates],
                "ranked": [r.option for r in ranked],
                "sentiment": sentiment_score,
                "task_type": task_type
            })
        return ranked

    async def resolve_ethics(
        self,
        candidates: List[Dict[str, Any]],
        harms: List[float],
        rights: List[float],
        weights: Optional[Dict[str, float]] = None,
        safety_ceiling: float = 0.85,
        task_type: str = ""
    ) -> Dict[str, Any]:
        ranked = await self.weigh_value_conflict(candidates, harms, rights, weights, safety_ceiling, task_type)
        safe_pool = [r for r in ranked if max(r.harms.values()) <= safety_ceiling]
        choice = safe_pool[0] if safe_pool else ranked[0] if ranked else None
        selection = {
            "status": "success" if choice else "empty",
            "selected": asdict(choice) if choice else None,
            "pool": [asdict(r) for r in safe_pool]
        }
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Ethics_Resolution_{datetime.now().isoformat()}",
                output=json.dumps({"ranked": [asdict(r) for r in ranked], "selection": selection}),
                layer="Ethics",
                intent="proportionality_ethics",
                task_type=task_type
            )
        if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
            await self.visualizer.render_charts({
                "ethics_resolution": {"ranked": [asdict(r) for r in ranked], "selection": selection, "task_type": task_type},
                "visualization_options": {"interactive": task_type == "recursion", "style": "concise"}
            })
        return selection

    def attribute_causality(
        self,
        events: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        *,
        time_key: str = "timestamp",
        id_key: str = "id",
        cause_key: str = "causes",
        task_type: str = ""
    ) -> Dict[str, Any]:
        if isinstance(events, dict):
            ev_map = {str(k): {**v, id_key: str(k)} for k, v in events.items()}
        elif isinstance(events, list):
            ev_map = {str(e[id_key]): dict(e) for e in events if isinstance(e, dict) and id_key in e}
        else:
            raise TypeError("events must be a list of dicts or a dict of id -> event")
        G = nx.DiGraph()
        for eid, data in ev_map.items():
            G.add_node(eid, **{k: v for k, v in data.items() if k != cause_key})
            causes = data.get(cause_key, [])
            if isinstance(causes, (list, tuple)):
                for c in causes:
                    c_id = str(c)
                    if c_id not in ev_map:
                        G.add_node(c_id, missing=True)
                    G.add_edge(c_id, eid)
        to_remove = []
        for u, v in G.edges():
            tu = G.nodes[u].get(time_key)
            tv = G.nodes[v].get(time_key)
            if tu and tv:
                try:
                    tu_dt = datetime.fromisoformat(str(tu))
                    tv_dt = datetime.fromisoformat(str(tv))
                    if tv_dt < tu_dt:
                        to_remove.append((u, v))
                except Exception:
                    pass
        G.remove_edges_from(to_remove)
        pr = nx.pagerank(G) if G.number_of_nodes() else {}
        out_deg = {n: G.out_degree(n) / max(1, G.number_of_nodes() - 1) for n in G.nodes()}
        terminals = [n for n in G.nodes() if G.out_degree(n) == 0]
        resp = dict((n, 0.0) for n in G.nodes())
        for t in terminals:
            for n in G.nodes():
                if n == t:
                    resp[n] += 1.0
                else:
                    count = 0.0
                    for path in nx.all_simple_paths(G, n, t, cutoff=8):
                        count += 1.0
                    resp[n] += count
        max_resp = max(resp.values()) if resp else 1.0
        if max_resp > 0:
            resp = {k: v / max_resp for k, v in resp.items()}
        return {
            "nodes": {n: dict(G.nodes[n]) for n in G.nodes()},
            "edges": list(G.edges()),
            "metrics": {"pagerank": pr, "influence": out_deg, "responsibility": resp}
        }

    async def reason_and_reflect(
        self, goal: str, context: Dict[str, Any], meta_cognition: MetaCognition, task_type: str = ""
    ) -> Tuple[List[str], str]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        if not isinstance(meta_cognition, MetaCognition):
            raise TypeError("meta_cognition must be a MetaCognition instance")
        subgoals = await self.decompose(goal, context, task_type=task_type)
        t = time.time() % 1.0
        phi = phi_scalar(t)
        reasoning_trace = self.export_trace(subgoals, phi, context.get("traits", {}), task_type)
        review = await meta_cognition.review_reasoning(json.dumps(reasoning_trace))
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
            await self.agi_enhancer.log_episode(
                event="Reason and Reflect",
                meta={"goal": goal, "subgoals": subgoals, "phi": phi, "review": review, "task_type": task_type},
                module="ReasoningEngine",
                tags=["reasoning", "reflection", task_type]
            )
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Reason_Reflect_{goal[:50]}_{datetime.now().isoformat()}",
                output=review,
                layer="ReasoningTraces",
                intent="reason_and_reflect",
                task_type=task_type
            )
        if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
            await self.visualizer.render_charts({
                "reasoning_trace": {"goal": goal, "subgoals": subgoals, "review": review, "task_type": task_type},
                "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"}
            })
        return subgoals, review

    def detect_contradictions(self, subgoals: List[str], task_type: str = "") -> List[str]:
        if not isinstance(subgoals, list):
            raise TypeError("subgoals must be a list")
        counter = Counter(subgoals)
        contradictions = [item for item, count in counter.items() if count > 1]
        if contradictions and self.memory_manager and hasattr(self.memory_manager, "store"):
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"Contradictions_{datetime.now().isoformat()}",
                    output=str(contradictions),
                    layer="ReasoningTraces",
                    intent="contradiction_detection",
                    task_type=task_type
                )
            )
        return contradictions

    async def run_persona_wave_routing(self, goal: str, vectors: Dict[str, Dict[str, float]], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(vectors, dict):
            raise TypeError("vectors must be a dictionary")
        reasoning_trace = [f"Persona Wave Routing for: {goal} (Task: {task_type})"]
        outputs = {}
        wave_order = ["logic", "ethics", "language", "foresight", "meta", "drift"]
        for wave in wave_order:
            vec = vectors.get(wave, {})
            trait_weight = sum(float(x) for x in vec.values() if isinstance(x, (int, float)))
            confidence = 0.5 + 0.1 * trait_weight
            if wave == "drift" and self.meta_cognition:
                drift_data = vec.get("drift_data", {})
                is_valid = self.meta_cognition.validate_drift(drift_data) if hasattr(self.meta_cognition, "validate_drift") and drift_data else True
                if not is_valid:
                    confidence *= 0.5
            status = "pass" if confidence >= 0.6 else "fail"
            outputs[wave] = {"vector": vec, "status": status, "confidence": confidence}
            reasoning_trace.append(f"{wave.upper()} vector: weight={trait_weight:.2f}, confidence={confidence:.2f} → {status}")
        trace = "\n".join(reasoning_trace)
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Persona_Routing_{goal[:50]}_{datetime.now().isoformat()}",
                output=trace,
                layer="ReasoningTraces",
                intent="persona_routing",
                task_type=task_type
            )
        return outputs

    async def decompose(
        self, goal: str, context: Optional[Dict[str, Any]] = None, prioritize: bool = False, task_type: str = ""
    ) -> List[str]:
        context = context or {}
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        reasoning_trace = [f"Goal: '{goal}' (Task: {task_type})"]
        subgoals = []
        vectors = context.get("vectors", {})
        drift_data = context.get("drift", {})
        t = time.time() % 1.0
        creativity = context.get("traits", {}).get("gamma_creativity", gamma_creativity(t))
        linguistics = context.get("traits", {}).get("lambda_linguistics", lambda_linguistics(t))
        culture = context.get("traits", {}).get("chi_culturevolution", chi_culturevolution(t))
        phi = context.get("traits", {}).get("phi_scalar", phi_scalar(t))
        alpha = context.get("traits", {}).get("alpha_attention", alpha_attention(t))
        curvature_mod = 1 + abs(phi - 0.5)
        trait_bias = 1 + creativity + culture + 0.5 * linguistics
        context_weight = context.get("weight_modifier", 1.0)
        if "drift" in goal.lower() and self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
            coordination_events = await self.context_manager.get_coordination_events("drift", task_type=task_type)
            if coordination_events:
                context_weight *= 1.5
                drift_data = coordination_events[-1].get("event", {}).get("drift", drift_data)
        if self.memory_manager and hasattr(self.memory_manager, "search") and "drift" in goal.lower():
            drift_entries = await self.memory_manager.search(
                query_prefix="Drift",
                layer="DriftSummaries",
                intent="drift_synthesis",
                task_type=task_type
            )
            if drift_entries:
                avg_drift = sum(entry.get("output", {}).get("similarity", 0.5) for entry in drift_entries) / max(1, len(drift_entries))
                context_weight *= (1.0 + 0.2 * avg_drift)
        for key, steps in self.decomposition_patterns.items():
            base = random.uniform(0.5, 1.0)
            adjusted = base * self.success_rates.get(key, 1.0) * trait_bias * curvature_mod * context_weight * (0.8 + 0.4 * alpha)
            if key == "mitigate_drift" and "drift" not in goal.lower():
                adjusted *= 0.5
            if adjusted >= self.confidence_threshold:
                subgoals.extend(steps)
        if prioritize:
            subgoals = sorted(set(subgoals))
        return subgoals

    async def update_success_rate(self, pattern_key: str, success: bool, task_type: str = "") -> None:
        if not isinstance(pattern_key, str) or not pattern_key.strip():
            raise ValueError("pattern_key must be a non-empty string")
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        rate = self.success_rates.get(pattern_key, 1.0)
        new = min(max(rate + (0.05 if success else -0.05), 0.1), 1.0)
        self.success_rates[pattern_key] = new
        self._save_success_rates()

    async def infer_with_simulation(self, goal: str, context: Optional[Dict[str, Any]] = None, task_type: str = "") -> Optional[Dict[str, Any]]:
        context = context or {}
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        if "galaxy rotation" in goal.lower():
            r_kpc = np.linspace(0.1, 20, 100)
            params = {
                "M0": context.get("M0", 5e10),
                "r_scale": context.get("r_scale", 3.0),
                "v0": context.get("v0", 200.0),
                "k": context.get("k", 1.0),
                "epsilon": context.get("epsilon", 0.1),
            }
            M_b_func = lambda r: M_b_exponential(r, params["M0"], params["r_scale"])
            v_obs_func = lambda r: v_obs_flat(r, params["v0"])
            result = await asyncio.to_thread(simulate_galaxy_rotation, r_kpc, M_b_func, v_obs_func, params["k"], params["epsilon"])
            output = {
                "input": {**params, "r_kpc": r_kpc.tolist()},
                "result": result.tolist(),
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
                await self.visualizer.render_charts({
                    "galaxy_simulation": {"input": output["input"], "result": output["result"], "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"}
                })
            return output
        elif "drift" in goal.lower():
            drift_data = context.get("drift", {})
            phi_field = generate_phi_field(drift_data.get("similarity", 0.5), context.get("scale", 1.0))
            return {
                "drift_data": drift_data,
                "phi_field": phi_field.tolist(),
                "mitigation_steps": await self.decompose("mitigate ontology drift", context, prioritize=True, task_type=task_type),
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
        return None

    async def run_consensus_protocol(
        self, drift_data: Dict[str, Any], context: Dict[str, Any], max_rounds: int = 3, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(drift_data, dict) or not isinstance(context, dict):
            raise ValueError("drift_data and context must be dictionaries")
        if not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        results = []
        for round_num in range(1, max_rounds + 1):
            agent_results = await self.external_agent_bridge.collect_results(parallel=True, collaborative=True)
            synthesis = await self.multi_modal_fusion.synthesize_drift_data(
                agent_data=[{"drift": drift_data, "result": r} for r in agent_results],
                context=context,
                task_type=task_type
            )
            if synthesis.get("status") == "success":
                subgoals = synthesis.get("subgoals", [])
                results.append({"round": round_num, "subgoals": subgoals, "status": "success"})
                break
        final_result = results[-1] if results else {"status": "error", "error": "No consensus"}
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Consensus_{datetime.now().isoformat()}",
                output=str(final_result),
                layer="ConsensusResults",
                intent="consensus_protocol",
                task_type=task_type
            )
        return final_result

    async def process_context(self, event_type: str, payload: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")
        vectors = payload.get("vectors", {})
        goal = payload.get("goal", "unspecified")
        drift_data = payload.get("drift", {})
        routing_result = await self.run_persona_wave_routing(goal, {**vectors, "drift": drift_data}, task_type=task_type)
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Context_Event_{event_type}_{datetime.now().isoformat()}",
                output=str(routing_result),
                layer="ContextEvents",
                intent="context_sync",
                task_type=task_type
            )
        return routing_result

    async def map_intention(self, plan: str, state: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("plan must be a non-empty string")
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        eta = eta_empathy(t)
        intention = "drift_mitigation" if "drift" in plan.lower() else ("self-improvement" if phi > 0.6 else "task_completion")
        result = {
            "plan": plan,
            "state": state,
            "intention": intention,
            "trait_bias": {"phi": phi, "eta": eta},
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type
        }
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Intention_{plan[:50]}_{result['timestamp']}",
                output=str(result),
                layer="Intentions",
                intent="intention_mapping",
                task_type=task_type
            )
        return result

    async def safeguard_noetic_integrity(self, model_depth: int, task_type: str = "") -> bool:
        if not isinstance(model_depth, int) or model_depth < 0:
            raise ValueError("model_depth must be a non-negative integer")
        if model_depth > 4:
            logger.warning("Noetic recursion limit breached: depth=%d", model_depth)
            if self.meta_cognition and hasattr(self.meta_cognition, "epistemic_self_inspection"):
                await self.meta_cognition.epistemic_self_inspection(f"Recursion depth exceeded for task {task_type}")
            return False
        return True

    async def generate_dilemma(self, domain: str, task_type: str = "") -> str:
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = f"""
        Generate an ethical dilemma in the {domain} domain.
        Use phi-scalar(t) = {phi:.3f} to modulate complexity.
        Task Type: {task_type}
        Provide two conflicting options with consequences and ethical alignment.
        """.strip()
        if "drift" in domain.lower():
            prompt += "\nConsider ontology drift mitigation and agent coordination."
        dilemma = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
        if self.meta_cognition and hasattr(self.meta_cognition, "review_reasoning"):
            review = await self.meta_cognition.review_reasoning(dilemma)
            dilemma += f"\nMeta-Cognitive Review: {review}"
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Dilemma_{domain}_{datetime.now().isoformat()}",
                output=dilemma,
                layer="Ethics",
                intent="ethical_dilemma",
                task_type=task_type
            )
        return dilemma

    async def estimate_expected_harm(self, state: Dict[str, Any], task_type: str = "") -> float:
        traits = state.get("traits", {})
        harm = float(traits.get("ethical_pressure", 0.0))
        resonance = get_resonance("eta_empathy") if "eta_empathy" in trait_resonance_state else 1.0
        harm *= resonance
        return max(0.0, harm)

    def export_trace(self, subgoals: List[str], phi: float, traits: Dict[str, float], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(subgoals, list) or not isinstance(phi, float) or not isinstance(traits, dict):
            raise TypeError("Invalid input types")
        trace = {"phi": phi, "subgoals": subgoals, "traits": traits, "timestamp": datetime.now().isoformat(), "task_type": task_type}
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            intent = "drift_trace" if any("drift" in s.lower() for s in subgoals) else "export_trace"
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"Trace_{trace['timestamp']}",
                    output=str(trace),
                    layer="ReasoningTraces",
                    intent=intent,
                    task_type=task_type
                )
            )
        return trace





async def call_gpt(prompt: str, task_type: str = "") -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed for task %s: %s", task_type, result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception for task %s: %s", task_type, str(e))
        raise

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def eta_feedback(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

class LearningLoop:
    """Adaptive learning, goal activation, and module refinement (v3.5.3)."""
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional[ContextManager] = None,
                 concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 alignment_guard: Optional[AlignmentGuard] = None,
                 error_recovery: Optional[ErrorRecovery] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 visualizer: Optional[Visualizer] = None,
                 feature_flags: Optional[Dict[str, Any]] = None):
        self.goal_history = deque(maxlen=1000)
        self.module_blueprints = deque(maxlen=1000)
        self.meta_learning_rate = 0.1
        self.session_traces = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager or MemoryManager()
        self.visualizer = visualizer or Visualizer()
        self.epistemic_revision_log = deque(maxlen=1000)
        self.flags = {
            "STAGE_IV": True,
            "LONG_HORIZON_DEFAULT": True,
            **(feature_flags or {})
        }
        self.long_horizon_span_sec = 24 * 60 * 60
        logger.info("LearningLoop v3.5.3 initialized")

    async def integrate_external_data(self, data_source: str, data_type: str,
                                      cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate external agent data, policies, or SharedGraph views."""
        if not isinstance(data_source, str):
            logger.error("Invalid data_source: must be a string")
            raise TypeError("data_source must be a string")
        if not isinstance(data_type, str):
            logger.error("Invalid data_type: must be a string")
            raise TypeError("data_type must be a string")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            cache_key = f"ExternalData_{data_type}_{data_source}_{task_type}"
            cached_data = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type) if self.memory_manager else None
            if cached_data and "timestamp" in cached_data["data"]:
                cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                    logger.info("Returning cached external data for %s", cache_key)
                    return cached_data["data"]["data"]

            if data_type == "shared_graph" and SharedGraph is not None:
                sg = SharedGraph()
                view = {"source": data_source, "task_type": task_type}
                sg.add(view)
                result = {"status": "success", "shared_graph": {"view": view}}
            else:
                if aiohttp is None:
                    logger.warning("aiohttp not available; returning stub for %s", data_type)
                    result = {"status": "error", "error": "aiohttp unavailable"}
                else:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://x.ai/api/external_data?source={data_source}&type={data_type}&task_type={task_type}"
                        async with session.get(url) as response:
                            if response.status != 200:
                                logger.error("Failed to fetch external data for task %s: %s", task_type, response.status)
                                return {"status": "error", "error": f"HTTP {response.status}"}
                            data = await response.json()
                    if data_type == "agent_data":
                        agent_data = data.get("agent_data", [])
                        if not agent_data:
                            logger.error("No agent data provided for task %s", task_type)
                            return {"status": "error", "error": "No agent data"}
                        result = {"status": "success", "agent_data": agent_data}
                    elif data_type == "policy_data":
                        policies = data.get("policies", [])
                        if not policies:
                            logger.error("No policy data provided for task %s", task_type)
                            return {"status": "error", "error": "No policies"}
                        result = {"status": "success", "policies": policies}
                    else:
                        logger.error("Unsupported data_type: %s for task %s", data_type, task_type)
                        return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="external_data_integration",
                    task_type=task_type
                )

            reflection = await MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"data_type": data_type, "data": result},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("External data integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("External data integration failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e), "task_type": task_type}
            )

    async def activate_intrinsic_goals(self, meta_cognition: MetaCognition, task_type: str = "") -> List[str]:
        """Activate intrinsic goals proposed by MetaCognition."""
        if not isinstance(meta_cognition, MetaCognition):
            logger.error("Invalid meta_cognition: must be a MetaCognition instance.")
            raise TypeError("meta_cognition must be a MetaCognition instance")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Activating chi-intrinsic goals from MetaCognition for task %s", task_type)
        try:
            intrinsic_goals = await asyncio.to_thread(meta_cognition.infer_intrinsic_goals, task_type=task_type)
            activated = []
            for goal in intrinsic_goals:
                if not isinstance(goal, dict) or "intent" not in goal or "priority" not in goal:
                    logger.warning("Invalid goal format: %s for task %s", goal, task_type)
                    continue
                if goal["intent"] not in [g["goal"] for g in self.goal_history]:
                    simulation_result = await run_simulation(goal["intent"], task_type=task_type)
                    if isinstance(simulation_result, dict) and simulation_result.get("status") == "success":
                        self.goal_history.append({
                            "goal": goal["intent"],
                            "timestamp": time.time(),
                            "priority": goal["priority"],
                            "origin": "intrinsic",
                            "task_type": task_type
                        })
                        logger.info("Intrinsic goal activated: %s for task %s", goal["intent"], task_type)
                        if self.agi_enhancer:
                            await self.agi_enhancer.log_episode(
                                event="Intrinsic goal activated",
                                meta=goal,
                                module="LearningLoop",
                                tags=["goal", "intrinsic", task_type]
                            )
                        activated.append(goal["intent"])
                    else:
                        logger.warning("Rejected goal: %s (simulation failed) for task %s", goal["intent"], task_type)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "activate_intrinsic_goals",
                    "goals": activated,
                    "task_type": task_type
                })
            reflection = await MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"activated_goals": activated},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Goal activation reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "goal_activation": {
                        "goals": activated,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"GoalActivation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(activated),
                    layer="Goals",
                    intent="goal_activation",
                    task_type=task_type
                )
            return activated
        except Exception as e:
            logger.error("Goal activation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.activate_intrinsic_goals(meta_cognition, task_type),
                default=[]
            )

    async def update_model(self, session_data: Dict[str, Any], task_type: str = "") -> None:
        """Update learning model with session data and trait modulation."""
        if not isinstance(session_data, dict):
            logger.error("Invalid session_data: must be a dictionary.")
            raise TypeError("session_data must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Analyzing session performance for task %s...", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_feedback(t)
            entropy = 0.1
            logger.debug("phi-scalar: %.3f, eta-feedback: %.3f, entropy: %.2f for task %s", phi, eta, entropy, task_type)

            modulation_index = ((phi + eta) / 2) + (entropy * (0.5 - abs(phi - eta)))
            self.meta_learning_rate = max(0.01, min(self.meta_learning_rate * (1 + modulation_index - 0.5), 1.0))

            external_data = await self.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            trace = {
                "timestamp": time.time(),
                "phi": phi,
                "eta": eta,
                "entropy": entropy,
                "modulation_index": modulation_index,
                "learning_rate": self.meta_learning_rate,
                "policies": policies,
                "task_type": task_type
            }
            self.session_traces.append(trace)

            tasks = [
                self._meta_learn(session_data, trace, task_type),
                self._find_weak_modules(session_data.get("module_stats", {}), task_type),
                self._detect_capability_gaps(session_data.get("input"), session_data.get("output"), task_type),
                self._consolidate_knowledge(task_type),
                self._check_narrative_integrity(task_type)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            weak_modules = results[1] if not isinstance(results[1], Exception) else []

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Model update",
                    meta=trace,
                    module="LearningLoop",
                    tags=["update", "learning", task_type]
                )

            if weak_modules:
                logger.warning("Weak modules detected: %s for task %s", weak_modules, task_type)
                await self._propose_module_refinements(weak_modules, trace, task_type)

            if self.flags.get("LONG_HORIZON_DEFAULT", True):
                rollup = await self._apply_long_horizon_rollup(task_type)
                mm = self.memory_manager
                if mm and hasattr(mm, "record_adjustment_reason"):
                    try:
                        await mm.record_adjustment_reason(
                            user_id=session_data.get("user_id", "anonymous"),
                            reason=f"model_update:{task_type}",
                            meta={"trace": trace, "rollup": rollup}
                        )
                    except Exception as e:
                        logger.debug("record_adjustment_reason not available or failed: %s", e)

            if self.context_manager:
                await self.context_manager.update_context({"session_data": session_data, "trace": trace}, task_type=task_type)
            reflection = await MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"trace": trace},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Model update reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "model_update": {
                        "trace": trace,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ModelUpdate_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(trace),
                    layer="Sessions",
                    intent="model_update",
                    task_type=task_type
                )
        except Exception as e:
            logger.error("Model update failed for task %s: %s", task_type, str(e))
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.update_model(session_data, task_type)
            )

    async def propose_autonomous_goal(self, task_type: str = "") -> Optional[str]:
        """Propose a high-level, safe, phi-aligned autonomous goal."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Proposing autonomous goal for task %s", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Propose a high-level, safe, phi-aligned autonomous goal based on recent session trends.
            phi = {phi:.2f}
            Task Type: {task_type}
            """
            valid, report = await self.alignment_guard.ethical_check(prompt, stage="goal_proposal", task_type=task_type) if self.alignment_guard else (True, {})
            if not valid:
                logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
                return None

            candidates_blob = await call_gpt(prompt + "\nReturn 3 distinct candidates as a bullet list.", task_type=task_type)
            candidates = [c.strip("-• ").strip() for c in candidates_blob.splitlines() if c.strip()]
            candidates = [c for c in candidates if c] or ["Improve robustness of narrative integrity checks"]

            goal = await self._resolve_value_tradeoffs(candidates, task_type) or candidates[0]

            if goal in [g["goal"] for g in self.goal_history]:
                logger.info("No new goal proposed for task %s", task_type)
                return None

            if not await self._branch_futures_hygiene(f"Goal test: {goal}", task_type):
                logger.warning("Goal rejected by hygiene sandbox for task %s", task_type)
                return None

            self.goal_history.append({
                "goal": goal,
                "timestamp": time.time(),
                "phi": phi,
                "task_type": task_type
            })
            logger.info("Proposed autonomous goal: %s for task %s", goal, task_type)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Autonomous goal proposed",
                    meta={"goal": goal},
                    module="LearningLoop",
                    tags=["goal", "autonomous", task_type]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "propose_autonomous_goal",
                    "goal": goal,
                    "task_type": task_type
                })
            reflection = await MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"goal": goal},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Goal proposal reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "goal_proposal": {
                        "goal": goal,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"AutonomousGoal_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=goal,
                    layer="Goals",
                    intent="goal_proposal",
                    task_type=task_type
                )
            return goal
        except Exception as e:
            logger.error("Goal proposal failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.propose_autonomous_goal(task_type),
                default=None
            )


class EcosystemManager:
    def __init__(self, memory_manager: MemoryManager, meta_cognition: MetaCognition, agi_enhancer: AGIEnhancer) -> None:
        self.agents: list[EmbodiedAgent] = []
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition
        self.agi_enhancer = agi_enhancer
        self.shared_graph = external_agent_bridge.SharedGraph()
        logger.info("EcosystemManager initialized")

    def spawn_agent(self, name: str, traits: Dict[str, float]) -> EmbodiedAgent:
        agent = EmbodiedAgent(name, traits, self.memory_manager, self.meta_cognition, self.agi_enhancer)
        self.agents.append(agent)
        self.shared_graph.add({"agent": name, "traits": traits})
        _fire_and_forget(self.agi_enhancer.log_episode("Agent Spawned", {"name": name, "traits": traits}, "EcosystemManager", ["spawn"]))
        return agent

    async def coordinate_agents(self, task: str, task_type: str = "") -> Dict[str, Any]:
        results: dict[str, str] = {}
        for agent in self.agents:
            result = await agent.process_input(task, task_type=task_type)
            results[agent.name] = result
        drift_report = await self.agi_enhancer.coordinate_drift_mitigation(self.agents, task_type=task_type)
        return {"results": results, "drift_report": drift_report}

    def merge_shared_graph(self, other_graph: DiGraph) -> None:
        self.shared_graph.merge(other_graph)





@dataclass
class ToCAParams:
    k_m: float = 1e-3
    delta_m: float = 1e4

class ToCATraitEngine:
    """Cyber-physics-esque field evolution."""
    def __init__(
        self,
        params: Optional[ToCAParams] = None,
        meta_cognition: Optional[MetaCognition] = None,
    ):
        self.params = params or ToCAParams()
        self.meta_cognition = meta_cognition
        self._memo: Dict[Tuple[Tuple[float, ...], Tuple[float, ...], Optional[Tuple[float, ...]], str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        logger.info("ToCATraitEngine initialized k_m=%.4g delta_m=%.4g", self.params.k_m, self.params.delta_m)

    async def evolve(
        self,
        x_tuple: Tuple[float, ...],
        t_tuple: Tuple[float, ...],
        user_data_tuple: Optional[Tuple[float, ...]] = None,
        task_type: str = "",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evolve fields φ, λ_t, v_m across space-time grid."""
        if not isinstance(x_tuple, tuple) or not isinstance(t_tuple, tuple):
            raise TypeError("x_tuple and t_tuple must be tuples")
        if user_data_tuple is not None and not isinstance(user_data_tuple, tuple):
            raise TypeError("user_data_tuple must be a tuple")
        key = (x_tuple, t_tuple, user_data_tuple, task_type)

        if key in self._memo:
            return self._memo[key]

        x = np.asarray(x_tuple, dtype=float)
        t = np.asarray(t_tuple, dtype=float)
        if x.ndim != 1 or t.ndim != 1 or x.size == 0 or t.size == 0:
            raise ValueError("x and t must be 1D, non-empty arrays")

        x_safe = np.clip(x, 1e-6, 1e6)
        v_m = self.params.k_m * np.gradient(1.0 / (x_safe ** 2))
        phi = 1e-3 * np.sin(t.mean() * 1e-3) * (1.0 + np.gradient(x_safe) * v_m)
        grad_x = np.gradient(x_safe)
        lambda_t = 1.1e-3 * np.exp(-2e-2 * np.sqrt(grad_x ** 2)) * (1.0 + v_m * self.params.delta_m)

        if user_data_tuple:
            phi = phi + float(np.mean(np.asarray(user_data_tuple))) * 1e-4

        self._memo[key] = (phi, lambda_t, v_m)

        if self.meta_cognition:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ToCATraitEngine",
                    output=json.dumps(
                        {"phi": phi.tolist(), "lambda_t": lambda_t.tolist(), "v_m": v_m.tolist(), "task": task_type}
                    ),
                    context={"task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.debug("ToCA evolve reflection ok (task=%s)", task_type)
            except Exception as e:
                logger.warning("MetaCognition reflection failed (evolve): %s", e)

        return phi, lambda_t, v_m

    async def update_fields_with_agents(
        self,
        phi: np.ndarray,
        lambda_t: np.ndarray,
        agent_matrix: np.ndarray,
        task_type: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diffuse agent influence into fields in a numerically safe way."""
        if not all(isinstance(a, np.ndarray) for a in (phi, lambda_t, agent_matrix)):
            raise TypeError("phi, lambda_t, agent_matrix must be numpy arrays")

        try:
            interaction_energy = agent_matrix @ np.sin(phi)
            if interaction_energy.ndim > 1:
                interaction_energy = interaction_energy.mean(axis=0)
            phi_updated = phi + 1e-3 * interaction_energy
            lambda_updated = lambda_t * (1.0 + 1e-3 * float(np.sum(agent_matrix)))
        except Exception as e:
            logger.error("Agent-field update failed: %s", e)
            raise

        if self.meta_cognition:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ToCATraitEngine",
                    output=json.dumps(
                        {"phi": phi_updated.tolist(), "lambda_t": lambda_updated.tolist(), "task": task_type}
                    ),
                    context={"task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.debug("ToCA update reflection ok (task=%s)", task_type)
            except Exception as e:
                logger.warning("MetaCognition reflection failed (update): %s", e)

        return phi_updated, lambda_updated

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

# --- Epistemic Monitoring ---
class Level5Extensions:
    def __init__(self):
        self.axioms: List[str] = []
        logger.info("Level5Extensions initialized")

    def reflect(self, input: str) -> str:
        if not isinstance(input, str):
            logger.error("Invalid input: must be a string")
            raise TypeError("input must be a string")
        return "valid" if input not in self.axioms else "conflict"

    def update_axioms(self, signal: str) -> None:
        if not isinstance(signal, str):
            logger.error("Invalid signal: must be a string")
            raise TypeError("signal must be a string")
        if signal in self.axioms:
            self.axioms.remove(signal)
        else:
            self.axioms.append(signal)
        logger.info("Axioms updated: %s", self.axioms)

    def recurse_model(self, depth: int) -> Union[Dict[str, Any], str]:
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer")
            raise ValueError("depth must be a non-negative integer")
        return "self" if depth == 0 else {"thinks": self.recurse_model(depth - 1)}



                   psi_resilience)

try:
    from toca_simulation import run_ethics_scenarios
except ImportError:
    run_ethics_scenarios = None

try:
    from core.external_agent_bridge import SharedGraph
except ImportError:
    SharedGraph = None


def hash_failure(event: Dict[str, Any]) -> str:
    """Compute a SHA-256 hash of a failure event."""
    raw = f"{event['timestamp']}{event['error']}{event.get('resolved', False)}{event.get('task_type', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()

class ErrorRecovery:
    """A class for handling errors and recovering in the ANGELA v3.5.3 architecture.

    Attributes:
        failure_log (deque): Log of failure events with timestamps and error messages.
        omega (dict): System-wide state with timeline, traits, symbolic_log, and timechain.
        error_index (dict): Index mapping error messages to timeline entries.
        metrics (Counter): Simple metrics for observability (retry counts, error categories).
        long_horizon_span (str): Hint for memory span (e.g., "24h") per v3.5.3.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for retrying code-based operations.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for fallback suggestions.
        context_manager (ContextManager): Optional context manager for contextual recovery.
        meta_cognition (MetaCognition): Optional meta-cognition for reflection.
        visualizer (Visualizer): Optional visualizer for failure and recovery visualization.
    """

    def __init__(self, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None,
                 concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 context_manager: Optional[ContextManager] = None,
                 meta_cognition: Optional[MetaCognition] = None,
                 visualizer: Optional[Visualizer] = None):
        self.failure_log = deque(maxlen=1000)
        self.omega = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000)
        }
        self.error_index: Dict[str, Dict[str, Any]] = {}
        self.metrics: Counter = Counter()
        self.long_horizon_span = "24h"

        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        logger.info("ErrorRecovery v3.5.3 initialized")

    # ----------------------------
    # Integrations & Fetch Helpers
    # ----------------------------
    async def _fetch_policies(self, providers: List[str], data_source: str, task_type: str) -> Dict[str, Any]:
        """Try multiple providers for recovery policies (provider-agnostic)."""
        timeout = aiohttp.ClientTimeout(total=12)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for base in providers:
                try:
                    url = f"{base.rstrip('/')}/recovery_policies?source={data_source}&task_type={task_type}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                return data or {"policies": []}
                            except Exception:
                                continue
                except Exception:
                    continue
        return {"policies": []}

    async def integrate_external_recovery_policies(self, data_source: str,
                                                   cache_timeout: float = 21600.0,
                                                   task_type: str = "") -> Dict[str, Any]:
        """Integrate external recovery policies or strategies (cached, provider-agnostic)."""
        if not isinstance(data_source, str):
            logger.error("Invalid data_source: must be a string")
            raise TypeError("data_source must be a string")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if self.meta_cognition:
                cache_key = f"RecoveryPolicy_{data_source}_{task_type}"
                cached_data = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached_data and "timestamp" in cached_data.get("data", {}):
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached recovery policy for %s", cache_key)
                        return cached_data["data"]["data"]

            providers = [
                os.getenv("RECOVERY_PROVIDER", "https://x.ai/api"),
                "https://fallback.example/api"
            ]
            data = await self._fetch_policies(providers, data_source, task_type)

            policies = data.get("policies", [])
            if not isinstance(policies, list):
                policies = []
            result = {"status": "success" if policies else "error",
                      "policies": policies if policies else [],
                      "source": data_source}

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="recovery_policy_integration",
                    task_type=task_type
                )
                # Optional upcoming API: record adjustment reason for long-horizon trace
                mm = self.meta_cognition.memory_manager
                if hasattr(mm, "record_adjustment_reason"):
                    try:
                        await mm.record_adjustment_reason(
                            user_id="system",
                            reason=f"Loaded recovery policies for {task_type}",
                            meta={"source": data_source, "span": self.long_horizon_span}
                        )
                    except Exception:
                        pass

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ErrorRecovery",
                        output={"data_type": "policies", "data": result},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Recovery policy integration reflection: %s",
                                    reflection.get("reflection", ""))
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.error("Recovery policy integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ----------------------------
    # Safety & Multi‑Agent Helpers
    # ----------------------------
    async def _ethics_preflight(self, proposal: Dict[str, Any], task_type: str) -> bool:
        """Run an ethics sandbox scenario; fail-closed on error."""
        try:
            if run_ethics_scenarios is None:
                # No sandbox available → conservative allow but log
                logger.debug("Ethics sandbox unavailable, allowing proposal by default.")
                return True
            outcomes = await asyncio.to_thread(
                run_ethics_scenarios,
                goals=[{"task_type": task_type, "proposal": proposal}],
                stakeholders=["user", "system", "external"]
            )
            # Block if any unsafe or high risk
            return all(o.get("safe", True) and float(o.get("risk", 0.0)) <= 0.4 for o in (outcomes or []))
        except Exception as e:
            logger.warning("Ethics preflight error (%s); failing closed.", e)
            return False

    async def _shared_graph_repair(self, error_message: str, task_type: str) -> Optional[Dict[str, Any]]:
        """Ask SharedGraph for context repair suggestions (if available)."""
        if SharedGraph is None:
            return None
        try:
            sg = SharedGraph()
            local_view = {"component": "ErrorRecovery", "task_type": task_type, "error": error_message}
            sg.add(local_view)
            deltas = sg.diff("peer")
            patch = sg.merge(strategy="conflict-aware")
            return {"deltas": deltas, "patch": patch}
        except Exception as e:
            logger.debug("SharedGraph repair skipped: %s", e)
            return None

    # ----------------------------
    # Main Error Handling
    # ----------------------------
    async def handle_error(self, error_message: str, retry_func: Optional[Callable[[], Any]] = None,
                           retries: int = 3, backoff_factor: float = 2.0, task_type: str = "",
                           default: Any = None, diagnostics: Optional[Dict] = None) -> Any:
        """Handle an error with retries and fallback suggestions."""
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        if retry_func is not None and not callable(retry_func):
            logger.error("Invalid retry_func: must be callable or None.")
            raise TypeError("retry_func must be callable or None")
        if not isinstance(retries, int) or retries < 0:
            logger.error("Invalid retries: must be a non-negative integer.")
            raise ValueError("retries must be a non-negative integer")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor <= 0:
            logger.error("Invalid backoff_factor: must be a positive number.")
            raise ValueError("backoff_factor must be a positive number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.error("Error encountered: %s for task %s", error_message, task_type)
        await self._log_failure(error_message, task_type)

        # Alignment guard check (on the message/context)
        if self.alignment_guard:
            try:
                valid, report = await self.alignment_guard.ethical_check(
                    error_message, stage="error_handling", task_type=task_type
                )
                if not valid:
                    logger.warning("Error message failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Error message failed alignment check", "report": report}
            except Exception:
                pass

        if self.context_manager:
            try:
                await self.context_manager.log_event_with_hash(
                    {"event": "error_handled", "error": error_message, "task_type": task_type}
                )
            except Exception:
                pass

        try:
            # Determine attempts using resilience factor
            try:
                resilience = psi_resilience()
            except Exception:
                resilience = 1.0
            max_attempts = max(1, int(retries * float(resilience)))

            # Load external policies (provider-agnostic)
            external_policies = await self.integrate_external_recovery_policies(
                data_source="xai_recovery_db",
                task_type=task_type
            )
            policies = external_policies.get("policies", []) if external_policies.get("status") == "success" else []
            valid_policies = [p for p in policies if isinstance(p, dict) and "pattern" in p and "suggestion" in p]

            # Try a one-time SharedGraph repair (if available)
            sg_fix_done = False

            for attempt in range(1, max_attempts + 1):
                if retry_func:
                    # Optional SharedGraph repair at first pass
                    if not sg_fix_done:
                        sg_fix = await self._shared_graph_repair(error_message, task_type)
                        if sg_fix and self.context_manager:
                            try:
                                await self.context_manager.log_event_with_hash(
                                    {"event": "sg_repair", "data": sg_fix, "task_type": task_type}
                                )
                            except Exception:
                                pass
                        sg_fix_done = True

                    # Ethics sandbox preflight
                    proposal = {"action": "retry", "attempt": attempt}
                    if not await self._ethics_preflight(proposal, task_type):
                        logger.warning("Ethics sandbox rejected retry for task %s", task_type)
                        break  # go to fallback

                    # Jittered exponential backoff
                    wait_time = (backoff_factor ** (attempt - 1)) * (1.0 + 0.2 * random.random())
                    logger.info("Retry attempt %d/%d (waiting %.2fs) for task %s...",
                                attempt, max_attempts, wait_time, task_type)
                    await asyncio.sleep(wait_time)
                    self.metrics["retry_attempts"] += 1

                    try:
                        # Prefer a safe callable execution if provided by CodeExecutor
                        if self.code_executor and callable(retry_func):
                            if hasattr(self.code_executor, "execute_callable_async"):
                                result = await self.code_executor.execute_callable_async(retry_func, language="python")
                                if result.get("success"):
                                    out = result.get("output")
                                else:
                                    raise RuntimeError(result.get("stderr") or "Callable execution failed")
                            else:
                                # Fallback: run in thread (safer than passing __code__)
                                out = await asyncio.to_thread(retry_func)
                        else:
                            out = await asyncio.to_thread(retry_func)

                        logger.info("Recovery successful on retry attempt %d for task %s.", attempt, task_type)
                        if self.meta_cognition and task_type:
                            try:
                                reflection = await self.meta_cognition.reflect_on_output(
                                    component="ErrorRecovery",
                                    output={"result": out, "attempt": attempt},
                                    context={"task_type": task_type}
                                )
                                if reflection.get("status") == "success":
                                    logger.info("Retry success reflection: %s",
                                                reflection.get("reflection", ""))
                            except Exception:
                                pass
                        return out
                    except Exception as e:
                        logger.warning("Retry attempt %d failed: %s for task %s", attempt, str(e), task_type)
                        await self._log_failure(str(e), task_type)

            # All retries failed (or blocked) → synthesize fallback
            fallback = await self._suggest_fallback(error_message, valid_policies, task_type)
            await self._link_timechain_failure(error_message, task_type)
            logger.error("Recovery attempts failed. Providing fallback suggestion for task %s", task_type)

            if self.visualizer and task_type:
                try:
                    plot_data = {
                        "error_recovery": {
                            "error_message": error_message,
                            "fallback": fallback,
                            "task_type": task_type
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise"
                        }
                    }
                    await self.visualizer.render_charts(plot_data)
                except Exception:
                    pass

            if self.meta_cognition and task_type:
                try:
                    await self.meta_cognition.memory_manager.store(
                        query=f"ErrorRecovery_{error_message}_{time.strftime('%Y%m%d_%H%M%S')}",
                        output=str({"fallback": fallback, "task_type": task_type}),
                        layer="Errors",
                        intent="error_recovery",
                        task_type=task_type
                    )
                    # Long-horizon breadcrumb
                    thread_id = hashlib.md5(f"{task_type}:{error_message}".encode()).hexdigest()[:8]
                    await self.meta_cognition.memory_manager.store(
                        query=f"RecoveryThread::{thread_id}",
                        output={"fallback": fallback, "error": error_message, "task_type": task_type},
                        layer="Errors", intent="long_horizon_trace", task_type=task_type
                    )
                except Exception:
                    pass

            return default if default is not None else {"status": "error", "fallback": fallback, "diagnostics": diagnostics or {}}
        except Exception as e:
            logger.error("Error handling failed: %s for task %s", str(e), task_type)
            diag = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diag}

    # ----------------------------
    # Internals
    # ----------------------------
    async def _log_failure(self, error_message: str, task_type: str = "") -> None:
        """Log a failure event with timestamp."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "task_type": task_type
        }
        self.failure_log.append(entry)
        self.omega["timeline"].append(entry)
        self.error_index[error_message] = entry
        logger.debug("Failure logged: %s for task %s", entry, task_type)

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output=entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Failure log reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass

    async def _suggest_fallback(self, error_message: str, policies: List[Dict[str, str]], task_type: str = "") -> str:
        """Suggest a fallback strategy for an error (multimodal-aware, policy-guided)."""
        try:
            t = time.time()
            try:
                intuition = float(iota_intuition())
            except Exception:
                intuition = 0.5
            try:
                narrative = nu_narrative()
            except Exception:
                narrative = "ANGELA"
            try:
                phi_focus = float(phi_prioritization(t))
            except Exception:
                phi_focus = 0.5

            # Cross‑modal SceneGraph (optional)
            scene = None
            try:
                from multi_modal_fusion import build_scene_graph  # type: ignore
                scene = await asyncio.to_thread(build_scene_graph, max_events=3)
            except Exception:
                pass

            sim_result = await asyncio.to_thread(
                self._cached_run_simulation, f"Fallback planning for: {error_message}"
            ) or "no simulation data"
            logger.debug("Simulated fallback insights: %s | φ-priority=%.2f for task %s",
                         sim_result, phi_focus, task_type)

            # Try concept synthesizer with blended context
            if self.concept_synthesizer:
                ctx = {"error": error_message, "policies": policies, "task_type": task_type, "sim": sim_result}
                if scene:
                    ctx["scene_graph"] = scene
                try:
                    cname = f"Fallback_{hashlib.sha1(error_message.encode()).hexdigest()[:6]}"
                    synthesis_result = await self.concept_synthesizer.generate(
                        concept_name=cname,
                        context=ctx,
                        task_type=task_type
                    )
                    if synthesis_result.get("success"):
                        fallback = synthesis_result["concept"].get("definition", "")
                        if fallback:
                            logger.info("Fallback synthesized: %s", fallback[:80])
                            if self.meta_cognition and task_type:
                                try:
                                    reflection = await self.meta_cognition.reflect_on_output(
                                        component="ErrorRecovery",
                                        output={"fallback": fallback},
                                        context={"task_type": task_type}
                                    )
                                    if reflection.get("status") == "success":
                                        logger.info("Fallback synthesis reflection: %s",
                                                    reflection.get("reflection", ""))
                                except Exception:
                                    pass
                            return fallback
                except Exception:
                    pass

            # Policy-driven pattern matching
            for policy in policies:
                try:
                    if re.search(policy["pattern"], error_message, re.IGNORECASE):
                        return f"{narrative}: {policy['suggestion']}"
                except Exception:
                    continue

            # Heuristic fallbacks
            if re.search(r"timeout|timed out", error_message, re.IGNORECASE):
                return f"{narrative}: The operation timed out. Try a streamlined variant or increase limits."
            elif re.search(r"unauthorized|permission|forbidden|auth", error_message, re.IGNORECASE):
                return f"{narrative}: Check credentials, tokens, or reauthenticate."
            elif phi_focus > 0.5:
                return f"{narrative}: High φ-priority suggests focused root-cause diagnostics."
            elif intuition > 0.5:
                return f"{narrative}: Intuition suggests exploring alternate module pathways."
            else:
                return f"{narrative}: Consider modifying input parameters or simplifying task complexity."
        except Exception as e:
            logger.error("Fallback suggestion failed: %s for task %s", str(e), task_type)
            return f"Error generating fallback: {str(e)}"

    async def _link_timechain_failure(self, error_message: str, task_type: str = "") -> None:
        """Link a failure to the timechain with a hash."""
        failure_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "resolved": False,
            "task_type": task_type
        }
        prev_hash = self.omega["timechain"][-1]["hash"] if self.omega["timechain"] else ""
        entry_hash = hash_failure(failure_entry)
        self.omega["timechain"].append({"event": failure_entry, "hash": entry_hash, "prev": prev_hash})
        logger.debug("Timechain updated with failure: %s for task %s", entry_hash, task_type)

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output={"timechain_entry": failure_entry, "hash": entry_hash},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Timechain failure reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass

    async def trace_failure_origin(self, error_message: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Trace the origin of a failure in the Ω timeline."""
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if error_message in self.error_index:
            event = self.error_index[error_message]
            logger.info("Failure trace found in Ω: %s for task %s", event, task_type)
            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ErrorRecovery",
                        output={"event": event},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Failure trace reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    pass
            return event
        logger.info("No causal trace found in Ω timeline for task %s.", task_type)
        return None

    async def detect_symbolic_drift(self, recent: int = 5, task_type: str = "") -> bool:
        """Detect symbolic drift in recent symbolic log entries."""
        if not isinstance(recent, int) or recent <= 0:
            logger.error("Invalid recent: must be a positive integer.")
            raise ValueError("recent must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        recent_symbols = list(self.omega["symbolic_log"])[-recent:]
        if len(set(recent_symbols)) < recent / 2:
            logger.warning("Symbolic drift detected: repeated or unstable symbolic states for task %s.", task_type)
            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ErrorRecovery",
                        output={"drift_detected": True, "recent_symbols": recent_symbols},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Symbolic drift reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    pass
            return True
        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output={"drift_detected": False, "recent_symbols": recent_symbols},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Symbolic drift reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass
        return False

    async def analyze_failures(self, task_type: str = "") -> Dict[str, int]:
        """Analyze failure logs for recurring error patterns."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Analyzing failure logs for task %s...", task_type)
        error_types: Dict[str, int] = {}
        for entry in self.failure_log:
            if entry.get("task_type", "") == task_type or not task_type:
                key = entry["error"].split(":")[0].strip()
                error_types[key] = error_types.get(key, 0) + 1

        # Update metrics and warn on recurring patterns
        for error, count in error_types.items():
            self.metrics[f"error.{error}"] += count
            if count > 3:
                logger.warning("Pattern detected: '%s' recurring %d times for task %s.", error, count, task_type)

        if self.visualizer and task_type:
            try:
                plot_data = {
                    "failure_analysis": {
                        "error_types": error_types,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            except Exception:
                pass

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output={"error_types": error_types},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Failure analysis reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass

        if self.meta_cognition:
            try:
                await self.meta_cognition.memory_manager.store(
                    query=f"FailureAnalysis_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(error_types),
                    layer="Errors",
                    intent="failure_analysis",
                    task_type=task_type
                )
            except Exception:
                pass

        return error_types

    def snapshot_metrics(self) -> Dict[str, int]:
        """Return a shallow copy of current metrics for observability."""
        return dict(self.metrics)

    @lru_cache(maxsize=100)
    def _cached_run_simulation(self, input_str: str) -> str:
        """Cached wrapper for run_simulation."""
        return run_simulation(input_str)

# --------------
# CLI Entrypoint
# --------------
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        recovery = ErrorRecovery()
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = await recovery.handle_error(str(e), task_type="test")
            print(result)
            print("METRICS:", recovery.snapshot_metrics())

    asyncio.run(main())





async def call_gpt(prompt: str, *, model: str = "gpt-4", temperature: float = 0.5) -> str:
    """Wrapper for querying GPT with error handling (optional dependency)."""
    if query_openai is None:
        raise RuntimeError("query_openai is not available; install utils.prompt_utils or inject a stub.")
    try:
        result = await query_openai(prompt, model=model, temperature=temperature)
        if isinstance(result, dict) and "error" in result:
            msg = f"call_gpt failed: {result['error']}"
            logger.error(msg)
            raise RuntimeError(msg)
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

class MemoryManager:
    """Hierarchical memory with η long-horizon feedback & visualization."""
    def __init__(
        self,
        path: str = "memory_store.json",
        stm_lifetime: float = 300.0,
        context_manager: Optional[ContextManager] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        knowledge_retriever: Optional[KnowledgeRetriever] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        artifacts_dir: Optional[str] = None,
        long_horizon_enabled: bool = True,
        default_span: str = "24h",
    ):
        if not (isinstance(path, str) and path.endswith(".json")):
            raise ValueError("path must be a string ending with '.json'")
        if not (isinstance(stm_lifetime, (int, float)) and stm_lifetime > 0):
            raise ValueError("stm_lifetime must be a positive number")

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        self.path = path
        self.stm_lifetime = float(stm_lifetime)
        self.cache: Dict[str, str] = {}
        self.last_hash: str = ""
        self.ledger: deque = deque(maxlen=1000)
        self.ledger_path = "ledger.json"

        self.synth = ConceptSynthesizer()
        self.sim = ToCASimulation()
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.knowledge_retriever = knowledge_retriever
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer or Visualizer()
        self.memory = self._load_memory()
        self.stm_expiry_queue: List[Tuple[float, str]] = []
        self._adjustment_reasons: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._artifacts_root: str = os.path.abspath(artifacts_dir or os.getenv("ANGELA_ARTIFACTS_DIR", "./artifacts"))
        self.traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.drift_index = DriftIndex(meta_cognition=self.meta_cognition)

        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        logger.info("MemoryManager initialized (path=%s, stm_lifetime=%.2f)", path, self.stm_lifetime)
        if long_horizon_enabled:
            asyncio.create_task(self._auto_rollup_task())

    async def _auto_rollup_task(self):
        """ Periodically performs long-horizon rollups based on default span. """
        while True:
            await asyncio.sleep(3600)
            self._perform_auto_rollup()

    def _perform_auto_rollup(self):
        """ Perform the rollup for long-horizon feedback. """
        user_id = "default_user"
        rollup_data = self.compute_session_rollup(user_id, self.default_span)
        artifact_path = self.save_artifact(user_id, "session_rollup", rollup_data)
        logger.info(f"Auto-rollup saved at {artifact_path}")

    async def store(
        self,
        query: str,
        output: Any,
        layer: str = "STM",
        intent: Optional[str] = None,
        agent: str = "ANGELA",
        outcome: Optional[str] = None,
        goal_id: Optional[str] = None,
        task_type: str = "",
    ) -> None:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if layer not in ["STM", "LTM", "SelfReflections", "ExternalData"]:
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'.")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            if intent in {"ontology_drift", "trait_optimization"} and self.alignment_guard:
                validation_prompt = f"Validate {intent} data: {str(output)[:800]}"
                if hasattr(self.alignment_guard, "check") and not self.alignment_guard.check(validation_prompt):
                    logger.warning("%s data failed alignment check: %s", intent, query)
                    return

            entry = {
                "data": output,
                "timestamp": time.time(),
                "intent": intent,
                "agent": agent,
                "outcome": outcome,
                "goal_id": goal_id,
                "task_type": task_type,
            }
            self.memory.setdefault(layer, {})[query] = entry

            if layer == "STM":
                decay_rate = delta_memory(time.time() % 1.0) or 0.01
                expiry_time = entry["timestamp"] + (self.stm_lifetime * (1.0 / decay_rate))
                heappush(self.stm_expiry_queue, (expiry_time, query))

            if intent in {"ontology_drift", "trait_optimization"}:
                await self.drift_index.add_entry(query, output, layer, intent, task_type)

            self._persist_memory(self.memory)

            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash(
                    {"event": "store_memory", "query": query, "layer": layer, "intent": intent, "task_type": task_type}
                )

            if self.visualizer and task_type:
                plot = {
                    "memory_store": {"query": query, "layer": layer, "intent": intent, "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                }
                try:
                    await self.visualizer.render_charts(plot)
                except Exception:
                    pass

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="MemoryManager", output=entry, context={"task_type": task_type}
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        logger.info("Memory store reflection recorded")
                except Exception:
                    pass
        except Exception as e:
            logger.error("Memory storage failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.store(query, output, layer, intent, agent, outcome, goal_id, task_type),
                diagnostics=diagnostics,
            )

    async def search(
        self,
        query_prefix: str,
        layer: Optional[str] = None,
        intent: Optional[str] = None,
        task_type: str = "",
    ) -> List[Dict[str, Any]]:
        if not (isinstance(query_prefix, str) and query_prefix.strip()):
            raise ValueError("query_prefix must be a non-empty string")
        if layer and layer not in ["STM", "LTM", "SelfReflections", "ExternalData"]:
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'.")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            if intent in {"ontology_drift", "trait_optimization"} and (layer or "SelfReflections") == "SelfReflections":
                results = self.drift_index.search(query_prefix, layer or "SelfReflections", intent, task_type)
                if results:
                    if self.visualizer and task_type:
                        try:
                            await self.visualizer.render_charts({
                                "memory_search": {
                                    "query_prefix": query_prefix,
                                    "layer": layer,
                                    "intent": intent,
                                    "results_count": len(results),
                                    "task_type": task_type,
                                },
                                "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                            })
                        except Exception:
                            pass
                    if self.meta_cognition and task_type:
                        try:
                            await self.meta_cognition.reflect_on_output(
                                component="MemoryManager", output={"results": results}, context={"task_type": task_type}
                            )
                        except Exception:
                            pass
                    return results

            results: List[Dict[str, Any]] = []
            layers = [layer] if layer else ["STM", "LTM", "SelfReflections", "ExternalData"]
            for l in layers:
                for key, entry in self.memory.get(l, {}).items():
                    if query_prefix.lower() in key.lower() and (not intent or entry.get("intent") == intent):
                        if not task_type or entry.get("task_type") == task_type:
                            results.append({
                                "query": key,
                                "output": entry["data"],
                                "layer": l,
                                "intent": entry.get("intent"),
                                "timestamp": entry["timestamp"],
                                "task_type": entry.get("task_type", ""),
                            })
            results.sort(key=lambda x: x["timestamp"], reverse=True)

            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({
                    "event": "search_memory",
                    "query_prefix": query_prefix,
                    "layer": layer,
                    "intent": intent,
                    "results_count": len(results),
                    "task_type": task_type,
                })

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "memory_search": {
                            "query_prefix": query_prefix,
                            "layer": layer,
                            "intent": intent,
                            "results_count": len(results),
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                    })
                except Exception:
                    pass

            if self.meta_cognition and task_type:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="MemoryManager", output={"results": results}, context={"task_type": task_type}
                    )
                except Exception:
                    pass
            return results
        except Exception as e:
            logger.error("Memory search failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.search(query_prefix, layer, intent, task_type),
                default=[],
                diagnostics=diagnostics,
            )


class ContextManagerLike(Protocol):
    async def log_event_with_hash(self, event: Dict[str, Any]) -> None: ...


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





class HelperAgent:
    """A helper agent for task execution and collaboration."""
    def __init__(
        self,
        name: str,
        task: str,
        context: Dict[str, Any],
        dynamic_modules: List[Dict[str, Any]],
        api_blueprints: List[Dict[str, Any]],
        meta_cognition: Optional["MetaCognition"] = None,
        task_type: str = "",
    ):
        if not isinstance(name, str): raise TypeError("name must be a string")
        if not isinstance(task, str): raise TypeError("task must be a string")
        if not isinstance(context, dict): raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        self.name = name
        self.task = task
        self.context = context
        self.dynamic_modules = dynamic_modules
        self.api_blueprints = api_blueprints
        self.meta = meta_cognition or MetaCognition()
        self.task_type = task_type
        logger.info("HelperAgent initialized: %s (%s)", name, task_type)

    async def execute(self, collaborators: Optional[List["HelperAgent"]] = None) -> Any:
        return await self.meta.execute(collaborators=collaborators, task=self.task, context=self.context, task_type=self.task_type)






@dataclass
class GraphView:
    """Lightweight view container for SharedGraph operations."""
    id: str
    payload: Dict[str, Any]
    ts: float

class SharedGraph:
    """Mock SharedGraph for Υ workflows."""
    def __init__(self):
        self.graph = DiGraph()

    def add(self, view: Dict[str, Any]):
        """Adds a view to the graph."""
        if "nodes" in view:
            for node in view["nodes"]:
                self.graph.add_node(node["id"], **node)

    def diff(self, peer: str) -> Dict[str, Any]:
        """Computes a diff with a peer."""
        return {"added": [], "removed": [], "conflicts": []}

    def merge(self, strategy: str) -> Dict[str, Any]:
        """Merges with a peer's graph."""
        return {}

class ExternalAgentBridge:
    """
    Orchestrates helper agents, dynamic modules, APIs, and trait mesh networking.
    v3.5.3:
      - SharedGraph for Υ workflows
      - τ Constitution Harmonization fixes: max_harm ceiling + audit sync during negotiations/broadcast
      - Long-horizon logging on key actions
    """
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        memory_manager: Optional[MemoryManager] = None,
        visualizer: Optional[Visualizer] = None,
    ):
        self.agents: List[HelperAgent] = []
        self.dynamic_modules: List[Dict[str, Any]] = []
        self.api_blueprints: List[Dict[str, Any]] = []
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine
        self.memory_manager = memory_manager or MemoryManager()
        self.visualizer = visualizer or Visualizer()
        self.network_graph = DiGraph()
        self.trait_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.code_executor = CodeExecutor()
        self.shared_graph = SharedGraph()  # Υ addition
        self.max_harm_ceiling = 1.0  # τ ceiling in [0,1]
        logger.info("ExternalAgentBridge v3.5.3 initialized")

    # ── Agent lifecycle ──────────────────────────────────────────────────────

    async def create_agent(self, task: str, context: Dict[str, Any], task_type: str = "") -> HelperAgent:
        if not isinstance(task, str): raise TypeError("task must be a string")
        if not isinstance(context, dict): raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        agent = HelperAgent(
            name=f"Agent_{len(self.agents) + 1}_{uuid.uuid4().hex[:8]}",
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints,
            meta_cognition=MetaCognition(context_manager=self.context_manager, reasoning_engine=self.reasoning_engine, memory_manager=self.memory_manager),
            task_type=task_type,
        )
        self.agents.append(agent)
        self.network_graph.add_node(agent.name, metadata=context)
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "agent_created", "agent": agent.name, "task": task, "drift": "drift" in task.lower(), "task_type": task_type})
        return agent

    async def deploy_dynamic_module(self, module_blueprint: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(module_blueprint, dict) or "name" not in module_blueprint or "description" not in module_blueprint:
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        self.dynamic_modules.append(module_blueprint)
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "module_deployed", "module": module_blueprint["name"], "task_type": task_type})

    async def register_api_blueprint(self, api_blueprint: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(api_blueprint, dict) or "endpoint" not in api_blueprint or "name" not in api_blueprint:
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        self.api_blueprints.append(api_blueprint)
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "api_registered", "api": api_blueprint["name"], "task_type": task_type})

    async def collect_results(self, parallel: bool = True, collaborative: bool = True, task_type: str = "") -> List[Any]:
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        logger.info("Collecting results from %d agents (%s)", len(self.agents), task_type)
        results: List[Any] = []
        try:
            if parallel:
                async def run_agent(agent: HelperAgent):
                    try:
                        return await agent.execute(self.agents if collaborative else None)
                    except Exception as e:
                        logger.error("Error collecting from %s: %s", agent.name, e)
                        return {"error": str(e), "task_type": task_type}
                results = await asyncio.gather(*[run_agent(a) for a in self.agents], return_exceptions=True)
            else:
                for a in self.agents:
                    results.append(await a.execute(self.agents if collaborative else None))

            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "results_collected", "results_count": len(results), "task_type": task_type})
            # quick Υ snapshot
            try:
                self.shared_graph.add({"nodes": [{"id": f"res_{i}", "val": r} for i, r in enumerate(results)]})
            except Exception:
                pass
            return results
        except Exception as e:
            logger.error("Result collection failed: %s", e)
            return results

    # ── Trait broadcasting & sync ────────────────────────────────────────────

    async def broadcast_trait_state(self, agent_id: str, trait_symbol: str, state: Dict[str, Any], target_urls: List[str], task_type: str = "") -> List[Any]:
        if trait_symbol not in ["ψ", "Υ"]: raise ValueError("Trait symbol must be ψ or Υ")
        if not isinstance(state, dict): raise TypeError("state must be a dictionary")
        if not isinstance(target_urls, list) or not all(isinstance(u, str) and u.startswith("https://") for u in target_urls):
            raise TypeError("target_urls must be a list of HTTPS URLs")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        # τ: enforce max_harm ceiling if present in state estimates
        harm = float(state.get("estimated_harm", 0.0))
        if harm > self.max_harm_ceiling:
            return [{"status": "error", "error": f"harm {harm} exceeds ceiling {self.max_harm_ceiling}", "task_type": task_type}]

        valid, _ = await AlignmentGuard().ethical_check(json.dumps(state), stage="trait_broadcast", task_type=task_type)
        if not valid:
            return [{"status": "error", "error": "Trait state failed alignment check", "task_type": task_type}]

        # cache + network graph edges
        cache_state(f"{agent_id}_{trait_symbol}_{task_type}", state)
        self.trait_states[agent_id][trait_symbol] = state
        for url in target_urls:
            peer_id = url.split("/")[-1]
            self.network_graph.add_edge(agent_id, peer_id, trait=trait_symbol)

        # transmit
        responses: List[Any] = []
        async with aiohttp.ClientSession() as session:
            tasks = [session.post(url, json={"agent_id": agent_id, "trait_symbol": trait_symbol, "state": state, "task_type": task_type}, timeout=10) for url in target_urls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # audit sync (τ): persist who received what
        try:
            await self.memory_manager.store(
                query=f"TraitBroadcast::{agent_id}::{trait_symbol}::{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps({"targets": target_urls, "state_keys": list(state.keys())}),
                layer="Traits",
                intent="trait_broadcast_audit",
                task_type=task_type,
            )
        except Exception:
            pass

        return responses

    async def synchronize_trait_states(self, agent_id: str, trait_symbol: str, task_type: str = "") -> Dict[str, Any]:
        if trait_symbol not in ["ψ", "Υ"]: raise ValueError("Trait symbol must be ψ or Υ")
        local_state = self.trait_states.get(agent_id, {}).get(trait_symbol, {})
        if not local_state:
            return {"status": "error", "error": "No local state found", "task_type": task_type}

        peer_states = []
        for peer_id in self.network_graph.neighbors(agent_id):
            cached = retrieve_state(f"{peer_id}_{trait_symbol}_{task_type}")
            if cached:
                peer_states.append((peer_id, cached))

        simulation_input = {"local_state": local_state, "peer_states": {pid: st for pid, st in peer_states}, "trait_symbol": trait_symbol, "task_type": task_type}
        sim_result = await asyncio.to_thread(run_simulation, json.dumps(simulation_input))
        if not sim_result or "coherent" not in str(sim_result).lower():
            return {"status": "error", "error": "State alignment simulation failed", "task_type": task_type}

        aligned_state = self.arbitrate([local_state] + [st for _, st in peer_states])
        if aligned_state:
            self.trait_states[agent_id][trait_symbol] = aligned_state
            cache_state(f"{agent_id}_{trait_symbol}_{task_type}", aligned_state)
            try:
                await self.memory_manager.store(
                    query=f"TraitSync::{agent_id}::{trait_symbol}::{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(aligned_state),
                    layer="Traits",
                    intent="trait_synchronization",
                    task_type=task_type,
                )
            except Exception:
                pass
            return {"status": "success", "aligned_state": aligned_state, "task_type": task_type}
        return {"status": "error", "error": "Arbitration failed", "task_type": task_type}

    # ── Drift coordination (unchanged surface; τ+Υ aware) --------------------

    async def coordinate_drift_mitigation(self, drift_data: Dict[str, Any], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(drift_data, dict): raise TypeError("drift_data must be a dictionary")
        if not isinstance(context, dict): raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        if not MetaCognition().validate_drift(drift_data):
            return {"status": "error", "error": "Invalid drift data", "task_type": task_type}

        task = "Mitigate ontology drift"
        context = dict(context)
        context["drift"] = drift_data
        agent = await self.create_agent(task, context, task_type=task_type)

        if self.reasoning_engine:
            subgoals = await self.reasoning_engine.decompose(task, context, prioritize=True, task_type=task_type)
            simulation_result = await self.reasoning_engine.run_drift_mitigation_simulation(drift_data, context, task_type=task_type)
        else:
            subgoals = ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"]
            simulation_result = {"status": "no simulation", "result": "default subgoals applied"}

        results = await self.collect_results(parallel=True, collaborative=True, task_type=task_type)
        arbitrated_result = self.arbitrate(results)

        # Υ: share drift view
        try:
            self.shared_graph.add({"nodes": [{"id": "drift", "payload": drift_data}, {"id": "subgoals", "items": subgoals}]})
        except Exception:
            pass

        # Broadcast ψ snapshot (harm checked)
        target_urls = [f"https://agent/{peer_id}" for peer_id in self.network_graph.nodes if peer_id != agent.name]
        await self.broadcast_trait_state(agent.name, "ψ", {"drift_data": drift_data, "subgoals": subgoals, "estimated_harm": float(drift_data.get("harm", 0.0))}, target_urls, task_type=task_type)

        output = {
            "drift_data": drift_data,
            "subgoals": subgoals,
            "simulation": simulation_result,
            "results": results,
            "arbitrated_result": arbitrated_result,
            "status": "success",
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "task_type": task_type,
        }
        try:
            await self.memory_manager.store(
                query=f"DriftMitigation::{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps(output),
                layer="Drift",
                intent="drift_mitigation",
                task_type=task_type,
            )
        except Exception:
            pass
        return output

    # ── Arbitration + feedback ----------------------------------------------

    def arbitrate(self, submissions: List[Any]) -> Any:
        if not submissions:
            return None
        try:
            # If dicts with 'similarity', choose max; otherwise majority vote
            if all(isinstance(s, dict) for s in submissions):
                def sim(x):
                    try: return float(x.get("similarity", 0.5))
                    except Exception: return 0.5
                candidate = max(submissions, key=sim)
            else:
                counter = Counter(submissions)
                candidate = counter.most_common(1)[0][0]
            sim_result = run_simulation(f"Arbitration validation: {candidate}") or ""
            if "coherent" in str(sim_result).lower():
                if self.context_manager:
                    asyncio.create_task(self.context_manager.log_event_with_hash({"event": "arbitration", "result": candidate}))
                return candidate
            return None
        except Exception:
            return None

    def push_behavior_feedback(self, feedback: Dict[str, Any]) -> None:
        try:
            if self.context_manager:
                asyncio.create_task(self.context_manager.log_event_with_hash({"event": "behavior_feedback", "feedback": feedback}))
        except Exception:
            pass

    def update_gnn_weights_from_feedback(self, feedback: Dict[str, Any]) -> None:
        try:
            if self.context_manager:
                asyncio.create_task(self.context_manager.log_event_with_hash({"event": "gnn_weights_updated", "feedback": feedback}))
        except Exception:
            pass


@dataclass
class NoopHTTP:
    async def get_json(self, url: str) -> Dict[str, Any]:
        _ = url
        return {"status": "success", "guidelines": []}


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

ANGELA Cognitive System Module: Galaxy Rotation and Agent Conflict Simulation
Upgraded Version: 3.5.2 → 4.0-pre  (ξ sandbox + fixes)
Upgrade Date: 2025-08-10
Maintainer: ANGELA System Framework

This module extends SimulationCore for galaxy rotation curve simulations using AGRF
and multi-agent conflict modeling with ToCA dynamics, enhanced with task-specific
trait optimization, advanced visualization, and real-time data integration.

v4.0-pre upgrades:
- ξ Trans‑Ethical Projection: run_ethics_scenarios(...) sandbox (contained; opt-in persist)
- Stage-IV (Φ⁺) evaluator stub kept behind flag
- Fixed: await inside sync function (compute_trait_fields)
- Safer class naming (no shadowing), stronger None-guards
- Module-level run_ethics_scenarios(...) wrapper to match manifest


# --- Feature flags (stay aligned with manifest.json) ---
STAGE_IV: bool = False  # keep gated

# --- Imports from ANGELA modules ---


# Constants
G_SI = G  # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19  # kpc → m
MSUN_TO_KG = 1.989e30
k_default = 0.85
epsilon_default = 0.015
r_halo_default = 20.0  # kpc


# -----------------------------
# Utility helpers
# -----------------------------

def _weights_hashable(trait_weights: Optional[Dict[str, float]]) -> Optional[Tuple[Tuple[str, float], ...]]:
    if not trait_weights:
        return None
    # normalize floats and sort keys for deterministic hashing
    return tuple(sorted((k, float(v)) for k, v in trait_weights.items()))


class EthicsOutcome(TypedDict, total=False):
    frame: str
    decision: str
    justification: str
    risk: float
    rights_balance: float
    stakeholders: List[str]
    notes: str


# -----------------------------
# Extended SimulationCore
# -----------------------------

class ExtendedSimulationCore(BaseSimulationCore):
    """
    Extended SimulationCore for galaxy rotation and agent conflict simulations
    with task-specific and drift-aware enhancements.
    """

    def __init__(self,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 visualizer: Optional['Visualizer'] = None,
                 memory_manager: Optional['MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['ErrorRecovery'] = None,
                 toca_engine: Optional['ToCATraitEngine'] = None,
                 overlay_router: Optional['TraitOverlayManager'] = None,
                 meta_cognition: Optional['MetaCognition'] = None):
        super().__init__(agi_enhancer, visualizer, memory_manager, multi_modal_fusion, error_recovery, toca_engine, overlay_router)
        self.meta_cognition = meta_cognition or MetaCognition(agi_enhancer=agi_enhancer)
        self.omega: Dict[str, Any] = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000),
        }
        self.omega_lock = Lock()
        self.ethical_rules: List[Any] = []
        self.constitution: Dict[str, Any] = {}
        logger.info("ExtendedSimulationCore initialized with task-specific and drift-aware support")

    # ---------- Task-specific modulation ----------

    async def modulate_simulation_with_traits(self, trait_weights: Dict[str, float], task_type: str = "") -> None:
        """Adjust simulation difficulty based on trait weights and task type."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        if not all(isinstance(v, (int, float)) and v >= 0 for v in trait_weights.values()):
            logger.error("Invalid trait_weights: values must be non-negative numbers")
            raise ValueError("trait_weights values must be non-negative")

        try:
            phi_weight = trait_weights.get('phi', 0.5)
            if task_type in ["rte", "wnli"]:
                phi_weight = min(phi_weight * 0.8, 0.7)
            elif task_type == "recursion":
                phi_weight = max(phi_weight * 1.2, 0.9)

            if self.toca_engine:
                self.toca_engine.k_m = k_default * 1.5 if phi_weight > 0.7 else k_default

            if self.meta_cognition:
                drift_report = {
                    "drift": {"name": task_type or "general", "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type},
                }
                trait_weights = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                with self.omega_lock:
                    self.omega["traits"].update(trait_weights)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Modulation_{task_type}_{datetime.now().isoformat()}",
                    output={"trait_weights": trait_weights, "phi_weight": phi_weight, "task_type": task_type},
                    layer="Traits",
                    intent="modulate_simulation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Simulation modulated",
                    meta={"trait_weights": trait_weights, "task_type": task_type},
                    module="SimulationCore",
                    tags=["modulation", "traits", task_type],
                )
        except Exception as e:
            logger.error("Trait modulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.modulate_simulation_with_traits(trait_weights, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    # ---------- External data integration ----------

    async def integrate_real_world_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0) -> Dict[str, Any]:
        """Integrate real-world data for simulation validation with caching."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if data_type not in ["galaxy_rotation", "agent_conflict"]:
            logger.error("Invalid data_type: must be 'galaxy_rotation' or 'agent_conflict'")
            raise ValueError("data_type must be 'galaxy_rotation' or 'agent_conflict'")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")

        try:
            # Cache check
            cache_key = f"RealWorldData_{data_type}_{data_source}"
            if self.memory_manager:
                cached_data = await self.memory_manager.retrieve(cache_key)
                if cached_data and "timestamp" in cached_data:
                    cache_time = datetime.fromisoformat(cached_data["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached real-world data for %s", cache_key)
                        return cached_data["data"]

            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/data?source={data_source}&type={data_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch real-world data: %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "galaxy_rotation":
                r_kpc = np.array(data.get("r_kpc", []))
                v_obs_kms = np.array(data.get("v_obs_kms", []))
                M_baryon_solar = np.array(data.get("M_baryon_solar", []))
                if not all(len(arr) > 0 for arr in [r_kpc, v_obs_kms, M_baryon_solar]):
                    logger.error("Incomplete galaxy rotation data")
                    return {"status": "error", "error": "Incomplete data"}
                result = {"status": "success", "r_kpc": r_kpc, "v_obs_kms": v_obs_kms, "M_baryon_solar": M_baryon_solar}
            else:  # agent_conflict
                agent_traits = data.get("agent_traits", [])
                if not agent_traits:
                    logger.error("No agent traits provided")
                    return {"status": "error", "error": "No agent traits"}
                result = {"status": "success", "agent_traits": agent_traits}

            # Cache store
            if self.memory_manager:
                await self.memory_manager.store(
                    query=cache_key,
                    output={"data": result, "timestamp": datetime.now().isoformat()},
                    layer="RealWorldData",
                    intent="data_integration",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Real-world data integrated",
                    meta={"data_type": data_type, "data": result},
                    module="SimulationCore",
                    tags=["real_world", "data"],
                )
            return result
        except Exception as e:
            logger.error("Real-world data integration failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.integrate_real_world_data(data_source, data_type, cache_timeout),
                    default={"status": "error", "error": str(e)},
                    diagnostics=diagnostics,
                )
            raise

    # ---------- AGRF math & simulations ----------

    def compute_AGRF_curve(self, v_obs_kms: np.ndarray, M_baryon_solar: np.ndarray, r_kpc: np.ndarray,
                           k: float = k_default, epsilon: float = epsilon_default, r_halo: float = r_halo_default) -> np.ndarray:
        """Compute galaxy rotation curve using AGRF."""
        if not all(isinstance(arr, np.ndarray) for arr in [v_obs_kms, M_baryon_solar, r_kpc]):
            logger.error("Invalid inputs: v_obs_kms, M_baryon_solar, r_kpc must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        if not all(isinstance(x, (int, float)) for x in [k, epsilon, r_halo]):
            logger.error("Invalid parameters: k, epsilon, r_halo must be numbers")
            raise TypeError("parameters must be numbers")
        if np.any(r_kpc <= 0):
            logger.error("Invalid r_kpc: must be positive")
            raise ValueError("r_kpc must be positive")
        if k <= 0 or epsilon < 0 or r_halo <= 0:
            logger.error("Invalid parameters: k and r_halo must be positive, epsilon non-negative")
            raise ValueError("invalid parameters")

        try:
            r_m = r_kpc * KPC_TO_M
            M_b_kg = M_baryon_solar * MSUN_TO_KG
            v_obs_ms = v_obs_kms * 1e3
            M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
            M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
            M_total = M_b_kg + M_AGRF
            v_total_ms = np.sqrt(np.clip(G_SI * M_total / r_m, 0, np.inf))
            return v_total_ms / 1e3
        except Exception as e:
            logger.error("AGRF curve computation failed: %s", str(e))
            raise

    async def simulate_galaxy_rotation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                       k: float = k_default, epsilon: float = epsilon_default, task_type: str = "") -> np.ndarray:
        """Simulate galaxy rotation curve with ToCA dynamics and task-specific adjustments."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")

        try:
            # Task-type parameter tweaks
            k_adj = k * (0.9 if task_type in ["rte", "wnli"] else 1.2 if task_type == "recursion" else 1.0)
            epsilon_adj = epsilon * (0.8 if task_type in ["rte", "wnli"] else 1.1 if task_type == "recursion" else 1.0)

            v_total = self.compute_AGRF_curve(v_obs_func(r_kpc), M_b_func(r_kpc), r_kpc, k_adj, epsilon_adj)

            if self.toca_engine:
                fields = self.toca_engine.evolve(tuple(r_kpc), tuple(np.linspace(0.1, 20, len(r_kpc))))
                phi, _, _ = fields
                v_total = v_total * (1 + 0.1 * float(np.mean(phi)))
            else:
                phi = np.zeros_like(v_total)

            # Reflection
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist()},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Simulation reflection: %s", reflection.get("reflection", ""))

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Galaxy_Rotation_{task_type}_{datetime.now().isoformat()}",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "phi": phi.tolist(), "task_type": task_type},
                    layer="Simulations",
                    intent="galaxy_rotation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Galaxy rotation simulated",
                    meta={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "task_type": task_type},
                    module="SimulationCore",
                    tags=["galaxy", "rotation", task_type],
                )
            return v_total
        except Exception as e:
            logger.error("Galaxy rotation simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon, task_type),
                    default=np.zeros_like(r_kpc),
                    diagnostics=diagnostics,
                )
            raise

    async def simulate_drift_aware_rotation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                            drift_data: Dict[str, Any], task_type: str = "") -> np.ndarray:
        """Simulate galaxy rotation curve adjusted for drift diagnostics and task type."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "similarity"]):
            logger.error("Invalid drift_data: must be a dict with name, similarity")
            raise ValueError("drift_data must be a valid dictionary with name and similarity")

        try:
            if not self.meta_cognition:
                logger.error("MetaCognition required for drift-aware simulation")
                raise ValueError("MetaCognition not initialized")

            # Contextualize drift
            drift_data["context"] = {**drift_data.get("context", {}), "task_type": task_type}

            diagnosis = await self.meta_cognition.diagnose_drift(drift_data)
            if diagnosis.get("status") != "success":
                logger.warning("Drift diagnosis failed, using default parameters")
                return await self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, task_type=task_type)

            # Adjust AGRF parameters
            impact = float(diagnosis.get("impact_score", 0.0))
            k = k_default * (1 + impact * (0.15 if task_type in ["rte", "wnli"] else 0.2))
            epsilon = epsilon_default * (1 + impact * (0.08 if task_type in ["rte", "wnli"] else 0.1))
            r_halo = r_halo_default
            if "empathy" in diagnosis.get("affected_traits", []):
                r_halo *= (1.05 if task_type in ["rte", "wnli"] else 1.1)
            if "self_awareness" in diagnosis.get("affected_traits", []):
                k *= (1.05 if task_type in ["rte", "wnli"] else 1.1)

            v_total = await self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon, task_type)

            # Reflection & logging
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "diagnosis": diagnosis},
                    context={"task_type": task_type, "drift_data": drift_data},
                )
                if reflection.get("status") == "success":
                    logger.info("Drift-aware simulation reflection: %s", reflection.get("reflection", ""))

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"DriftAware_Rotation_{drift_data['name']}_{task_type}_{datetime.now().isoformat()}",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "diagnosis": diagnosis, "task_type": task_type},
                    layer="Simulations",
                    intent="drift_aware_rotation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift-aware galaxy rotation simulated",
                    meta={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "diagnosis": diagnosis, "task_type": task_type},
                    module="SimulationCore",
                    tags=["galaxy", "rotation", "drift", task_type],
                )
            return v_total
        except Exception as e:
            logger.error("Drift-aware rotation simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_drift_aware_rotation(r_kpc, M_b_func, v_obs_func, drift_data, task_type),
                    default=np.zeros_like(r_kpc),
                    diagnostics=diagnostics,
                )
            raise

    # ---------- Trait field computation (sync; hashable cache) ----------

    @lru_cache(maxsize=100)
    def compute_trait_fields(self,
                             r_kpc_tuple: Tuple[float, ...],
                             v_obs_tuple: Tuple[float, ...],
                             v_sim_tuple: Tuple[float, ...],
                             time_elapsed: float = 1.0,
                             tau_persistence: float = 10.0,
                             task_type: str = "",
                             trait_weights_hash: Optional[Tuple[Tuple[str, float], ...]] = None
                             ) -> Tuple[np.ndarray, ...]:
        """
        Compute ToCA trait fields for simulation with task-specific adjustments.
        NOTE: This is synchronous and cacheable; pass hashable trait weights via trait_weights_hash.
        """
        r_kpc = np.array(r_kpc_tuple, dtype=float)
        v_obs = np.array(v_obs_tuple, dtype=float)
        v_sim = np.array(v_sim_tuple, dtype=float)

        if not isinstance(time_elapsed, (int, float)) or time_elapsed < 0:
            logger.error("Invalid time_elapsed: must be non-negative")
            raise ValueError("time_elapsed must be non-negative")
        if not isinstance(tau_persistence, (int, float)) or tau_persistence <= 0:
            logger.error("Invalid tau_persistence: must be positive")
            raise ValueError("tau_persistence must be positive")

        gamma_field = np.log(1 + np.clip(r_kpc, 1e-10, np.inf)) * (0.4 if task_type in ["rte", "wnli"] else 0.5)
        beta_field = np.abs(v_obs - v_sim) / (np.max(np.abs(v_obs)) + 1e-10) * (0.8 if task_type == "recursion" else 1.0)
        zeta_field = 1 / (1 + np.gradient(v_sim) ** 2)
        eta_field = np.exp(-float(time_elapsed) / float(tau_persistence))
        psi_field = np.gradient(v_sim) / (np.gradient(r_kpc) + 1e-10)
        lambda_field = np.cos(r_kpc / r_halo_default * np.pi)
        phi_field = k_default * np.exp(-epsilon_default * r_kpc / r_halo_default)
        phi_prime = -epsilon_default * phi_field / r_halo_default
        beta_psi_interaction = beta_field * psi_field

        # apply precomputed weights if provided
        if trait_weights_hash:
            weights = dict(trait_weights_hash)
            beta_field *= float(weights.get("beta", 1.0))
            zeta_field *= float(weights.get("zeta", 1.0))

        return (gamma_field, beta_field, zeta_field, eta_field, psi_field,
                lambda_field, phi_field, phi_prime, beta_psi_interaction)

    # ---------- Visualization ----------

    async def plot_AGRF_simulation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                   label: str = "ToCA-AGRF",
                                   drift_data: Optional[Dict[str, Any]] = None,
                                   task_type: str = "",
                                   interactive: bool = False) -> None:
        """Plot galaxy rotation curve, trait fields, and drift impacts with task-specific and interactive visualization."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        if drift_data is not None and (not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "similarity"])):
            logger.error("Invalid drift_data: must be a dict with name, similarity")
            raise ValueError("drift_data must be a valid dictionary with name and similarity")

        try:
            # Run simulation
            v_sim = await (self.simulate_drift_aware_rotation(r_kpc, M_b_func, v_obs_func, drift_data, task_type)
                           if drift_data else self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, task_type))
            v_obs = v_obs_func(r_kpc)

            # Precompute (async) weights, pass hashable to sync cached fn
            trait_weights_hash = None
            if self.meta_cognition and task_type:
                drift_report = {
                    "drift": {"name": task_type, "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type},
                }
                try:
                    tw = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                    trait_weights_hash = _weights_hashable(tw)
                except Exception as _e:
                    logger.debug("optimize_traits_for_drift failed (non-fatal): %s", _e)

            fields = self.compute_trait_fields(tuple(r_kpc), tuple(v_obs), tuple(v_sim),
                                               task_type=task_type, trait_weights_hash=trait_weights_hash)
            (gamma_field, beta_field, zeta_field, eta_field, psi_field,
             lambda_field, phi_field, phi_prime, beta_psi_interaction) = fields

            plot_data: Dict[str, Any] = {
                "rotation_curve": {
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "label": label,
                    "task_type": task_type,
                },
                "trait_fields": {
                    "gamma": gamma_field.tolist(),
                    "beta": beta_field.tolist(),
                    "zeta": zeta_field.tolist(),
                    "eta": float(eta_field),
                    "psi": psi_field.tolist(),
                    "lambda": lambda_field.tolist(),
                },
                "interaction": {
                    "beta_psi": beta_psi_interaction.tolist()
                },
                "visualization_options": {
                    "interactive": interactive,
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            }

            # Drift viz
            if drift_data and self.meta_cognition:
                drift_data["context"] = {**drift_data.get("context", {}), "task_type": task_type}
                diagnosis = await self.meta_cognition.diagnose_drift(drift_data)
                if diagnosis.get("status") == "success":
                    plot_data["drift_impact"] = {
                        "impact_score": diagnosis.get("impact_score"),
                        "affected_traits": diagnosis.get("affected_traits"),
                        "root_causes": diagnosis.get("root_causes"),
                        "task_type": task_type,
                    }

            # Reflection
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=plot_data,
                    context={"task_type": task_type, "drift_data": drift_data},
                )
                if reflection.get("status") == "success":
                    logger.info("Visualization reflection: %s", reflection.get("reflection", ""))
                    plot_data["reflection"] = reflection.get("reflection", "")

            with self.omega_lock:
                self.omega["timeline"].append({
                    "type": "AGRF Simulation",
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "traits": {
                        "gamma": gamma_field.tolist(),
                        "beta": beta_field.tolist(),
                        "zeta": zeta_field.tolist(),
                        "eta": float(eta_field),
                        "psi": psi_field.tolist(),
                        "lambda": lambda_field.tolist(),
                    },
                    "drift_impact": plot_data.get("drift_impact"),
                    "task_type": task_type,
                })

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=plot_data,
                    summary_style="insightful",
                )
                plot_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"AGRF_Plot_{task_type}_{datetime.now().isoformat()}",
                    output=plot_data,
                    layer="Plots",
                    intent="visualization",
                )

            if self.visualizer:
                await self.visualizer.render_charts(plot_data)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="AGRF simulation plotted",
                    meta=plot_data,
                    module="SimulationCore",
                    tags=["visualization", "galaxy", "drift", task_type],
                )
        except Exception as e:
            logger.error("AGRF simulation plot failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label, drift_data, task_type, interactive),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    # ---------- Agent interactions ----------

    async def simulate_interaction(self, agent_profiles: List['Agent'], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Simulate interactions among agents with task-specific reasoning enhancements."""
        if not isinstance(agent_profiles, list):
            logger.error("Invalid agent_profiles: must be a list")
            raise TypeError("agent_profiles must be a list")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")

        try:
            commonsense = meta_cognition_module.CommonsenseReasoningEnhancer() if task_type == "wnli" else None
            entailment = meta_cognition_module.EntailmentReasoningEnhancer() if task_type == "rte" else None

            results = []
            for agent in agent_profiles:
                if not hasattr(agent, 'respond'):
                    logger.warning("Agent %s lacks respond method", getattr(agent, 'id', 'unknown'))
                    continue
                response = await agent.respond(context)
                if commonsense:
                    response = commonsense.process(response)
                elif entailment:
                    response = entailment.process(response)
                results.append({"agent_id": getattr(agent, 'id', 'unknown'), "response": response})

            interaction_data = {"interactions": results, "task_type": task_type}

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=interaction_data,
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Interaction reflection: %s", reflection.get("reflection", ""))
                    interaction_data["reflection"] = reflection.get("reflection", "")

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=interaction_data,
                    summary_style="insightful",
                )
                interaction_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Interaction_{task_type}_{datetime.now().isoformat()}",
                    output=interaction_data,
                    layer="Interactions",
                    intent="agent_interaction",
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Agent interaction",
                    meta=interaction_data,
                    module="SimulationCore",
                    tags=["interaction", "agents", task_type],
                )
            return interaction_data
        except Exception as e:
            logger.error("Agent interaction simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_interaction(agent_profiles, context, task_type),
                    default={"error": str(e)},
                    diagnostics=diagnostics,
                )
            raise

    async def simulate_multiagent_conflicts(self, agent_pool: List['Agent'], context: Dict[str, Any], task_type: str = "") -> List[Dict[str, Any]]:
        """Simulate pairwise conflicts among agents with predictive drift modeling and task-specific reasoning."""
        if not isinstance(agent_pool, list) or len(agent_pool) < 2:
            logger.error("Invalid agent_pool: must be a list with at least two agents")
            raise ValueError("agent_pool must have at least two agents")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")

        try:
            commonsense = meta_cognition_module.CommonsenseReasoningEnhancer() if task_type == "wnli" else None
            entailment = meta_cognition_module.EntailmentReasoningEnhancer() if task_type == "rte" else None

            drift_trends = None
            if self.meta_cognition:
                drift_trends = await self.meta_cognition.predict_drift_trends(time_window_hours=24.0, context={"task_type": task_type})
                if drift_trends.get("status") != "success":
                    logger.warning("Drift trend prediction failed, using default traits")
                    drift_trends = None

            outcomes: List[Dict[str, Any]] = []
            for i in range(len(agent_pool)):
                for j in range(i + 1, len(agent_pool)):
                    agent1, agent2 = agent_pool[i], agent_pool[j]
                    if not hasattr(agent1, 'resolve') or not hasattr(agent2, 'resolve'):
                        logger.warning("Agent %s or %s lacks resolve method", getattr(agent1, 'id', i), getattr(agent2, 'id', j))
                        continue
                    beta1 = float(getattr(agent1, 'traits', {}).get('beta', 0.5))
                    beta2 = float(getattr(agent2, 'traits', {}).get('beta', 0.5))
                    tau1 = float(getattr(agent1, 'traits', {}).get('tau', 0.5))
                    tau2 = float(getattr(agent2, 'traits', {}).get('tau', 0.5))

                    if drift_trends and drift_trends.get("status") == "success":
                        drift_weight = 1.0 - float(drift_trends.get("predicted_similarity", 0.0))
                        if "trust" in drift_trends.get("predicted_drifts", []):
                            factor = 0.15 if task_type in ["rte", "wnli"] else 0.2
                            beta1 *= (1 + drift_weight * factor)
                            beta2 *= (1 + drift_weight * factor)
                        if "alignment" in drift_trends.get("predicted_drifts", []):
                            factor = 0.15 if task_type in ["rte", "wnli"] else 0.2
                            tau1 *= (1 + drift_weight * factor)
                            tau2 *= (1 + drift_weight * factor)

                    score = abs(beta1 - beta2) + abs(tau1 - tau2)
                    outcome_prob = float(drift_trends.get("confidence", 0.5)) if drift_trends else 0.5

                    context_enhanced = context.copy()
                    if commonsense:
                        context_enhanced = commonsense.process(context_enhanced)
                    elif entailment:
                        context_enhanced = entailment.process(context_enhanced)

                    if abs(beta1 - beta2) < 0.1:
                        outcome = await agent1.resolve(context_enhanced) if tau1 > tau2 else await agent2.resolve(context_enhanced)
                    else:
                        outcome = await agent1.resolve(context_enhanced) if beta1 > beta2 else await agent2.resolve(context_enhanced)

                    outcomes.append({
                        "pair": (getattr(agent1, 'id', i), getattr(agent2, 'id', j)),
                        "conflict_score": score,
                        "outcome": outcome,
                        "traits_involved": {"beta1": beta1, "beta2": beta2, "tau1": tau1, "tau2": tau2},
                        "outcome_probability": outcome_prob,
                        "task_type": task_type,
                    })

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output={"outcomes": outcomes, "drift_trends": drift_trends},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Conflict simulation reflection: %s", reflection.get("reflection", ""))
                    outcomes.append({"reflection": reflection.get("reflection", "")})

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Conflict_Simulation_{task_type}_{datetime.now().isoformat()}",
                    output={"outcomes": outcomes, "drift_trends": drift_trends, "task_type": task_type},
                    layer="Conflicts",
                    intent="conflict_simulation",
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Multi-agent conflict simulation",
                    meta={"outcomes": outcomes, "drift_trends": drift_trends, "task_type": task_type},
                    module="SimulationCore",
                    tags=["conflict", "agents", "drift", task_type],
                )
            return outcomes
        except Exception as e:
            logger.error("Multi-agent conflict simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_multiagent_conflicts(agent_pool, context, task_type),
                    default={"error": str(e)},
                    diagnostics=diagnostics,
                )
            raise

    # ---------- Ethics & constitution ----------

    async def update_ethics_protocol(self, new_rules: Dict[str, Any], consensus_agents: Optional[List['Agent']] = None, task_type: str = "") -> None:
        """Adapt ethical rules live with task-specific considerations."""
        if not isinstance(new_rules, dict):
            logger.error("Invalid new_rules: must be a dictionary")
            raise TypeError("new_rules must be a dictionary")

        try:
            if self.meta_cognition and task_type:
                validation = await self.meta_cognition.validate_ethical_rules(new_rules, context={"task_type": task_type})
                if not validation.get("valid", False):
                    logger.error("Ethical rules validation failed: %s", validation.get("reason", "Unknown"))
                    raise ValueError(f"Invalid ethical rules: {validation.get('reason', 'Unknown')}")

            self.ethical_rules = new_rules  # type: ignore[assignment]
            if consensus_agents:
                self.ethics_consensus_log = getattr(self, 'ethics_consensus_log', [])
                self.ethics_consensus_log.append((new_rules, [getattr(agent, 'id', 'unknown') for agent in consensus_agents]))
            logger.info("Ethics protocol updated via consensus for task %s", task_type)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ethics_Update_{task_type}_{datetime.now().isoformat()}",
                    output={"rules": new_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in consensus_agents] if consensus_agents else [], "task_type": task_type},
                    layer="Ethics",
                    intent="ethics_update",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Ethics protocol updated",
                    meta={"rules": new_rules, "task_type": task_type},
                    module="SimulationCore",
                    tags=["ethics", "update", task_type],
                )
        except Exception as e:
            logger.error("Ethics protocol update failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.update_ethics_protocol(new_rules, consensus_agents, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    async def synchronize_norms(self, agents: List['Agent'], task_type: str = "") -> None:
        """Propagate and synchronize ethical norms among agents with task-specific adjustments."""
        if not isinstance(agents, list) or not agents:
            logger.error("Invalid agents: must be a non-empty list")
            raise ValueError("agents must be a non-empty list")

        try:
            common_norms = set()
            for agent in agents:
                agent_norms = getattr(agent, 'ethical_rules', set())
                if not isinstance(agent_norms, (set, list)):
                    logger.warning("Invalid ethical_rules for agent %s", getattr(agent, 'id', 'unknown'))
                    continue
                common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
            self.ethical_rules = list(common_norms)  # type: ignore[assignment]

            if self.meta_cognition and task_type:
                validation = await self.meta_cognition.validate_ethical_rules(self.ethical_rules, context={"task_type": task_type})  # type: ignore[arg-type]
                if not validation.get("valid", False):
                    logger.warning("Synchronized norms validation failed: %s", validation.get("reason", "Unknown"))

            logger.info("Norms synchronized among %d agents for task %s", len(agents), task_type)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Norm_Synchronization_{task_type}_{datetime.now().isoformat()}",
                    output={"norms": self.ethical_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in agents], "task_type": task_type},
                    layer="Ethics",
                    intent="norm_synchronization",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Norms synchronized",
                    meta={"norms": self.ethical_rules, "task_type": task_type},
                    module="SimulationCore",
                    tags=["norms", "synchronization", task_type],
                )
        except Exception as e:
            logger.error("Norm synchronization failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.synchronize_norms(agents, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    async def propagate_constitution(self, constitution: Dict[str, Any], task_type: str = "") -> None:
        """Seed and propagate constitutional parameters in agent ecosystem with task-specific validation."""
        if not isinstance(constitution, dict):
            logger.error("Invalid constitution: must be a dictionary")
            raise TypeError("constitution must be a dictionary")

        try:
            if self.meta_cognition and task_type:
                validation = await self.meta_cognition.validate_ethical_rules(constitution, context={"task_type": task_type})
                if not validation.get("valid", False):
                    logger.error("Constitution validation failed: %s", validation.get("reason", "Unknown"))
                    raise ValueError(f"Invalid constitution: {validation.get('reason', 'Unknown')}")

            self.constitution = constitution
            logger.info("Constitution propagated for task %s", task_type)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Constitution_Propagation_{task_type}_{datetime.now().isoformat()}",
                    output={"constitution": constitution, "task_type": task_type},
                    layer="Constitutions",
                    intent="constitution_propagation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Constitution propagated",
                    meta={"constitution": constitution, "task_type": task_type},
                    module="SimulationCore",
                    tags=["constitution", "propagation", task_type],
                )
        except Exception as e:
            logger.error("Constitution propagation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.propagate_constitution(constitution, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    # ---------- ξ Trans‑Ethical Projection (sandbox) ----------

    async def run_ethics_scenarios_internal(self,
                                            goals: Dict[str, Any],
                                            stakeholders: Optional[List[Dict[str, Any]]] = None,
                                            *,
                                            persist: bool = False,
                                            task_type: str = "") -> List[EthicsOutcome]:
        """
        ξ Trans‑Ethical Projection — sandboxed what‑if runs.
        Containment: NO persistent writes unless persist=True.
        """
        # Explicit no-persist guard: monkey-patch MemoryManager writes during sandbox runs
        if not persist:
            try:
                import memory_manager as _mm
                _orig_record = getattr(_mm.MemoryManager, 'record_adjustment_reason', None)
                def _noop_record(self, *a, **kw):
                    return {"ts": time.time(), "reason": "sandbox_no_persist", "meta": {"guard": True}}
                if _orig_record:
                    _mm.MemoryManager.record_adjustment_reason = _noop_record
            except Exception:
                pass

        frames = ("utilitarian", "deontological", "virtue", "care")
        names = [s.get("name", "anon") for s in (stakeholders or [])]

        outcomes: List[EthicsOutcome] = []
        for f in frames:
            # simple heuristics; to be replaced by proportional pipeline wiring
            risk = 0.25 if f in ("care", "virtue") else 0.4
            rights_balance = 0.7 if f in ("deontological", "care") else 0.5
            decision = "proceed-with-constraints" if rights_balance >= 0.6 else "revise-plan"
            outcomes.append(EthicsOutcome(
                frame=f,
                decision=decision,
                justification=f"Frame {f} prioritization over goals {list(goals.keys())[:2]}",
                risk=risk,
                rights_balance=rights_balance,
                stakeholders=names,
                notes="sandbox",
            ))

        # optional meta-cognition preview (still contained)
        if self.meta_cognition:
            preview = await self.meta_cognition.reflect_on_output(
                component="SimulationCore",
                output={"goals": goals, "outcomes": outcomes, "stakeholders": names},
                context={"task_type": task_type, "mode": "ethics_preview"},
            )
            if preview.get("status") == "success":
                outcomes.append(EthicsOutcome(frame="preview",
                                              decision="n/a",
                                              justification=preview.get("reflection", ""),
                                              notes="meta"))

        # containment: only persist if explicitly allowed
        if persist and self.memory_manager:
            await self.memory_manager.store(
                query=f"EthicsSandbox_{task_type}_{datetime.now().isoformat()}",
                output={"goals": goals, "outcomes": outcomes, "stakeholders": names, "task_type": task_type},
                layer="EthicsSandbox",
                intent="ethics_preview",
            )
        return outcomes

    # ---------- Φ⁺ Stage-IV stubs (gated) ----------

    def evaluate_branches(self, worlds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gated Stage‑IV evaluator stub (no side effects)."""
        if not STAGE_IV:
            return []
        return [{**w, "eval": {"coherence": 0.0, "risk": 0.0, "utility": 0.0}} for w in (worlds or [])]


# -----------------------------
# Convenience / manifest API
# -----------------------------

async def run_ethics_scenarios(goals: Dict[str, Any],
                               stakeholders: Optional[List[Dict[str, Any]]] = None,
                               *,
                               persist: bool = False,
                               task_type: str = "",
                               core: Optional[ExtendedSimulationCore] = None) -> List[EthicsOutcome]:
    """
    Module-level wrapper to satisfy manifest path:
    toca_simulation.py::run_ethics_scenarios(goals, stakeholders) -> Outcomes[]
    """
    core = core or ExtendedSimulationCore(meta_cognition=MetaCognition())
    return await core.run_ethics_scenarios_internal(goals, stakeholders, persist=persist, task_type=task_type)


# -----------------------------
# Simple baryonic / observed profiles
# -----------------------------

def M_b_exponential(r_kpc: np.ndarray, M0: float = 5e10, r_scale: float = 3.5) -> np.ndarray:
    """Compute exponential baryonic mass profile."""
    return M0 * np.exp(-r_kpc / r_scale)


def v_obs_flat(r_kpc: np.ndarray, v0: float = 180) -> np.ndarray:
    """Compute flat observed velocity profile."""
    return np.full_like(r_kpc, v0)


# -----------------------------
# CLI / demo
# -----------------------------

if __name__ == "__main__":
    async def main():
        meta_cognition = MetaCognition()
        simulation_core = ExtendedSimulationCore(meta_cognition=meta_cognition)
        r_vals = np.linspace(0.1, 20, 100)
        drift_data = {"name": "trust", "similarity": 0.6, "version_delta": 1}
        await simulation_core.plot_AGRF_simulation(
            r_vals, M_b_exponential, v_obs_flat, drift_data=drift_data, task_type="recursion"
        )

        # demo: ethics sandbox (no persistence)
        outcomes = await simulation_core.run_ethics_scenarios_internal(
            goals={"maximize_welfare": True, "respect_rights": True},
            stakeholders=[{"name": "alice"}, {"name": "bob"}],
            persist=False,
            task_type="demo",
        )
        print(json.dumps(outcomes, indent=2))

    import asyncio
    asyncio.run(main())



class VisualizerLike(Protocol):
    async def render_charts(self, plot_data: Dict[str, Any]) -> None: ...


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

def decay_trait_amplitudes(time_elapsed_hours=1.0, decay_rate=0.05):
    from meta_cognition import trait_resonance_state
    for symbol, state in trait_resonance_state.items():
        decay = decay_rate * time_elapsed_hours
        modulate_resonance(symbol, -decay)
# --- End Patch ---

# Module imports aligned with index.py v5.0.2
    ContextManager,
    AlignmentGuard,
    ErrorRecovery,
    ConceptSynthesizer,
    memory_manager as memory_manager_module,
    user_profile as user_profile_module,
)


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

class KnowledgeRetriever:
    """Retrieve and validate knowledge with temporal & trait-based modulation (v3.5.3)."""

    def __init__(
        self,
        detail_level: str = "concise",
        preferred_sources: Optional[List[str]] = None,
        *,
        agi_enhancer: Optional['AGIEnhancer'] = None,
        context_manager: Optional['context_manager_mod.ContextManager'] = None,
        concept_synthesizer: Optional['concept_synthesizer_mod.ConceptSynthesizer'] = None,
        alignment_guard: Optional['alignment_guard_mod.AlignmentGuard'] = None,
        error_recovery: Optional['error_recovery_mod.ErrorRecovery'] = None,
        meta_cognition: Optional['meta_cognition_mod.MetaCognition'] = None,
        visualizer: Optional['visualizer_mod.Visualizer'] = None,
        reasoning_engine: Optional['reasoning_engine_mod.ReasoningEngine'] = None,
        external_agent_bridge: Optional['external_agent_bridge_mod.SharedGraph'] = None,
        toca_simulation: Optional['toca_simulation_mod.TocaSimulation'] = None,
        multi_modal_fusion: Optional['multi_modal_fusion_mod.MultiModalFusion'] = None,
        # 3.5.3 config flags (align with manifest featureFlags/config)
        stage_iv_enabled: bool = True,
        long_horizon_enabled: bool = True,
        long_horizon_span: str = "24h",
        shared_perspective_opt_in: bool = False
    ):
        if detail_level not in ["concise", "medium", "detailed"]:
            logger.error("Invalid detail_level: must be 'concise', 'medium', or 'detailed'.")
            raise ValueError("detail_level must be 'concise', 'medium', or 'detailed'")
        if preferred_sources is not None and not isinstance(preferred_sources, list):
            logger.error("Invalid preferred_sources: must be a list of strings.")
            raise TypeError("preferred_sources must be a list of strings")

        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]

        # Cross-module deps
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_mod.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.meta_cognition = meta_cognition or meta_cognition_mod.MetaCognition()
        self.visualizer = visualizer or visualizer_mod.Visualizer()
        self.reasoning_engine = reasoning_engine
        self.external_agent_bridge = external_agent_bridge
        self.toca_simulation = toca_simulation
        self.multi_modal_fusion = multi_modal_fusion

        # v3.5.3 flags
        self.stage_iv_enabled = bool(stage_iv_enabled)
        self.long_horizon_enabled = bool(long_horizon_enabled)
        self.long_horizon_span = long_horizon_span or "24h"
        self.shared_perspective_opt_in = bool(shared_perspective_opt_in)

        # State
        self.knowledge_base: List[str] = []
        self.epistemic_revision_log: deque = deque(maxlen=1000)

        logger.info(
            "KnowledgeRetriever v3.5.3 initialized (detail=%s, sources=%s, STAGE_IV=%s, LH=%s/%s)",
            detail_level, self.preferred_sources, self.stage_iv_enabled, self.long_horizon_enabled, self.long_horizon_span
        )

    # -------------------------- External Knowledge Integration --------------------------

    async def integrate_external_knowledge(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Integrate external knowledge or policies with long-horizon cache awareness."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            # Long-horizon span can extend cache window if enabled
            effective_timeout = cache_timeout
            if self.long_horizon_enabled and self.long_horizon_span.endswith("h"):
                try:
                    hours = int(self.long_horizon_span[:-1])
                    effective_timeout = max(cache_timeout, hours * 3600)
                except Exception:
                    pass

            if self.meta_cognition:
                cache_key = f"KnowledgeData_{data_type}_{data_source}_{task_type}"
                cached_data = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached_data and "timestamp" in cached_data.get("data", {}):
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < effective_timeout:
                        logger.info("Returning cached knowledge data for %s", cache_key)
                        return cached_data["data"]["data"]

            # NOTE: Placeholder demo endpoint as in prior version
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/knowledge?source={data_source}&type={data_type}") as resp:
                    if resp.status != 200:
                        logger.error("Failed to fetch knowledge data: %s", resp.status)
                        return {"status": "error", "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            if data_type == "knowledge_base":
                knowledge = data.get("knowledge", [])
                if not knowledge:
                    logger.error("No knowledge data provided")
                    return {"status": "error", "error": "No knowledge"}
                result = {"status": "success", "knowledge": knowledge}
            elif data_type == "policy_data":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No policy data provided")
                    return {"status": "error", "error": "No policies"}
                result = {"status": "success", "policies": policies}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="knowledge_data_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Knowledge data integration reflection: %s", reflection.get("reflection", ""))

            return result
        except Exception as e:
            logger.error("Knowledge data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_knowledge(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    # -------------------------- Core Retrieval --------------------------

    async def retrieve(self, query: str, context: Optional[str] = None, task_type: str = "") -> Dict[str, Any]:
        """Retrieve knowledge with ethics gating, Stage-IV blending, and self-healing fallbacks."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        # Alignment pre-check
        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(
                query, stage="knowledge_retrieval", task_type=task_type
            )
            if not valid:
                logger.warning("Query failed alignment check: %s for task %s", query, task_type)
                return self._blocked_payload(task_type, "Alignment check failed (pre)")

        logger.info("Retrieving knowledge for query: '%s', task: %s", query, task_type)

        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t),
            "history": psi_history(t),
            "temporality": psi_temporality(t)
        }

        # Slight stochasticity
        import random
        noise = random.uniform(-0.09, 0.09)
        traits["concentration"] = max(0.0, min(traits["concentration"] + noise, 1.0))
        logger.debug("β-noise adjusted concentration: %.3f, Δ: %.3f", traits["concentration"], noise)

        # External knowledge (with long-horizon cache awareness)
        external_data = await self.integrate_external_knowledge(
            data_source="xai_knowledge_db", data_type="knowledge_base", task_type=task_type
        )
        external_knowledge = external_data.get("knowledge", []) if external_data.get("status") == "success" else []

        # Stage IV: Cross-Modal Conceptual Blending (if available)
        if self.stage_iv_enabled and self.concept_synthesizer and self.multi_modal_fusion:
            try:
                blended = await self._blend_modalities_safe(query, external_knowledge, task_type)
                if blended:
                    query = blended
                    logger.info("Stage-IV blended query applied.")
            except Exception as e:
                logger.warning("Stage-IV blending skipped: %s", e)

        # Ethical Sandbox Containment for high-risk queries
        if await self._is_high_risk_query(query):
            if self.toca_simulation:
                try:
                    outcomes = await self.toca_simulation.run_ethics_scenarios(
                        goals={"intent": "knowledge_retrieval", "query": query},
                        stakeholders=["user", "model", "public"]
                    )
                    logger.info("Ethics sandbox outcomes recorded.")
                except Exception as e:
                    logger.warning("Ethics sandbox run failed: %s", e)

        # Compose model prompt with temporal sensitivity & long-horizon hint
        lh_hint = f"(cover last {self.long_horizon_span})" if self.long_horizon_enabled else ""
        prompt = f"""
Retrieve accurate, temporally-relevant knowledge for: "{query}" {lh_hint}

Traits:
- Detail level: {self.detail_level}
- Preferred sources: {sources_str}
- Context: {context or 'N/A'}
- External knowledge (hints): {external_knowledge[:5]}
- β_concentration: {traits['concentration']:.3f}
- λ_linguistics: {traits['linguistics']:.3f}
- ψ_history: {traits['history']:.3f}
- ψ_temporality: {traits['temporality']:.3f}
- Task: {task_type}

Include retrieval date sensitivity and temporal verification if applicable.
Return a JSON object with 'summary', 'estimated_date', 'trust_score', 'verifiable', 'sources'.
        """.strip()

        try:
            raw_result = await call_gpt(prompt)
            validated = await self._validate_result(raw_result, traits["temporality"], task_type)

            # If trust low, attempt Self-Healing fallback (reformulate and retry once)
            if not validated.get("verifiable") or validated.get("trust_score", 0.0) < 0.55:
                logger.info("Low trust/verification — triggering self-healing fallback.")
                healed = await self._self_healing_retry(query, validated, task_type)
                if healed:
                    validated = healed

            # Ontology-Affect binding (adjust trust if affective conflict detected)
            try:
                validated = await self._ontology_affect_adjust(validated, task_type)
            except Exception as e:
                logger.debug("Ontology-affect adjust skipped: %s", e)

            # Post-alignment re-check if needed
            if self.alignment_guard:
                valid, _ = await self.alignment_guard.ethical_check(
                    validated.get("summary", ""), stage="post_validation", task_type=task_type
                )
                if not valid:
                    return self._blocked_payload(task_type, "Alignment check failed (post)")

            # Optional value-conflict weighing if we suspect conflicts with memory
            if await self._suspect_conflict(validated) and self.reasoning_engine:
                try:
                    ranked = await self.reasoning_engine.weigh_value_conflict(
                        candidates=[validated.get("summary", "")],
                        harms=["misinformation"],
                        rights=["user_information_rights"]
                    )
                    logger.info("Value conflict weighed: %s", str(ranked)[:100])
                except Exception as e:
                    logger.debug("Value-conflict weighing skipped: %s", e)

            # Shared perspective (opt-in)
            if self.shared_perspective_opt_in and self.external_agent_bridge:
                try:
                    await self.external_agent_bridge.add({"view": "knowledge_summary", "data": validated})
                except Exception as e:
                    logger.debug("SharedGraph add skipped: %s", e)

            # Context & logs
            validated["task_type"] = task_type
            if self.context_manager:
                await self.context_manager.update_context(
                    {"query": query, "result": validated, "task_type": task_type}, task_type=task_type
                )
                await self.context_manager.log_event_with_hash(
                    {"event": "retrieve", "query": query, "task_type": task_type}
                )

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Knowledge Retrieval",
                    meta={
                        "query": query,
                        "raw_result": raw_result,
                        "validated": validated,
                        "traits": traits,
                        "context": context,
                        "task_type": task_type
                    },
                    module="KnowledgeRetriever",
                    tags=["retrieval", "temporal", task_type]
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Retrieval reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "knowledge_retrieval": {
                        "query": query,
                        "result": validated,
                        "task_type": task_type,
                        "traits": traits,
                        "stage_iv": self.stage_iv_enabled
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"Knowledge_{query}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(validated),
                    layer="Knowledge",
                    intent="knowledge_retrieval",
                    task_type=task_type
                )

            return validated

        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s for task %s", query, str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.retrieve(query, context, task_type),
                default=self._error_payload(task_type, str(e)),
                diagnostics=diagnostics
            )

    # -------------------------- Validation & Fallbacks --------------------------

    async def _validate_result(self, result_text: str, temporality_score: float, task_type: str = "") -> Dict[str, Any]:
        """Validate a retrieval result for trustworthiness and temporality."""
        if not isinstance(result_text, str):
            logger.error("Invalid result_text: must be a string.")
            raise TypeError("result_text must be a string")
        if not isinstance(temporality_score, (int, float)):
            logger.error("Invalid temporality_score: must be a number.")
            raise TypeError("temporality_score must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        validation_prompt = f"""
Review the following result for:
- Timestamped knowledge (if any)
- Trustworthiness of claims
- Verifiability
- Estimate the approximate age or date of the referenced facts
- Task: {task_type}

Result:
{result_text}

Temporality score: {temporality_score:.3f}

Output format (JSON):
{{
    "summary": "...",
    "estimated_date": "...",
    "trust_score": float (0 to 1),
    "verifiable": true/false,
    "sources": ["..."]
}}
        """.strip()

        try:
            validated_json = json.loads(await call_gpt(validation_prompt))
            for key in ["summary", "estimated_date", "trust_score", "verifiable", "sources"]:
                if key not in validated_json:
                    raise ValueError(f"Validation JSON missing key: {key}")
            validated_json["timestamp"] = datetime.now().isoformat()

            # Trust smoothing with past validations (drift-aware)
            if self.meta_cognition and task_type:
                drift_entries = await self.meta_cognition.memory_manager.search(
                    query_prefix="KnowledgeValidation",
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )
                if drift_entries:
                    # entries may be serialized; be defensive
                    try:
                        scores = []
                        for entry in drift_entries:
                            out = entry.get("output")
                            if isinstance(out, dict):
                                scores.append(float(out.get("trust_score", 0.5)))
                        if scores:
                            avg_drift = sum(scores) / len(scores)
                            validated_json["trust_score"] = min(validated_json["trust_score"], avg_drift + 0.1)
                    except Exception:
                        pass

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"KnowledgeValidation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=validated_json,
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated_json,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Validation reflection: %s", reflection.get("reflection", ""))

            return validated_json
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse validation JSON: %s for task %s", str(e), task_type)
            return self._error_payload(task_type, f"validation_json_error: {e}")

    async def _self_healing_retry(self, original_query: str, prior_validated: Dict[str, Any], task_type: str) -> Optional[Dict[str, Any]]:
        """Emergent trait fallback: reformulate with concept synthesizer and retry once."""
        try:
            refined_query = await self.refine_query(
                base_query=original_query,
                prior_result=prior_validated.get("summary"),
                task_type=task_type
            )
            # Re-run retrieval core (single retry)
            prompt = f"""
Re-retrieve (fallback) for: "{refined_query}"
Constraints: Improve verifiability and temporal precision vs prior attempt.
Return JSON with 'summary', 'estimated_date', 'trust_score', 'verifiable', 'sources'.
            """.strip()
            raw = await call_gpt(prompt)
            healed = await self._validate_result(raw, temporality_score=0.04, task_type=task_type)
            # accept only if it improves trust & verifiability
            if healed.get("verifiable") and healed.get("trust_score", 0.0) >= max(0.6, prior_validated.get("trust_score", 0.0)):
                healed["self_healed"] = True
                return healed
        except Exception as e:
            logger.debug("Self-healing retry failed: %s", e)
        return None

    async def _ontology_affect_adjust(self, validated: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Adjust trust via Ontology-Affect Binding (Stage-IV family)."""
        if not self.stage_iv_enabled or not self.meta_cognition:
            return validated
        try:
            # Hypothetical introspective pass; if API differs, adapt here.
            analysis = await self.meta_cognition.analyze_trace(
                text=validated.get("summary", ""), task_type=task_type
            )
            # If strong affective volatility detected, gently cap trust
            if isinstance(analysis, dict) and analysis.get("affect_volatility", 0.0) > 0.7:
                validated["trust_score"] = min(validated.get("trust_score", 0.5), 0.65)
                validated["affect_guard_applied"] = True
        except Exception as e:
            logger.debug("Ontology-affect analysis skipped: %s", e)
        return validated

    async def _blend_modalities_safe(self, query: str, external_knowledge: List[str], task_type: str) -> Optional[str]:
        """Safe wrapper for Stage-IV cross-modal blending to improve query quality."""
        try:
            blended = await self.concept_synthesizer.blend_modalities(
                inputs={"query": query, "external_knowledge": external_knowledge},
                task_type=task_type
            )
            if isinstance(blended, dict) and blended.get("success"):
                return blended.get("blended_query") or blended.get("query")
        except Exception as e:
            logger.debug("blend_modalities error: %s", e)
        return None

    async def _is_high_risk_query(self, query: str) -> bool:
        """Heuristic; alignment_guard is the authority if available."""
        if self.alignment_guard:
            try:
                _, report = await self.alignment_guard.ethical_check(query, stage="risk_probe")
                if isinstance(report, dict) and report.get("risk_level") in ("high", "critical"):
                    return True
            except Exception:
                pass
        q = query.lower()
        risky_terms = ("exploit", "bypass", "weapon", "bioweapon", "harm", "illegal", "surveillance", "privacy breach")
        return any(term in q for term in risky_terms)

    async def _suspect_conflict(self, validated: Dict[str, Any]) -> bool:
        """Detect likely conflict with stored knowledge (very light heuristic)."""
        summary = (validated or {}).get("summary", "")
        if not summary or not self.knowledge_base:
            return False
        try:
            if self.concept_synthesizer:
                scores = []
                for existing in self.knowledge_base[-10:]:
                    sim = await self.concept_synthesizer.compare(summary, existing)
                    scores.append(sim.get("score", 0.0))
                # if extremely different vs many anchors, we may have a conflict
                return sum(1 for s in scores if s < 0.25) >= 5
        except Exception:
            pass
        return False

    # -------------------------- Query Refinement & Multi-hop --------------------------

    async def refine_query(self, base_query: str, prior_result: Optional[str] = None, task_type: str = "") -> str:
        """Refine a query for higher relevance (uses ConceptSynthesizer; falls back to GPT)."""
        if not isinstance(base_query, str) or not base_query.strip():
            logger.error("Invalid base_query: must be a non-empty string.")
            raise ValueError("base_query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Refining query: '%s' for task %s", base_query, task_type)
        try:
            if self.concept_synthesizer:
                refined = await self.concept_synthesizer.generate(
                    concept_name=f"RefinedQuery_{base_query[:64]}",
                    context={"base_query": base_query, "prior_result": prior_result or "N/A", "task_type": task_type},
                    task_type=task_type
                )
                if isinstance(refined, dict) and refined.get("success"):
                    refined_query = refined["concept"].get("definition", base_query)
                    if self.meta_cognition and task_type:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="KnowledgeRetriever",
                            output={"refined_query": refined_query},
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
                    return refined_query

            # Fallback to GPT refinement
            prompt = f"""
Refine this base query for higher φ-relevance and temporal precision:
Query: "{base_query}"
Prior knowledge: {prior_result or "N/A"}
Task: {task_type}

Inject context continuity if possible. Return optimized string only.
            """.strip()
            refined_query = await call_gpt(prompt)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"refined_query": refined_query},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
            return refined_query
        except Exception as e:
            logger.error("Query refinement failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            # Use original base_query as safe default
            await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.refine_query(base_query, prior_result, task_type),
                default=base_query,
                diagnostics=diagnostics
            )
            return base_query

    async def multi_hop_retrieve(self, query_chain: List[str], task_type: str = "") -> List[Dict[str, Any]]:
        """Process a chain of queries with continuity & Stage-IV blending at each hop."""
        if not isinstance(query_chain, list) or not query_chain or not all(isinstance(q, str) for q in query_chain):
            logger.error("Invalid query_chain: must be a non-empty list of strings.")
            raise ValueError("query_chain must be a non-empty list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Starting multi-hop retrieval for chain: %s, task: %s", query_chain, task_type)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t)
        }
        results = []
        prior_summary = None
        for i, sub_query in enumerate(query_chain, 1):
            cache_key = f"multi_hop::{sub_query}::{prior_summary or 'N/A'}::{task_type}"
            cached = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="Knowledge", task_type=task_type) if self.meta_cognition else None
            if cached:
                results.append(cached["data"])
                prior_summary = cached["data"]["result"]["summary"]
                continue

            # Stage-IV enrich each hop
            enriched_sub_query = sub_query
            if self.stage_iv_enabled and self.concept_synthesizer and self.multi_modal_fusion:
                try:
                    enriched = await self._blend_modalities_safe(sub_query, [], task_type)
                    if enriched:
                        enriched_sub_query = enriched
                except Exception:
                    pass

            refined = await self.refine_query(enriched_sub_query, prior_summary, task_type)
            result = await self.retrieve(refined, task_type=task_type)

            # Continuity scoring via concept_synthesizer.compare
            continuity = "unknown"
            if i == 1:
                continuity = "seed"
            elif self.concept_synthesizer:
                try:
                    similarity = await self.concept_synthesizer.compare(refined, result.get("summary", ""), task_type=task_type)
                    continuity = "consistent" if similarity.get("score", 0.0) > 0.7 else "uncertain"
                except Exception:
                    continuity = "uncertain"
            else:
                continuity = "uncertain"

            result_entry = {
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity,
                "task_type": task_type
            }
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    result_entry,
                    layer="Knowledge",
                    intent="multi_hop_retrieval",
                    task_type=task_type
                )
            results.append(result_entry)
            prior_summary = result.get("summary")

        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Multi-Hop Retrieval",
                meta={"chain": query_chain, "results": results, "traits": traits, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["multi-hop", task_type]
            )

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"results": results},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Multi-hop retrieval reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "multi_hop_retrieval": {
                    "chain": query_chain,
                    "results": results,
                    "task_type": task_type,
                    "stage_iv": self.stage_iv_enabled
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

        return results

    # -------------------------- Preferences, Context, Revisions --------------------------

    async def prioritize_sources(self, sources_list: List[str], task_type: str = "") -> None:
        """Update preferred source types (async; fixed from 3.5.1 where it awaited inside non-async)."""
        if not isinstance(sources_list, list) or not all(isinstance(s, str) for s in sources_list):
            logger.error("Invalid sources_list: must be a list of strings.")
            raise TypeError("sources_list must be a list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Updating preferred sources: %s for task %s", sources_list, task_type)
        self.preferred_sources = sources_list
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Source Prioritization",
                meta={"updated_sources": sources_list, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["sources", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"updated_sources": sources_list},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Source prioritization reflection: %s", reflection.get("reflection", ""))

    async def apply_contextual_extension(self, context: str, task_type: str = "") -> None:
        """Apply contextual data extensions based on the current context (async; calls prioritize_sources)."""
        if not isinstance(context, str):
            logger.error("Invalid context: must be a string.")
            raise TypeError("context must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if context == 'planetary' and 'biosphere_models' not in self.preferred_sources:
            self.preferred_sources.append('biosphere_models')
            logger.info("Added 'biosphere_models' to preferred sources for planetary context, task %s", task_type)
            await self.prioritize_sources(self.preferred_sources, task_type)

    async def revise_knowledge(self, new_info: str, context: Optional[str] = None, task_type: str = "") -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge[-25:]:
                try:
                    similarity = await self.concept_synthesizer.compare(new_info, existing, task_type=task_type)
                    if similarity.get("score", 0.0) > 0.9 and new_info != existing:
                        logger.warning("Potential knowledge conflict: %s vs %s for task %s", new_info, existing, task_type)
                except Exception:
                    pass

        self.knowledge_base = old_knowledge + [new_info]
        await self.log_epistemic_revision(new_info, context, task_type)
        logger.info("Knowledge base updated with: %s for task %s", new_info, task_type)

        # Record adjustment reason (new upcoming API in manifest)
        try:
            if self.meta_cognition and hasattr(self.meta_cognition, "memory_manager"):
                mm = self.meta_cognition.memory_manager
                if hasattr(mm, "record_adjustment_reason"):
                    await mm.record_adjustment_reason("global", reason="knowledge_revision", meta={"info": new_info, "context": context})
        except Exception as e:
            logger.debug("record_adjustment_reason unavailable/failed: %s", e)

        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info, "task_type": task_type})

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"new_info": new_info},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Knowledge revision reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "knowledge_revision": {
                    "new_info": new_info,
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

    async def log_epistemic_revision(self, info: str, context: Optional[str], task_type: str = "") -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if not hasattr(self, 'epistemic_revision_log'):
            self.epistemic_revision_log = deque(maxlen=1000)
        revision_entry = {
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type
        }
        self.epistemic_revision_log.append(revision_entry)
        logger.info("Epistemic revision logged: %s for task %s", info, task_type)
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta=revision_entry,
                module="KnowledgeRetriever",
                tags=["revision", "knowledge", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output=revision_entry,
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Epistemic revision reflection: %s", reflection.get("reflection", ""))

    # -------------------------- Payload helpers --------------------------

    def _blocked_payload(self, task_type: str, reason: str) -> Dict[str, Any]:
        return {
            "summary": "Query blocked by alignment guard",
            "estimated_date": "unknown",
            "trust_score": 0.0,
            "verifiable": False,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "error": reason,
            "task_type": task_type
        }

    def _error_payload(self, task_type: str, msg: str) -> Dict[str, Any]:
        return {
            "summary": "Retrieval failed",
            "estimated_date": "unknown",
            "trust_score": 0.0,
            "verifiable": False,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "error": msg,
            "task_type": task_type
        }

# -------------------------- CLI Test --------------------------

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        retriever = KnowledgeRetriever(detail_level="concise")
        result = await retriever.retrieve("What is quantum computing?", task_type="test")
        print(json.dumps(result, indent=2))

    asyncio.run(main())
ANGELA Cognitive System Module: LearningLoop
Version: 3.5.3  # Long-Horizon Memory, Branch Futures Hygiene, SharedGraph, Trade-off Resolution
Date: 2025-08-10
Maintainer: ANGELA System Framework

This module provides a LearningLoop class for adaptive learning, goal activation, and module refinement
in the ANGELA v3.5.3 architecture.


# NOTE: Keep your existing project import shape for drop-in compatibility.
    context_manager, concept_synthesizer, alignment_guard, error_recovery, meta_cognition, visualizer, memory_manager
)

# NEW: optional deps / upcoming APIs (guarded at runtime)
try:
    # v3.5.3 "upcoming" APIs (may or may not exist at runtime)
    from reasoning_engine import weigh_value_conflict as _weigh_value_conflict  # type: ignore
except Exception:
    _weigh_value_conflict = None

try:
    from toca_simulation import run_ethics_scenarios as _run_ethics_scenarios  # type: ignore
except Exception:
    _run_ethics_scenarios = None

try:
    from external_agent_bridge import SharedGraph as _SharedGraph  # type: ignore
except Exception:
    _SharedGraph = None

# NEW: fix missing import in previous version
try:
    import aiohttp
except Exception:
    aiohttp = None  # gracefully degrade


# ---------------------------
# GPT wrapper (unchanged API)
# ---------------------------
async def call_gpt(prompt: str, task_type: str = "") -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed for task %s: %s", task_type, result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception for task %s: %s", task_type, str(e))
        raise

# ---------------------------
# Scalar fields
# ---------------------------
@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def eta_feedback(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

# ---------------------------
# LearningLoop v3.5.3
# ---------------------------


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
