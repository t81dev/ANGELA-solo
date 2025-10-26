
import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import aiohttp
import networkx as nx
from core.alignment_guard import AlignmentGuard
from core.concept_synthesizer import ConceptSynthesizer
from core.context_manager import ContextManager
from core.error_recovery import ErrorRecovery
from core.memory_manager import MemoryManager
from core.meta_cognition import MetaCognition
from core.reasoning_engine import ReasoningEngine
from core.visualizer import Visualizer
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MultiModalFusion")

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
