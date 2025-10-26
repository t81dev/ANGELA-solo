
import asyncio
import json
import logging
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np
from core.alignment_guard import AlignmentGuard
from core.context_manager import ContextManager
from core.error_recovery import ErrorRecovery
from core.external_agent_bridge import ExternalAgentBridge
from core.level5_extensions import Level5Extensions
from core.memory_manager import MemoryManager
from core.meta_cognition import MetaCognition
from core.multi_modal_fusion import MultiModalFusion
from core.visualizer import Visualizer
from filelock import FileLock
from toca_simulation import (M_b_exponential, generate_phi_field,
                           simulate_galaxy_rotation, v_obs_flat)

logger = logging.getLogger("ANGELA.ReasoningEngine")

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

from core.hook_registry import (
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
