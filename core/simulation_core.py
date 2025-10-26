
import asyncio
import hashlib
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import numpy as np
from core.alignment_guard import AlignmentGuard
from core.error_recovery import ErrorRecovery
from core.memory_manager import MemoryManager
from core.meta_cognition import MetaCognition
from core.multi_modal_fusion import MultiModalFusion, SceneGraphT
from core.reasoning_engine import ReasoningEngine
from core.toca_trait_engine import ToCATraitEngine
from core.visualizer import Visualizer
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.SimulationCore")

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
