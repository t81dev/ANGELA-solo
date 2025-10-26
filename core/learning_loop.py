
import asyncio
import json
import logging
import math
import time
from collections import deque
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import aiohttp
from core.alignment_guard import AlignmentGuard
from core.concept_synthesizer import ConceptSynthesizer
from core.context_manager import ContextManager
from core.error_recovery import ErrorRecovery
from core.external_agent_bridge import SharedGraph
from core.memory_manager import MemoryManager
from core.meta_cognition import MetaCognition
from core.reasoning_engine import weigh_value_conflict
from core.visualizer import Visualizer
from toca_simulation import run_ethics_scenarios, run_simulation
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.LearningLoop")

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
