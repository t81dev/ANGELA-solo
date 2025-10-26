
import asyncio
import json
import logging
import math
import time
from datetime import datetime
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional, Protocol

from core.alignment_guard import AlignmentGuard
from core.concept_synthesizer import ConceptSynthesizer
from core.context_manager import ContextManager
from core.error_recovery import ErrorRecovery
from core.memory_manager import MemoryManager
from core.meta_cognition import MetaCognition
from core.multi_modal_fusion import MultiModalFusion
from core.reasoning_engine import ReasoningEngine
from core.simulation_core import SimulationCore
from toca_simulation import run_AGRF_with_traits

logger = logging.getLogger("ANGELA.RecursivePlanner")

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
