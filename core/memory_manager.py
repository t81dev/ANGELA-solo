
import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from core.alignment_guard import AlignmentGuard
from core.concept_synthesizer import ConceptSynthesizer
from core.context_manager import ContextManager
from core.drift_index import DriftIndex
from core.error_recovery import ErrorRecovery
from core.knowledge_retriever import KnowledgeRetriever
from core.meta_cognition import MetaCognition
from core.time_chain_mixin import delta_memory
from core.visualizer import Visualizer
from filelock import FileLock
from toca_simulation import ToCASimulation
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MemoryManager")

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
