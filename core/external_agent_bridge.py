
import asyncio
import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from networkx import DiGraph

from core.alignment_guard import AlignmentGuard
from core.code_executor import CodeExecutor
from core.context_manager import ContextManager
from core.helper_agent import HelperAgent
from core.memory_manager import MemoryManager, cache_state, retrieve_state
from core.meta_cognition import MetaCognition
from core.reasoning_engine import ReasoningEngine
from core.visualizer import Visualizer
from toca_simulation import run_simulation

logger = logging.getLogger("ANGELA.ExternalAgentBridge")

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
