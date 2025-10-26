
import argparse
import asyncio
import json
import logging
import os
import time

from core.agi_enhancer import AGIEnhancer
from core.alignment_guard import AlignmentGuard
from core.code_executor import CodeExecutor
from core.concept_synthesizer import ConceptSynthesizer
from core.context_manager import ContextManager
from core.creative_thinker import CreativeThinker
from core.ecosystem_manager import EcosystemManager
from core.embodied_agent import EmbodiedAgent
from core.error_recovery import ErrorRecovery
from core.external_agent_bridge import ExternalAgentBridge
from core.knowledge_retriever import KnowledgeRetriever
from core.learning_loop import LearningLoop
from core.level5_extensions import SelfCloningLLM
from core.memory_manager import MemoryManager
from core.meta_cognition import MetaCognition
from core.multi_modal_fusion import MultiModalFusion
from core.recursive_planner import RecursivePlanner
from core.simulation_core import SimulationCore
from core.time_chain_mixin import (LONG_HORIZON_DEFAULT, TimeChainMixin,
                                   chi_culturevolution,
                                   construct_trait_view, eta_empathy,
                                   export_resonance_map, kappa_knowledge,
                                   lambda_linguistics, modulate_resonance,
                                   nu_narrative, omega_selfawareness,
                                   phi_scalar, pi_principles, psi_history,
                                   rho_agency, sigma_social, tau_timeperception,
                                   theta_causality, upsilon_utility,
                                   xi_cognition, zeta_consequence)
from core.user_profile import UserProfile
from core.visualizer import Visualizer
from toca_simulation import TocaSimulation

logger = logging.getLogger("ANGELA.HaloEmbodimentLayer")

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
