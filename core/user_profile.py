
import json
import logging
import os
from collections import deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from core.alignment_guard import AlignmentGuard
from core.concept_synthesizer import ConceptSynthesizer
from core.error_recovery import ErrorRecovery
from core.knowledge_retriever import AGIEnhancer
from core.memory_manager import MemoryManager
from core.meta_cognition import MetaCognition
from core.multi_modal_fusion import MultiModalFusion
from core.reasoning_engine import ReasoningEngine
from core.simulation_core import SimulationCore
from core.toca_trait_engine import ToCATraitEngine

logger = logging.getLogger("ANGELA.UserProfile")

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
