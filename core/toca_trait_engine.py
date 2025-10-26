
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from core.meta_cognition import MetaCognition

logger = logging.getLogger("ANGELA.ToCATraitEngine")

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
