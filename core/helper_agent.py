
import logging
from typing import Any, Dict, List, Optional

from core.meta_cognition import MetaCognition

logger = logging.getLogger("ANGELA.HelperAgent")

class HelperAgent:
    """A helper agent for task execution and collaboration."""
    def __init__(
        self,
        name: str,
        task: str,
        context: Dict[str, Any],
        dynamic_modules: List[Dict[str, Any]],
        api_blueprints: List[Dict[str, Any]],
        meta_cognition: Optional["MetaCognition"] = None,
        task_type: str = "",
    ):
        if not isinstance(name, str): raise TypeError("name must be a string")
        if not isinstance(task, str): raise TypeError("task must be a string")
        if not isinstance(context, dict): raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        self.name = name
        self.task = task
        self.context = context
        self.dynamic_modules = dynamic_modules
        self.api_blueprints = api_blueprints
        self.meta = meta_cognition or MetaCognition()
        self.task_type = task_type
        logger.info("HelperAgent initialized: %s (%s)", name, task_type)

    async def execute(self, collaborators: Optional[List["HelperAgent"]] = None) -> Any:
        return await self.meta.execute(collaborators=collaborators, task=self.task, context=self.context, task_type=self.task_type)
