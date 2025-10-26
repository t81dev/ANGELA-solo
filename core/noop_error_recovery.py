from dataclasses import dataclass
from typing import Optional, Callable, Awaitable, Any, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class NoopErrorRecovery:
    async def handle_error(self, error_msg: str, *, retry_func: Optional[Callable[[], Awaitable[Any]]] = None, default: Any = None, diagnostics: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug("ErrorRecovery(noop): %s", error_msg)
        return default
