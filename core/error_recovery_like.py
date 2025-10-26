from typing import Protocol, Optional, Callable, Awaitable, Any, Dict

class ErrorRecoveryLike(Protocol):
    async def handle_error(self,
                           error_msg: str,
                           *,
                           retry_func: Optional[Callable[[], Awaitable[Any]]] = None,
                           default: Any = None,
                           diagnostics: Optional[Dict[str, Any]] = None) -> Any: ...
