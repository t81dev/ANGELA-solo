from typing import Protocol, Dict, Any

class HTTPClient(Protocol):
    async def get_json(self, url: str) -> Dict[str, Any]:
        """Return JSON from a GET request or raise on failure."""
