import httpx


class HttpTransport:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        headers: dict[str, str] | None = None,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
        )

    async def post(self, json: dict) -> httpx.Response:
        return await self._client.post("", json=json)

    async def close(self):
        await self._client.aclose()