from typing import Type
from uuid import uuid4

import httpx
from pydantic import BaseModel
from typing import get_args

from .errors import (
    ClientError,
    TransportError,
    HttpError,
    RpcError,
    RequestTimeoutError,
)
from .transport import HttpTransport


def _extract_method(request_model: type[BaseModel]) -> str:
    field = request_model.model_fields.get("method")
    if field is None:
        raise ClientError(
            f"{request_model.__name__} does not define a 'method' field"
        )

    args = get_args(field.annotation)
    if len(args) != 1 or not isinstance(args[0], str):
        raise ClientError(
            f"Invalid JSON-RPC method definition in {request_model.__name__}"
        )

    return args[0]


class NearBaseClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 10.0,
        headers: dict[str, str] | None = None,
    ):
        self._transport = HttpTransport(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
        )

    async def _call(
        self,
        *,
        request_model: Type[BaseModel],
        response_model: Type[BaseModel],
        params: BaseModel,
        debug=False
    ):

        request = request_model(
            jsonrpc="2.0",
            id=str(uuid4()),
            method=_extract_method(request_model),
            params=params,
        )

        payload = request.model_dump(by_alias=True)
        if debug:
            print("➡️ JSON-RPC Request payload:")
            print(payload)

        try:
            response = await self._transport.post(json=payload)

            if debug:
                print("⬅️ JSON-RPC Raw Response:")
                print(response.text)

        except httpx.TimeoutException as e:
            raise RequestTimeoutError() from e
        except httpx.RequestError as e:
            raise TransportError(str(e)) from e

        if 500 <= response.status_code < 600:
            raise HttpError(
                status_code=response.status_code,
                body=response.text,
            )

        try:
            parsed = response_model.model_validate(response.json())
        except Exception as e:
            raise ClientError("Invalid response format") from e

        inner = parsed.root

        if hasattr(inner, "error"):
            raise RpcError(inner)

        return inner.result

    async def close(self):
        await self._transport.close()
