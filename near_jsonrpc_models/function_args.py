"""This type is used to mark function arguments.

NOTE: The main reason for this to exist (except the type-safety) is that the value is
transparently serialized and deserialized as a base64-encoded string when serde is used
(serde_json)."""

from pydantic import BaseModel
from pydantic import RootModel


class FunctionArgs(RootModel[str]):
    pass

