"""This type is used to mark keys (arrays of bytes) that are queried from store.

NOTE: Currently, this type is only used in the view_client and RPC to be able to transparently
pretty-serialize the bytes arrays as base64-encoded strings (see `serialize.rs`)."""

from pydantic import BaseModel
from pydantic import RootModel


class StoreKey(RootModel[str]):
    pass

