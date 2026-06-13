"""Versions of the delegate action carried by `Action::DelegateV2`. New
versions add a variant here rather than a new `Action` variant. The variant
is part of the signed payload, so a signature can't be ambiguous across
versions."""

from near_jsonrpc_models.delegate_action_v2 import DelegateActionV2
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class VersionedDelegateActionPayloadV2(StrictBaseModel):
    V2: DelegateActionV2

class VersionedDelegateActionPayload(RootModel[Union[VersionedDelegateActionPayloadV2]]):
    pass

