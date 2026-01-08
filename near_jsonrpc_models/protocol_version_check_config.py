"""Configures whether the node checks the next or the next next epoch for network version compatibility."""

from pydantic import RootModel
from typing import Literal


class ProtocolVersionCheckConfig(RootModel[Literal['Next', 'NextNext']]):
    pass

