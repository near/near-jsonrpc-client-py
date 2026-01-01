from pydantic import RootModel
from typing import Literal


class SyncCheckpoint(RootModel[Literal['genesis', 'earliest_available']]):
    pass

