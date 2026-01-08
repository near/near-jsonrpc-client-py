from pydantic import RootModel
from typing import Literal


class Direction(RootModel[Literal['Left', 'Right']]):
    pass

