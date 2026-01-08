"""Different types of finality."""

from pydantic import RootModel
from typing import Literal


class Finality(RootModel[Literal['optimistic', 'near-final', 'final']]):
    pass

