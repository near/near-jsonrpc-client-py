from pydantic import RootModel
from types import NoneType


class GenesisConfigError(RootModel[NoneType]):
    pass

