from pydantic import RootModel
from typing import Literal


class MethodResolveError(RootModel[Literal['MethodEmptyName', 'MethodNotFound', 'MethodInvalidSignature']]):
    pass

