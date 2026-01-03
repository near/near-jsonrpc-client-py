from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint


class NearGas(RootModel[conint(ge=0, le=18446744073709551615)]):
    pass

