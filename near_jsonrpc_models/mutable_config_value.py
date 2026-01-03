from pydantic import BaseModel
from pydantic import RootModel


class MutableConfigValue(RootModel[str]):
    pass

