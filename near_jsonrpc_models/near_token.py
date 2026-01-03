from pydantic import BaseModel
from pydantic import RootModel


class NearToken(RootModel[str]):
    pass

