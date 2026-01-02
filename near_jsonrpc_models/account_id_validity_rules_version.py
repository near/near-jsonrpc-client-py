from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint


class AccountIdValidityRulesVersion(RootModel[conint(ge=0, le=255)]):
    pass

