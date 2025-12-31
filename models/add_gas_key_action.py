from models.access_key_permission import AccessKeyPermission
from models.public_key import PublicKey
from pydantic import BaseModel
from pydantic import conint


class AddGasKeyAction(BaseModel):
    num_nonces: conint(ge=0, le=4294967295)
    permission: AccessKeyPermission
    public_key: PublicKey
