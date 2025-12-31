from models.near_token import NearToken
from pydantic import BaseModel


class TransferAction(BaseModel):
    deposit: NearToken
