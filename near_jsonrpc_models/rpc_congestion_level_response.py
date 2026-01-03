from pydantic import BaseModel


class RpcCongestionLevelResponse(BaseModel):
    congestion_level: float
