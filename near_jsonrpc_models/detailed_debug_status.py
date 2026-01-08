from near_jsonrpc_models.block_status_view import BlockStatusView
from near_jsonrpc_models.catchup_status_view import CatchupStatusView
from near_jsonrpc_models.network_info_view import NetworkInfoView
from pydantic import BaseModel
from pydantic import conint
from typing import List


class DetailedDebugStatus(BaseModel):
    block_production_delay_millis: conint(ge=0, le=18446744073709551615)
    catchup_status: List[CatchupStatusView]
    current_head_status: BlockStatusView
    current_header_head_status: BlockStatusView
    network_info: NetworkInfoView
    sync_status: str
