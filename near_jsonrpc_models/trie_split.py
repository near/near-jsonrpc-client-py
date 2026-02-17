"""The result of splitting a memtrie into two possibly even parts, according to `memory_usage`
stored in the trie nodes.

**NOTE: This is an artificial value calculated according to `TRIE_COST`. Hence, it does not
represent actual memory allocation, but the split ratio should be roughly consistent with that.**"""

from near_jsonrpc_models.account_id import AccountId
from pydantic import BaseModel
from pydantic import conint


class TrieSplit(BaseModel):
    # Account ID representing the split path
    boundary_account: AccountId
    # Total `memory_usage` of the left part (excluding the split path)
    left_memory: conint(ge=0, le=18446744073709551615)
    # Total `memory_usage` of the right part (including the split path)
    right_memory: conint(ge=0, le=18446744073709551615)
