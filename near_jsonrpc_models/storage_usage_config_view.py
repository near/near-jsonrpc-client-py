"""Describes cost of storage per block"""

from pydantic import BaseModel
from pydantic import conint


class StorageUsageConfigView(BaseModel):
    # Number of bytes for an account record, including rounding up for account id.
    num_bytes_account: conint(ge=0, le=18446744073709551615) = None
    # Additional number of bytes for a k/v record
    num_extra_bytes_record: conint(ge=0, le=18446744073709551615) = None
