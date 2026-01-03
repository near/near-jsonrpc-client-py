from near_jsonrpc_models.external_storage_location import ExternalStorageLocation
from pydantic import BaseModel
from pydantic import conint


class ExternalStorageConfig(BaseModel):
    # The number of attempts the node will make to obtain a part from peers in
    # the network before it fetches from external storage.
    external_storage_fallback_threshold: conint(ge=0, le=18446744073709551615) = 3
    # Location of state parts.
    location: ExternalStorageLocation = None
    # When fetching state parts from external storage, throttle fetch requests
    # to this many concurrent requests.
    num_concurrent_requests: conint(ge=0, le=255) = 25
    # During catchup, the node will use a different number of concurrent requests
    # to reduce the performance impact of state sync.
    num_concurrent_requests_during_catchup: conint(ge=0, le=255) = 5
