"""A versioned struct that contains all information needed to assign accounts to shards.

Because of re-sharding, the chain may use different shard layout to split shards at different
times. Currently, `ShardLayout` is stored as part of `EpochConfig`, which is generated each
epoch given the epoch protocol version. In mainnet/testnet, we use two shard layouts since
re-sharding has only happened once. It is stored as part of genesis config, see
default_simple_nightshade_shard_layout() Below is an overview for some important
functionalities of ShardLayout interface."""

from near_jsonrpc_models.shard_layout_v0 import ShardLayoutV0
from near_jsonrpc_models.shard_layout_v1 import ShardLayoutV1
from near_jsonrpc_models.shard_layout_v2 import ShardLayoutV2
from near_jsonrpc_models.shard_layout_v3 import ShardLayoutV3
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class ShardLayoutV0Option(StrictBaseModel):
    V0: ShardLayoutV0

class ShardLayoutV1Option(StrictBaseModel):
    V1: ShardLayoutV1

class ShardLayoutV2Option(StrictBaseModel):
    V2: ShardLayoutV2

class ShardLayoutV3Option(StrictBaseModel):
    V3: ShardLayoutV3

class ShardLayout(RootModel[Union[ShardLayoutV0Option, ShardLayoutV1Option, ShardLayoutV2Option, ShardLayoutV3Option]]):
    pass

