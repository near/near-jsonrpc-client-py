from models.block_id import BlockId
from models.finality import Finality
from models.strict_model import StrictBaseModel
from models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class BlockReferenceBlockId(StrictBaseModel):
    block_id: BlockId

class BlockReferenceFinality(StrictBaseModel):
    finality: Finality

class BlockReferenceSyncCheckpoint(StrictBaseModel):
    sync_checkpoint: SyncCheckpoint

class BlockReference(RootModel[Union[BlockReferenceBlockId, BlockReferenceFinality, BlockReferenceSyncCheckpoint]]):
    pass

