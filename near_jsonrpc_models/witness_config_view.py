"""Configuration specific to ChunkStateWitness."""

from pydantic import BaseModel
from pydantic import conint


class WitnessConfigView(BaseModel):
    # Maximum size of transactions contained inside ChunkStateWitness.
    # 
    # A witness contains transactions from both the previous chunk and the current one.
    # This parameter limits the sum of sizes of transactions from both of those chunks.
    combined_transactions_size_limit: conint(ge=0, le=4294967295) = None
    # Size limit for storage proof generated while executing receipts in a chunk.
    # After this limit is reached we defer execution of any new receipts.
    main_storage_proof_size_soft_limit: conint(ge=0, le=18446744073709551615) = None
    # Soft size limit of storage proof used to validate new transactions in ChunkStateWitness.
    new_transactions_validation_state_size_soft_limit: conint(ge=0, le=18446744073709551615) = None
