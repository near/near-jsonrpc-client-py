"""NONE: Transaction is waiting to be included into the blockINCLUDED: Transaction is included into the block. The block may be not finalized yetEXECUTED_OPTIMISTIC: Transaction is included into the block +
All non-refund transaction receipts finished their execution.
The corresponding blocks for tx and each receipt may be not finalized yetINCLUDED_FINAL: Transaction is included into finalized blockEXECUTED: Transaction is included into finalized block +
All non-refund transaction receipts finished their execution.
The corresponding blocks for each receipt may be not finalized yetFINAL: Transaction is included into finalized block +
Execution of all transaction receipts is finalized, including refund receipts"""

from pydantic import RootModel
from typing import Literal


class TxExecutionStatus(RootModel[Literal['NONE', 'INCLUDED', 'EXECUTED_OPTIMISTIC', 'INCLUDED_FINAL', 'EXECUTED', 'FINAL']]):
    pass

