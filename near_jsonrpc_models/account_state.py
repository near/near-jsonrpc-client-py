"""Whether an account's state has been installed.

Only universal accounts can be uninitialized: they come into existence when
a transfer funds a `0u` id whose state init has not been applied yet. A
deterministic `0s` account waiting for its state init is an ordinary V1
account with no contract, not this."""

from pydantic import RootModel
from typing import Literal


class AccountState(RootModel[Literal['initialized', 'uninitialized']]):
    pass

