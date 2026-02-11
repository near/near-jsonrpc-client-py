"""Reason why a gas key transaction failed at the deposit/account level.
In these cases, gas is still charged from the gas key."""

from pydantic import RootModel
from typing import Literal


class DepositCostFailureReason(RootModel[Literal['NotEnoughBalance', 'LackBalanceForState']]):
    pass

