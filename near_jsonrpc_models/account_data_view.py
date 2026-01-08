"""AccountData is a piece of global state that a validator
signs and broadcasts to the network.

It is essentially the data that a validator wants to share with the network.
All the nodes in the network are collecting the account data
broadcasted by the validators.
Since the number of the validators is bounded and their
identity is known (and the maximal size of allowed AccountData is bounded)
the global state that is distributed in the form of AccountData is bounded
as well.
Find more information in the docs [here](https://github.com/near/nearcore/blob/560f7fc8f4b3106e0d5d46050688610b1f104ac6/chain/client/src/client.rs#L2232)"""

from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.tier1proxy_view import Tier1ProxyView
from pydantic import BaseModel
from typing import List


class AccountDataView(BaseModel):
    # Account key of the validator signing this AccountData.
    account_key: PublicKey
    # ID of the node that handles the account key (aka validator key).
    peer_id: PublicKey
    # Proxy nodes that are directly connected to the validator node
    # (this list may include the validator node itself).
    # TIER1 nodes should connect to one of the proxies to sent TIER1
    # messages to the validator.
    proxies: List[Tier1ProxyView]
    # UTC timestamp of when the AccountData has been signed.
    timestamp: str
