"""Describes information about an access key including its on-trie
identifier. For ed25519/secp256k1 access keys the `public_key` field
is the full public key (string form unchanged from before); for
ML-DSA-65 access keys it is a `ml-dsa-65-hash:...` SHA3-384 digest
(the full pubkey is not stored on-chain)."""

from near_jsonrpc_models.access_key_view import AccessKeyView
from near_jsonrpc_models.key_handle import KeyHandle
from pydantic import BaseModel


class AccessKeyInfoView(BaseModel):
    access_key: AccessKeyView
    public_key: KeyHandle
