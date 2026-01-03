"""Epoch identifier -- wrapped hash, to make it easier to distinguish.
EpochId of epoch T is the hash of last block in T-2
EpochId of first two epochs is 0"""

from near_jsonrpc_models.crypto_hash import CryptoHash


EpochId = CryptoHash
