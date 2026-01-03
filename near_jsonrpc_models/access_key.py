"""Access key provides limited access to an account. Each access key belongs to some account and
is identified by a unique (within the account) public key. One account may have large number of
access keys. Access keys allow to act on behalf of the account by restricting transactions
that can be issued.
`account_id,public_key` is a key in the state"""

from near_jsonrpc_models.access_key_permission import AccessKeyPermission
from pydantic import BaseModel
from pydantic import conint


class AccessKey(BaseModel):
    # Nonce for this access key, used for tx nonce generation. When access key is created, nonce
    # is set to `(block_height - 1) * 1e6` to avoid tx hash collision on access key re-creation.
    # See <https://github.com/near/nearcore/issues/3779> for more details.
    nonce: conint(ge=0, le=18446744073709551615)
    # Defines permissions for this access key.
    permission: AccessKeyPermission
