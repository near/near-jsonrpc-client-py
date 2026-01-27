"""NEAR Account Identifier.

This is a unique, syntactically valid, human-readable account identifier on the NEAR network.

[See the crate-level docs for information about validation.](index.html#account-id-rules)

Also see [Error kind precedence](AccountId#error-kind-precedence).

## Examples

```
use near_account_id::AccountId;

let alice: AccountId = "alice.near".parse().unwrap();

assert!("ƒelicia.near".parse::<AccountId>().is_err()); // (ƒ is not f)
```"""

from pydantic import BaseModel
from pydantic import RootModel


class AccountId(RootModel[str]):
    pass

