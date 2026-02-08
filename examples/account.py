import asyncio

from near_jsonrpc_client import NearClientAsync, RpcClientError, RpcError, RpcHttpError, RpcTimeoutError
from near_jsonrpc_models import Finality, \
    AccountId, RpcQueryRequest, RpcQueryRequestViewAccountByFinality


async def main():
    client = NearClientAsync(rpc_urls="https://rpc.mainnet.near.org")

    try:
        params = RpcQueryRequest(
            RpcQueryRequestViewAccountByFinality(
                account_id=AccountId("neardome2340.near"),
                finality=Finality('near-final'),
                request_type='view_account'
            )
        )

        acc = await client.query(params=params)
        print("View Account Result:", acc)

    except RpcError as e:
        print(f"{e}: {e.error}")
    except RpcTimeoutError as e:
        print(f"{e}")
    except RpcHttpError as e:
        print(f"{e}: status: {e.status_code}, body: {e.body}")
    except RpcClientError as e:
        print("Invalid response:", e)
    finally:
        await client.close()


asyncio.run(main())
