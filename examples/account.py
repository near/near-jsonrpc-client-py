import asyncio

from near_jsonrpc_client import NearClientAsync, ClientError, RpcError, HttpError, RequestTimeoutError
from near_jsonrpc_models import Finality, \
    AccountId, RpcQueryRequest, RpcQueryRequestViewAccountByFinality


async def main():
    client = NearClientAsync(base_url="https://rpc.mainnet.near.org")

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
    except RequestTimeoutError as e:
        print(f"{e}")
    except HttpError as e:
        print(f"{e}: status: {e.status_code}, body: {e.body}")
    except ClientError as e:
        print("Invalid response:", e)
    finally:
        await client.close()


asyncio.run(main())
