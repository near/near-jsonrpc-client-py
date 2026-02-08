import asyncio

from near_jsonrpc_client import NearClientAsync, RpcClientError, RpcError, RpcHttpError, RpcTimeoutError
from near_jsonrpc_models import CryptoHash, RpcTransactionStatusRequest, AccountId, \
    RpcTransactionStatusRequestSenderAccountIdTxHash


async def main():
    client = NearClientAsync(rpc_urls="https://rpc.mainnet.near.org")

    try:
        params = RpcTransactionStatusRequest(
            RpcTransactionStatusRequestSenderAccountIdTxHash(
                sender_account_id=AccountId("sweat-relayer.near").root,
                tx_hash=CryptoHash("B4PGu3RicwMrjhv4k4MGaUhnZrTqPrrRu5gH9jxtHH4J").root,
                wait_until='EXECUTED_OPTIMISTIC'
            )
        )

        block = await client.tx(params=params)
        print("Tx Result:", block)

    except RpcError as e:
        print(f"{e}: {e.error}")
    except RpcTimeoutError as e:
        print(f"{e}")
    except RpcHttpError as e:
        print(f"{e}: status: {e.status_code}, body: {e.body}")
    except RpcClientError as e:
        print("Invalid response:", e)
    await client.close()

asyncio.run(main())
