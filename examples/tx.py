import asyncio
from near_jsonrpc_client import NearClientAsync, ClientError, RpcError, HttpError, RequestTimeoutError
from near_jsonrpc_models import RpcBlockRequest, BlockId, CryptoHash, BlockIdBlockHeight, BlockIdCryptoHash, \
    RpcBlockRequestBlockId, RpcBlockRequestFinality, RpcTransactionStatusRequest, \
    RpcTransactionStatusRequestOption2Option, AccountId


async def main():
    client = NearClientAsync(base_url="https://rpc.mainnet.near.org")

    try:
        params = RpcTransactionStatusRequest(
            RpcTransactionStatusRequestOption2Option(
                sender_account_id=AccountId("sweat-relayer.near").root,
                tx_hash=CryptoHash("B4PGu3RicwMrjhv4k4MGaUhnZrTqPrrRu5gH9jxtHH4J").root,
                wait_until='EXECUTED_OPTIMISTIC'
            )
        )

        block = await client.tx(params=params)
        print("Tx Result:", block)

    except RpcError as e:
        print(f"{e}: {e.error}")
    except RequestTimeoutError as e:
        print(f"{e}")
    except HttpError as e:
        print(f"{e}: status: {e.status_code}, body: {e.body}")
    except ClientError as e:
        print("Invalid response:", e)
    await client.close()

asyncio.run(main())
