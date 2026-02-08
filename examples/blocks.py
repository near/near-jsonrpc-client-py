import asyncio
from near_jsonrpc_client import NearClientAsync, RpcClientError, RpcError, RpcHttpError, RpcTimeoutError
from near_jsonrpc_models import RpcBlockRequest, BlockId, CryptoHash, BlockIdBlockHeight, BlockIdCryptoHash, \
    RpcBlockRequestBlockId, RpcBlockRequestFinality


async def main():
    client = NearClientAsync(rpc_urls="https://rpc.mainnet.near.org")

    try:
        params = RpcBlockRequest(
            RpcBlockRequestBlockId(
                block_id=BlockId(BlockIdBlockHeight(178682261))
            )
        )

        block = await client.block(params=params)
        print("Block Result:", block)

    except RpcError as e:
        print(f"{e}: {e.error}")
    except RpcTimeoutError as e:
        print(f"{e}")
    except RpcHttpError as e:
        print(f"{e}: status: {e.status_code}, body: {e.body}")
    except RpcClientError as e:
        print("Invalid response:", e)

    try:
        params = RpcBlockRequest(
            RpcBlockRequestBlockId(
                block_id=BlockId(BlockIdCryptoHash(CryptoHash("FL6JnFZSZvgRsn9s7qHM3SrC8VXXAfNGRMyMtBfrAiQC").root))
            )
        )

        block = await client.block(params=params)
        print("Block Result1:", block)

        params = RpcBlockRequest(RpcBlockRequestFinality(finality='final'))
        block = await client.block(params=params)
        print("Block Result2:", block)

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
