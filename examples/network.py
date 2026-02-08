import asyncio
from near_jsonrpc_client import NearClientAsync, RpcClientError, RpcError, RpcHttpError, RpcTimeoutError


async def main():
    client = NearClientAsync(rpc_urls="https://rpc.mainnet.near.org")

    try:
        status = await client.status()
        print("Node status:", status)

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
