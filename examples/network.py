import asyncio
from client import NearClient, ClientError, RpcError, HttpError
from client.errors import RequestTimeoutError


async def main():
    client = NearClient(base_url="https://rpc.mainnet.near.org")

    try:
        status = await client.status()
        print("Node status:", status)

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
