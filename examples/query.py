import asyncio
from near_jsonrpc_client import NearClientAsync, ClientError, RpcError, HttpError, RequestTimeoutError
from near_jsonrpc_models import RpcQueryRequest, AccountId, FunctionArgs, RpcQueryRequestCallFunctionByFinality, \
    RpcQueryRequestViewAccountByFinality, Finality


async def main():
    client = NearClientAsync(base_url="https://rpc.mainnet.near.org")

    try:
        params = RpcQueryRequest(
            RpcQueryRequestCallFunctionByFinality(
                finality=Finality('final').root,
                account_id=AccountId(root='wrap.near'),
                args_base64=FunctionArgs('e30='),
                method_name='ft_balance_of2',
                request_type='call_function',
            )
        )

        tx = await client.query(params=params)
        print("call ft balance:", tx)

        params = RpcQueryRequest(
            RpcQueryRequestViewAccountByFinality(
                finality=Finality('final').root,
                account_id=AccountId(root='wrap.near').root,
                request_type='view_account',
            )
        )

        tx = await client.query(params=params)
        print("call view account by finality:", tx)

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
