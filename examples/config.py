from near_jsonrpc_client import RpcTimeoutError, NearClientSync, RpcClientError, RpcError, RpcHttpError


def main():
    client = NearClientSync(rpc_urls="https://rpc.mainnet.near.org")

    try:
        config = client.genesis_config()
        print("Genesis config:", config)

        config = client.client_config()
        print("Client config:", config)

    except RpcError as e:
        print(f"{e}: {e.error}")
    except RpcHttpError as e:
        print(f"{e}: status: {e.status_code}, body: {e.body}")
    except RpcClientError as e:
        print("Invalid response:", e)
    finally:
        client.close()


main()
