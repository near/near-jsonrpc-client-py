from near_jsonrpc_client import RequestTimeoutError, NearClientSync, ClientError, RpcError, HttpError


def main():
    client = NearClientSync(base_url="https://rpc.mainnet.near.org")

    try:
        config = client.genesis_config()
        print("Genesis config:", config)

        config = client.client_config()
        print("Client config:", config)

    except RpcError as e:
        print(f"{e}: {e.error}")
    except HttpError as e:
        print(f"{e}: status: {e.status_code}, body: {e.body}")
    except ClientError as e:
        print("Invalid response:", e)
    finally:
        client.close()


main()
