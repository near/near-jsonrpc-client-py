from pathlib import Path

from generator import model_generator
from generator.api_generator import ApiGenerator
from generator.client_test_generator import ClientTestGenerator
from generator.context import GeneratorContext
from generator.loader import load_openapi
from generator.mock_generator import MockGenerator
from generator.model_test_generator import ModelTestGenerator

OPENAPI_URL = "https://raw.githubusercontent.com/near/nearcore/master/chain/jsonrpc/openapi/openapi.json"


def main():
    spec = load_openapi(OPENAPI_URL)
    ctx = GeneratorContext(spec)

    model_generator.generate_models(ctx)

    samples_dir = Path("tests/fixtures/json")
    MockGenerator.generate(ctx, [samples_dir])

    ModelTestGenerator.generate_tests_for_models(
        ctx=ctx,
        output_dir=Path("tests"),
    )

    ApiGenerator.generate(ctx, output_dir=Path("client"), models_module="models")

    ClientTestGenerator.generate(
        ctx,
        output_dir=Path("tests"),
        models_module="models",
        client_module="client",
        rpc_base_url="https://rpc.mainnet.near.org",
    )


if __name__ == "__main__":
    main()
