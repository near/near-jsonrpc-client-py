from pathlib import Path
from typing import List
from generator.context import GeneratorContext
from generator.helpers.name_resolvers import snake_case, pascal_case


class ModelTestGenerator:

    @staticmethod
    def generate_tests_for_models(
        ctx: GeneratorContext,
        output_dir: Path,
        test_file_name: str = "test_model_serialization.py",
    ) -> None:

        schemas = sorted(ctx.schemas.keys())
        output_dir.mkdir(parents=True, exist_ok=True)

        file_content = ModelTestGenerator._build_test_file(
            schemas=schemas,
        )

        test_file = output_dir / test_file_name
        test_file.write_text(file_content, encoding="utf-8")

        print(f"✅ Model serialization tests generated at: {test_file.resolve()}")

    @staticmethod
    def _build_test_file(
        schemas: List[str],
    ) -> str:

        lines: List[str] = []

        # imports
        lines += [
            "import json",
            "from pathlib import Path",
            "import pytest",
            "import models",
            "",
            "",
            "# ----------------------------------------------------------------------",
            "# Generated model (de)serialization tests",
            "# ----------------------------------------------------------------------",
            "",
            f'FIXTURES_DIR = Path(__file__).parent / "fixtures" / "json"',
            "",
            "def load_mock_json(filename: str) -> str:",
            "    path = FIXTURES_DIR / filename",
            "    if not path.exists():",
            "        pytest.fail(f'Mock file {filename} does not exist!')",
            "    return path.read_text(encoding='utf-8')",
            "",
            "",
        ]

        # tests
        for schema_name in schemas:
            class_name = schema_name  # assuming class name == schema name
            file_name = f"{snake_case(schema_name)}.json"

            lines += [
                f"def test_{snake_case(class_name)}_encode_decode():",
                f"    data = load_mock_json('{file_name}')",
                "",
                "    try:",
                f"        model_cls = getattr(models, '{pascal_case(class_name)}')",
                "        obj1 = model_cls.model_validate_json(data)",
                "        json2 = obj1.model_dump_json()",
                "        obj2 = model_cls.model_validate_json(json2)",
                "",
                "        assert obj1 == obj2",
                "    except Exception as e:",
                f"        pytest.fail(f'Serialization test failed for {pascal_case(class_name)}')",
                "",
                "",
            ]

        return "\n".join(lines)
