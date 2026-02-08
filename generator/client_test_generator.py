from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Mapping, Optional, List

from generator.helpers.name_resolvers import snake_case, pascal_case
from .context import GeneratorContext


def ref_name(ref: str) -> str:
    return ref.split("/")[-1]


class ClientTestGenerator:
    @staticmethod
    def generate(
        ctx: GeneratorContext,
        *,
        output_dir: Path = Path("tests"),
        models_module: str = "near_jsonrpc_models",
        client_module: str = "client",
        rpc_base_url: str = "https://rpc.test",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        schemas: Mapping[str, Any] = ctx.schemas
        paths: Mapping[str, Any] = ctx.paths

        tests: List[str] = []
        for path, path_item in sorted(paths.items()):
            if not isinstance(path_item, dict):
                continue
            for http_method, op in path_item.items():
                if http_method.lower() != "post":
                    continue

                operation_id = op.get("operationId") or ClientTestGenerator._make_operation_id_from_path(path, http_method)
                method_name = snake_case(operation_id)
                description = ClientTestGenerator._clean_description(op.get("description", ""))

                # request $ref -> schema name
                request_ref = ClientTestGenerator._extract_request_ref(op)
                if not request_ref:
                    continue
                request_schema_name = ref_name(request_ref)
                request_model_class = pascal_case(request_schema_name)

                # params type name (may differ)
                params_schema_name = ClientTestGenerator._extract_params_type_name(schemas, request_schema_name) or request_schema_name
                params_class_name = pascal_case(params_schema_name)

                # response $ref -> schema name
                response_ref = ClientTestGenerator._extract_response_ref(op)
                if not response_ref:
                    continue
                response_schema_name = ref_name(response_ref)
                response_model_class = pascal_case(response_schema_name)

                # decide if request params are root-null (enum: [null])
                is_root_null = ClientTestGenerator._is_root_null_schema(schemas, params_schema_name)

                # fixture filenames derived from schema names via snake_case
                req_fixture_name = snake_case(request_schema_name)
                resp_fixture_name = snake_case(response_schema_name)

                method_src = ClientTestGenerator._render_test_function(
                    method_name=method_name,
                    description=description,
                    models_module=models_module,
                    client_module=client_module,
                    request_model_class=request_model_class,
                    response_model_class=response_model_class,
                    params_class_name=params_class_name,
                    rpc_base_url=rpc_base_url,
                    is_root_null=is_root_null,
                    req_fixture_name=req_fixture_name,
                    resp_fixture_name=resp_fixture_name,
                )
                tests.append(method_src)

        test_module = ClientTestGenerator._render_test_module(
            models_module=models_module,
            client_module=client_module,
            tests=tests,
        )
        test_path = output_dir / "test_api_integration_mocked.py"
        test_path.write_text(test_module, encoding="utf-8")
        print(f"[ClientTestGenerator] Wrote {test_path}")

    # ---- helpers for reading spec ----
    @staticmethod
    def _extract_request_ref(op: Mapping[str, Any]) -> Optional[str]:
        rb = op.get("requestBody")
        if not rb:
            return None
        content = rb.get("content", {})
        app_json = content.get("application/json", {})
        schema = app_json.get("schema", {})
        return schema.get("$ref")

    @staticmethod
    def _extract_response_ref(op: Mapping[str, Any]) -> Optional[str]:
        responses = op.get("responses", {})
        r200 = responses.get("200") or responses.get("201") or {}
        content = r200.get("content", {})
        app_json = content.get("application/json", {})
        schema = app_json.get("schema", {})
        return schema.get("$ref")

    @staticmethod
    def _extract_params_type_name(schemas: Mapping[str, Any], request_schema_name: str) -> Optional[str]:
        schema = schemas.get(request_schema_name)
        if not schema:
            return None
        props = schema.get("properties", {}) or {}
        params = props.get("params")
        if not params:
            return None
        ref = params.get("$ref")
        if not ref:
            return None
        return ref_name(ref)

    @staticmethod
    def _is_root_null_schema(schemas: Mapping[str, Any], schema_name: str) -> bool:
        schema = schemas.get(schema_name)
        if not isinstance(schema, dict):
            return False
        enum = schema.get("enum")
        if isinstance(enum, list) and len(enum) == 1 and enum[0] is None:
            return True
        return False

    # ---- rendering test functions ----
    @staticmethod
    def _render_test_function(
        *,
        method_name: str,
        description: str,
        models_module: str,
        client_module: str,
        request_model_class: str,
        response_model_class: str,
        params_class_name: str,
        rpc_base_url: str,
        is_root_null: bool,
        req_fixture_name: str,
        resp_fixture_name: str,
    ) -> str:
        safe_desc = description.replace('"""', '\\"\\"\\"')

        lines: List[str] = []
        lines.append("")  # separation
        lines.append(f"def test_{method_name}_mocked():")
        lines.append('    """')
        lines.append(f"    {safe_desc}")
        lines.append('    """')
        lines.append("")
        lines.append(f"    client = NearClientSync(rpc_urls=\"{rpc_base_url}\")")
        lines.append("")
        # runtime fixture loader usage
        lines.append(f"    req_fixture = load_mock_json('{req_fixture_name}.json')")
        lines.append(f"    resp_fixture = load_mock_json('{resp_fixture_name}.json')")
        lines.append("")
        lines.append("    expected_request = req_fixture")
        lines.append("    expected_response = resp_fixture")
        lines.append("")
        lines.append("    with respx.mock:")
        # use fixture response directly if already JSON-RPC envelope
        lines.append("        if isinstance(expected_response, dict) and {'jsonrpc','id','result'}.issubset(expected_response):")
        lines.append("            mock_response_json = expected_response")
        lines.append("        else:")
        lines.append("            mock_response_json = {'jsonrpc': '2.0', 'id': 'test', 'result': expected_response}")
        lines.append(f"        route = respx.post(\"{rpc_base_url}\").mock(return_value=Response(200, json=mock_response_json))")
        lines.append("")
        # prepare params
        lines.append("        req = expected_request")
        lines.append("        if isinstance(req, dict) and 'params' in req:")
        lines.append("            req_params = req['params']")
        lines.append("        else:")
        lines.append("            req_params = req")
        lines.append("")
        # call client: omit params if None
        lines.append("        if req_params is None:")
        lines.append(f"            result = client.{method_name}()")
        lines.append("        else:")
        lines.append(f"            params = {models_module}.{params_class_name}.model_validate(req_params)")
        lines.append(f"            result = client.{method_name}(params=params)")
        lines.append("")
        # ensure called and parse request body robustly
        lines.append("        assert route.called")
        lines.append("        assert len(route.calls) >= 1")
        lines.append("        sent = None")
        lines.append("        req_obj = route.calls[0].request")
        lines.append("        body = getattr(req_obj, 'content', None)")
        lines.append("        if body is None:")
        lines.append("            try:")
        lines.append("                body = req_obj.read()")
        lines.append("            except Exception:")
        lines.append("                body = None")
        lines.append("        if isinstance(body, bytes):")
        lines.append("            try:")
        lines.append("                body = body.decode('utf-8')")
        lines.append("            except Exception:")
        lines.append("                body = body.decode('latin-1')")
        lines.append("        if isinstance(body, str) and body.strip():")
        lines.append("            try:")
        lines.append("                sent = json.loads(body)")
        lines.append("            except Exception:")
        lines.append("                sent = None")
        lines.append("")
        # compare params
        lines.append("        if isinstance(sent, dict):")
        lines.append("            sent_params = sent.get('params')")
        lines.append("            exp = req_params")
        lines.append("            if isinstance(expected_request, dict) and 'params' in expected_request:")
        lines.append("                exp = expected_request.get('params')")
        lines.append("            if exp is None:")
        lines.append(f"                assert ('params' not in sent) or (sent.get('params') is None), \"params mismatch for {method_name}: expected None but found present\"")
        lines.append("            else:")
        lines.append(f"                assert sent_params == exp, \"params mismatch for {method_name}: sent={{}} expected={{}}\".format(sent_params, exp)")
        lines.append("")
        # prepare expected result
        lines.append("        exp_resp = expected_response")
        lines.append("        if isinstance(exp_resp, dict) and 'result' in exp_resp:")
        lines.append("            exp_resp = exp_resp['result']")
        lines.append("")
        # normalize result using module-level _normalize helper
        lines.append("        result_value = _normalize(result)")
        lines.append("")
        lines.append(f"        assert result_value == exp_resp, \"result mismatch for {method_name}: got={{}} expected={{}}\".format(result_value, exp_resp)")
        lines.append("")  # trailing newline

        return "\n".join(lines)

    @staticmethod
    def _render_test_module(*, models_module: str, client_module: str, tests: List[str]) -> str:
        header = textwrap.dedent(
            """\
            # Auto-generated integration tests (mocked) - generated by ClientTestGenerator
            # Uses: pytest, respx, httpx
            """
        ).strip() + "\n\n"

        # module-level fixture helper, normalize helper and imports
        helper = textwrap.dedent(
            """\
            import pytest
            import json
            from pathlib import Path
            import respx
            from httpx import Response
            import {models_module}
            from {client_module} import NearClientSync

            FIXTURES_DIR = Path(__file__).parent / "fixtures" / "json"

            def load_mock_json(name: str):
                p = Path(FIXTURES_DIR) / name
                if not p.exists():
                    return None
                with p.open("r", encoding="utf-8") as fh:
                    return json.load(fh)

            def _normalize(obj):
                \"\"\"Recursively convert pydantic near_jsonrpc_models (and lists/dicts containing them)
                to JSON-serializable Python primitives comparable with fixture JSON.

                Rules:
                - If object has model_dump, call model_dump(mode='json', exclude_none=True)
                - If list, normalize each element
                - If dict, normalize values
                - Otherwise, return as-is
                \"\"\"
                # pydantic model
                try:
                    # check duck-typing attribute (works for pydantic v1 & v2)
                    if hasattr(obj, "model_dump") or hasattr(obj, "dict"):
                        if hasattr(obj, "model_dump"):
                            try:
                                dumped = obj.model_dump(mode="json", exclude_none=True)
                            except TypeError:
                                # fallback for different pydantic signatures
                                dumped = obj.model_dump()
                        else:
                            # pydantic v1 fallback
                            dumped = obj.dict(exclude_none=True)
                        return _normalize(dumped)
                except Exception:
                    # if something unexpected happens, fall through and try other checks
                    pass

                # list -> normalize items
                if isinstance(obj, list):
                    return [_normalize(v) for v in obj]

                # dict -> normalize values
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        out[k] = _normalize(v)
                    return out

                # primitive
                return obj

            """.replace("{models_module}", models_module).replace("{client_module}", client_module)
        )

        body = "\n\n".join(tests)

        return header + helper + body

    @staticmethod
    def _make_operation_id_from_path(path: str, method: str) -> str:
        clean = re.sub(r"[/{}}]+", "_", path).strip("_")
        return f"{clean}_{method}"

    @staticmethod
    def _clean_description(description: str) -> str:
        return description.replace("\n", " ").strip()
