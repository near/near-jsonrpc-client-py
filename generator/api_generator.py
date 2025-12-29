from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Mapping, Optional, List

from .context import GeneratorContext
from generator.helpers.name_resolvers import snake_case, pascal_case


def ref_name(ref: str) -> str:
    return ref.split("/")[-1]


class ApiGenerator:
    @staticmethod
    def generate(
        ctx: GeneratorContext,
        *,
        output_dir: Path = Path("client"),
        models_module: str = "models",
        base_client_module: str = "base_client",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        schemas: Mapping[str, Any] = ctx.schemas
        paths: Mapping[str, Any] = ctx.paths

        methods: List[str] = []
        for path, path_item in sorted(paths.items()):
            if not isinstance(path_item, dict):
                continue
            for http_method, op in path_item.items():
                if http_method.lower() != "post":
                    continue

                operation_id = op.get("operationId") or ApiGenerator._make_operation_id_from_path(path, http_method)
                method_name = snake_case(operation_id)
                description = ApiGenerator._clean_description(op.get("description", ""))

                request_ref = ApiGenerator._extract_request_ref(op)
                if not request_ref:
                    continue
                request_schema_name = ref_name(request_ref)
                request_model_class = pascal_case(request_schema_name)

                params_schema_name = ApiGenerator._extract_params_type_name(schemas, request_schema_name) or request_schema_name
                params_class_name = pascal_case(params_schema_name)

                response_ref = ApiGenerator._extract_response_ref(op)
                if not response_ref:
                    continue
                response_schema_name = ref_name(response_ref)
                response_model_class = pascal_case(response_schema_name)

                result_schema_name = ApiGenerator._extract_result_type_from_response(schemas, response_schema_name)
                if result_schema_name:
                    return_type_hint = f"{models_module}.{pascal_case(result_schema_name)}"
                else:
                    return_type_hint = f"{models_module}.{response_model_class}"

                use_model_validate_none = ApiGenerator._is_root_null_schema(schemas, params_schema_name)

                method_src = ApiGenerator._render_method(
                    method_name=method_name,
                    description=description,
                    models_module=models_module,
                    request_model_class=request_model_class,
                    response_model_class=response_model_class,
                    params_class_name=params_class_name,
                    return_type_hint=return_type_hint,
                    operation_id=operation_id,
                    use_model_validate_none=use_model_validate_none,
                )

                # اضافه کردن indentation مناسب برای قرار گرفتن داخل کلاس
                method_src_indented = textwrap.indent(method_src, '    ')
                methods.append(method_src_indented)

        api_methods_content = ApiGenerator._render_api_module(
            models_module=models_module,
            methods=methods,
            base_client_module=base_client_module,
        )
        api_methods_path = output_dir / "api_methods.py"
        api_methods_path.write_text(api_methods_content, encoding="utf-8")

        client_py_path = output_dir / "client.py"
        base_import_line = f"from .{base_client_module} import NearBaseClient"
        if not client_py_path.exists():
            content = textwrap.dedent(f"""{base_import_line}
from .api_methods import APIMixin

class NearClient(NearBaseClient, APIMixin):
    \"\"\"NearClient with generated API methods mixed in.\"\"\"
    pass
""")
            client_py_path.write_text(content, encoding="utf-8")

        print(f"✅ [ApiGenerator] Wrote {api_methods_path} and patched {client_py_path}")

    # ---- helpers ----
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
    def _extract_result_type_from_response(schemas: Mapping[str, Any], response_schema_name: str) -> Optional[str]:
        schema = schemas.get(response_schema_name)
        if not schema:
            return None
        one_of = schema.get("oneOf")
        if not isinstance(one_of, list):
            return None
        for branch in one_of:
            props = branch.get("properties", {}) or {}
            result_prop = props.get("result")
            if isinstance(result_prop, dict):
                ref = result_prop.get("$ref")
                if ref:
                    return ref_name(ref)
        return None

    @staticmethod
    def _is_root_null_schema(schemas: Mapping[str, Any], schema_name: str) -> bool:
        schema = schemas.get(schema_name)
        if not isinstance(schema, dict):
            return False
        enum = schema.get("enum")
        if isinstance(enum, list) and len(enum) == 1 and enum[0] is None:
            return True
        return False

    @staticmethod
    def _render_method(
        *,
        method_name: str,
        description: str,
        models_module: str,
        request_model_class: str,
        response_model_class: str,
        params_class_name: str,
        return_type_hint: str,
        operation_id: str,
        use_model_validate_none: bool,
    ) -> str:
        doc = (description or f"Call {operation_id}").strip()
        safe_doc = doc.replace('"""', '\\"\\"\\"')

        if use_model_validate_none:
            method = f'''
async def {method_name}(self: "NearBaseClient") -> {return_type_hint}:
    """
    {safe_doc}
    High-level method: returns the result model or raises NearClientError/NearRpcError/NearHttpError.
    """
    params = {models_module}.{params_class_name}.model_validate(None)
    return await self._call(
        request_model={models_module}.{request_model_class},
        response_model={models_module}.{response_model_class},
        params=params,
    )
'''
        else:
            method = f'''
async def {method_name}(self: "NearBaseClient", *, params: {models_module}.{params_class_name}) -> {return_type_hint}:
    """
    {safe_doc}
    High-level method: returns the result model or raises NearClientError/NearRpcError/NearHttpError.
    """
    return await self._call(
        request_model={models_module}.{request_model_class},
        response_model={models_module}.{response_model_class},
        params=params,
    )
'''
        return method

    @staticmethod
    def _render_api_module(*, models_module: str, methods: List[str], base_client_module: str) -> str:
        header = textwrap.dedent(f'''\
            """Auto-generated API mixin (generated by ClientGenerator).

            This file contains APIMixin which provides generated API methods.
            The runtime provides a NearBaseClient implementation (located in '{base_client_module}')
            that exposes `_call(...)`.
            """
        ''')
        imports = textwrap.dedent(f'''\
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                from .{base_client_module} import NearBaseClient

            import {models_module}
        ''')
        class_def = "class APIMixin:\n\n"
        class_body = "    pass\n" if not methods else "".join(methods)
        return "\n\n".join([header, imports, class_def + class_body])

    @staticmethod
    def _make_operation_id_from_path(path: str, method: str) -> str:
        clean = re.sub(r"[/{}]+", "_", path).strip("_")
        return f"{clean}_{method}"

    @staticmethod
    def _clean_description(description: str) -> str:
        return description.replace("\n", " ").strip()
