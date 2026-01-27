import ast
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, List, Tuple

from generator.context import GeneratorContext
from generator.helpers.name_resolvers import snake_case, pascal_case, to_class_name

# numeric bounds
_INT32_MIN = -2_147_483_648
_INT32_MAX = 2_147_483_647
_UINT8_MAX = 2 ** 8 - 1
_UINT16_MAX = 2 ** 16 - 1
_UINT32_MAX = 2 ** 32 - 1
_UINT64_MAX = 2 ** 64 - 1


def ensure_optional_annotation(ann: str) -> str:
    s = str(ann).strip()
    if s.endswith("| None") or "Optional[" in s:
        return ann
    return f"{ann} | None"


def _is_mutable_default(v: Any) -> bool:
    return isinstance(v, (list, dict, set))


def _field_is_ref_like(field_data: Dict[str, Any]) -> bool:
    if not isinstance(field_data, dict):
        return False
    if "$ref" in field_data:
        return True
    if "allOf" in field_data and isinstance(field_data["allOf"], list):
        for entry in field_data["allOf"]:
            if isinstance(entry, dict) and "$ref" in entry:
                return True
    return False


def _is_nullable(field_data: Dict[str, Any]) -> bool:
    if not isinstance(field_data, dict):
        return False
    if field_data.get("nullable") is True:
        return True
    if field_data.get("enum") == [None]:
        return True

    def contains_null_variant(arr: Any) -> bool:
        if not isinstance(arr, list):
            return False
        for o in arr:
            if not isinstance(o, dict):
                continue
            if o.get("nullable") is True:
                return True
            if o.get("enum") == [None]:
                return True
            if o.get("type") == "null":
                return True
            if o.get("enum") and None in o.get("enum"):
                return True
        return False

    if contains_null_variant(field_data.get("anyOf")):
        return True
    if contains_null_variant(field_data.get("oneOf")):
        return True

    return False


def _inject_class_docstring(class_def: str, class_name: str, description: Optional[str]) -> str:
    if not description:
        return class_def
    desc_sanitized = description.replace('"""', '\"\"\"')
    headers = [
        f"class {class_name}(BaseModel):\n",
        f"class {class_name}(StrictBaseModel):\n",
        f"class {class_name}(RootModel",
    ]
    for header in headers:
        idx = class_def.find(header)
        if idx != -1:
            lines = class_def.splitlines(True)
            for i, line in enumerate(lines):
                if line.startswith(f"class {class_name}"):
                    return (
                        "".join(lines[: i + 1])
                        + f'    """{desc_sanitized}"""\n'
                        + "".join(lines[i + 1:])
                    )
    lines = class_def.splitlines(True)
    if lines:
        return lines[0] + f'    """{desc_sanitized}"""\n' + "".join(lines[1:])
    return class_def


# -------------------------
# validators
# -------------------------
def generate_pattern_properties_validator(field_name: str, pattern: str) -> str:
    method_name = f"validate_{field_name}_keys"
    return (
        f"    @field_validator('{field_name}')\n"
        f"    def {method_name}(cls, v):\n"
        f"        import re\n"
        f"        pattern = re.compile(r\"{pattern}\")\n"
        f"        if not isinstance(v, dict):\n"
        f"            raise TypeError('{field_name} must be a dict')\n"
        f"        for key in v.keys():\n"
        f"            if not pattern.match(key):\n"
        f"                raise ValueError(f\"Invalid key '{{key}}' in {field_name}. Must match {pattern!r}\")\n"
        f"        return v\n\n"
    )


def generate_datetime_string_parser_validator(field_name: str) -> str:
    method_name = f"parse_{field_name}_to_datetime"
    return (
        f"    @field_validator('{field_name}', mode='before')\n"
        f"    def {method_name}(cls, v):\n"
        f"        from datetime import datetime\n"
        f"        if v is None:\n"
        f"            return v\n"
        f"        if isinstance(v, datetime):\n"
        f"            return v\n"
        f"        if isinstance(v, str):\n"
        f"            s = v\n"
        f"            if s.endswith('Z'):\n"
        f"                s = s[:-1] + '+00:00'\n"
        f"            try:\n"
        f"                return datetime.fromisoformat(s)\n"
        f"            except Exception as e:\n"
        f"                raise ValueError(f\"{field_name} must be an ISO-8601 datetime string: {{e}}\")\n"
        f"        raise TypeError('{field_name} must be a datetime or ISO-8601 string')\n\n"
    )


# -------------------------
# unique-name helper
# -------------------------
def _ensure_unique_name(candidate: str, used: Set[str], from_title: bool = False) -> str:
    """
    Deterministic minimal-suffix unique namer.
    If candidate unused -> return and mark used.
    If from_title is True and candidate already used -> return candidate as-is (do NOT append numeric suffix).
    Else append smallest positive integer to the candidate's base (base = candidate without trailing digits).
    """
    if candidate not in used:
        used.add(candidate)
        return candidate

    if from_title:
        # keep candidate as-is when it was derived from a title — do not append numeric suffix
        return candidate

    m = re.match(r"^(.*?)(\d+)$", candidate)
    base = m.group(1) if m else candidate
    if not base:
        base = candidate

    i = 1
    while True:
        cand = f"{base}{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


# -------------------------
# descriptive suffix helpers
# -------------------------
def _prop_name_to_suffix(part: str) -> str:
    """
    Convert a property name like 'block_id' -> 'BlockId', 'tx_hash' -> 'TxHash' (but we prefer 'Id' when prop name endswith '_id').
    """
    # keep underscores, use pascal_case from helper
    return pascal_case(part)


def _make_suffix_from_props(props: Dict[str, Any], prop_order: List[str]) -> Optional[str]:
    """
    Build suffix using properties in the given order.
    Returns None if props empty.
    Example: ['block_id','shard_id'] -> 'BlockIdShardId'
    """
    if not prop_order:
        return None
    parts = []
    for p in prop_order:
        # only include props that actually exist in props dict
        if p in props:
            parts.append(_prop_name_to_suffix(p))
    if parts:
        return "".join(parts)
    # fallback: if props dict exists but we couldn't order, pick first two keys
    keys = list(props.keys())
    if keys:
        return "".join(_prop_name_to_suffix(k) for k in keys[:2])
    return None


def _extract_single_name_enum(props: Dict[str, Any]) -> Optional[str]:
    if not isinstance(props, dict):
        return None

    def extract_from_field(field_name: str) -> Optional[str]:
        field = props.get(field_name)
        if not isinstance(field, dict):
            return None

        enum_val = None
        if "enum" in field and isinstance(field["enum"], list) and len(field["enum"]) == 1:
            enum_val = field["enum"][0]
        elif "const" in field:
            enum_val = field["const"]

        if isinstance(enum_val, str) and enum_val.strip():
            return pascal_case(enum_val)

        return None

    # Priority: name first, then type
    return extract_from_field("name") or extract_from_field("type")


# -------------------------
# version-property helper
# -------------------------
def _prop_is_version_name(prop: str) -> bool:
    """Return True for names like 'V0', 'V1', 'v2' — indicates version-property naming that should be preserved."""
    return bool(re.match(r"^[Vv]\d+$", str(prop)))


def _needs_option_suffix(combined_name: str, props: Dict[str, Any]) -> bool:
    """
    Return True when any property inside `props` references a model whose
    pascal-cased name equals combined_name. This indicates a name collision
    like combined == 'ShardLayoutV0' while a property refers to ShardLayoutV0.
    """
    if not combined_name or not isinstance(props, dict):
        return False
    for v in props.values():
        if not isinstance(v, dict):
            continue
        # direct $ref
        if "$ref" in v:
            ref_name = pascal_case(v["$ref"].split("/")[-1])
            if ref_name == combined_name:
                return True
        # allOf containing $ref(s)
        if "allOf" in v and isinstance(v["allOf"], list):
            for e in v["allOf"]:
                if isinstance(e, dict) and "$ref" in e:
                    ref_name = pascal_case(e["$ref"].split("/")[-1])
                    if ref_name == combined_name:
                        return True
    return False


# -------------------------
# type resolution (unchanged in essence)
# returns (placeholder/python-object, annotation-string)
# -------------------------
def get_python_type_and_string(
        field: str,
        json_type: Optional[str],
        field_data: Dict[str, Any],
        ctx: GeneratorContext,
        imports: Set[str],
        parent_name: Optional[str] = None,
        extra_defs: Optional[List[str]] = None,
) -> Tuple[Any, str]:
    if extra_defs is None:
        extra_defs = []

    # special allOf single-$ref
    if "allOf" in field_data and isinstance(field_data["allOf"], list):
        if len(field_data["allOf"]) == 1 and isinstance(field_data["allOf"][0], dict) and "$ref" in field_data["allOf"][0]:
            ref = field_data["allOf"][0]["$ref"]
            model_name = pascal_case(ref.split("/")[-1])
            imports.add(f"from near_jsonrpc_models.{snake_case(model_name)} import {model_name}")
            return model_name, model_name

    # inline enum -> Literal
    if "enum" in field_data and field_data.get("type") == "string" and "$ref" not in field_data:
        imports.add("from typing import Literal")
        vals = field_data["enum"]
        if len(vals) == 1:
            return str, f"Literal[{repr(vals[0])}]"
        return str, "Literal[" + ", ".join(repr(v) for v in vals) + "]"

    # anyOf pattern nullable + $ref -> Model | None
    if "anyOf" in field_data:
        ref_opt = next((o for o in field_data["anyOf"] if isinstance(o, dict) and "$ref" in o), None)
        null_opt = next((o for o in field_data["anyOf"] if isinstance(o, dict) and (o.get("enum") == [None] or o.get("nullable") is True)), None)
        if ref_opt and null_opt:
            name = pascal_case(ref_opt["$ref"].split("/")[-1])
            imports.add(f"from near_jsonrpc_models.{snake_case(name)} import {name}")
            union = f"{name} | None"
            return union, union

    # oneOf string-enum collapse
    if "oneOf" in field_data:
        flattenable = True
        vals = []
        for opt in field_data["oneOf"]:
            if isinstance(opt, dict) and opt.get("type") == "string" and "enum" in opt and isinstance(opt["enum"], list):
                vals.extend(opt["enum"])
            else:
                flattenable = False
                break
        if flattenable and vals:
            seen = set()
            dedup_vals = []
            for v in vals:
                if v not in seen:
                    dedup_vals.append(v)
                    seen.add(v)
            imports.add("from typing import Literal")
            if len(dedup_vals) == 1:
                return str, f"Literal[{repr(dedup_vals[0])}]"
            return str, "Literal[" + ", ".join(repr(v) for v in dedup_vals) + "]"

    # direct $ref
    if "$ref" in field_data:
        name = pascal_case(field_data["$ref"].split("/")[-1])
        imports.add(f"from near_jsonrpc_models.{snake_case(name)} import {name}")
        return name, name

    # integer
    if json_type == "integer":
        minimum = field_data.get("minimum")
        maximum = field_data.get("maximum")
        fmt = field_data.get("format")
        if fmt == "int32" and minimum is None and maximum is None:
            minimum = _INT32_MIN
            maximum = _INT32_MAX
        if fmt in ("uint64", "unint64"):
            if minimum is None:
                minimum = 0
            if maximum is None:
                maximum = _UINT64_MAX
        if fmt == "uint32":
            if minimum is None:
                minimum = 0
            if maximum is None:
                maximum = _UINT32_MAX
        if fmt == "uint16":
            if minimum is None:
                minimum = 0
            if maximum is None:
                maximum = _UINT16_MAX
        if fmt == "uint8":
            if minimum is None:
                minimum = 0
            if maximum is None:
                maximum = _UINT8_MAX
        if fmt == "uint":
            if minimum is None:
                minimum = 0
            if maximum is None:
                maximum = _UINT32_MAX
        if minimum is not None or maximum is not None:
            imports.add("from pydantic import conint")
            args = []
            if minimum is not None:
                args.append(f"ge={minimum}")
            if maximum is not None:
                args.append(f"le={maximum}")
            return int, f"conint({', '.join(args)})" if args else "conint()"
        return int, "int"

    # number
    if json_type == "number":
        minimum = field_data.get("minimum")
        maximum = field_data.get("maximum")
        if minimum is not None or maximum is not None:
            imports.add("from pydantic import condecimal")
            args = []
            if minimum is not None:
                args.append(f"ge={minimum}")
            if maximum is not None:
                args.append(f"le={maximum}")
            return float, f"condecimal({', '.join(args)})"
        return float, "float"

    # string
    if json_type == "string":
        fmt = field_data.get("format")
        if fmt == "bytes" or fmt == "byte":
            return str, "str"
        if fmt == "date-time":
            imports.add("from datetime import datetime")
            return datetime, "datetime"
        min_len = field_data.get("minLength")
        max_len = field_data.get("maxLength")
        pattern = field_data.get("pattern")
        if any(v is not None for v in (min_len, max_len, pattern)):
            imports.add("from pydantic import constr")
            args = []
            if min_len is not None:
                args.append(f"min_length={min_len}")
            if max_len is not None:
                args.append(f"max_length={max_len}")
            if pattern is not None:
                args.append(f"regex={repr(pattern)}")
            return str, f"constr({', '.join(args)})"
        return str, "str"

    # boolean
    if json_type == "boolean":
        return bool, "bool"

    # array
    if json_type == "array":
        items = field_data.get("items", {})
        min_items = field_data.get("minItems")
        max_items = field_data.get("maxItems")

        def build_container_annotation(item_annotation: str) -> str:
            if min_items is None and max_items is None:
                return f"List[{item_annotation}]"
            imports.add("from pydantic import conlist")
            args = []
            if min_items is not None:
                args.append(f"min_length={min_items}")
            if max_items is not None:
                args.append(f"max_length={max_items}")
            args_str = ", ".join(args)
            return f"conlist({item_annotation}, {args_str})" if args_str else f"conlist({item_annotation})"

        # --- handle tuple-style fixed-length arrays where items is a list ---
        if isinstance(items, list):
            sub_ann_list: List[str] = []
            for i, sub_item in enumerate(items):
                # $ref case
                if isinstance(sub_item, dict) and "$ref" in sub_item:
                    ref_model = pascal_case(sub_item["$ref"].split("/")[-1])
                    imports.add(f"from near_jsonrpc_models.{snake_case(ref_model)} import {ref_model}")
                    sub_ann_list.append(ref_model)
                    continue

                # inline object -> create nested class
                if isinstance(sub_item, dict) and sub_item.get("type") == "object" and "properties" in sub_item:
                    root = parent_name if parent_name else pascal_case(field or "Anon")
                    item_class = to_class_name(root, f"{field.capitalize()}Item{i}")
                    nested_defs: List[str] = []
                    class_def = _build_class_from_properties(
                        item_class,
                        sub_item.get("properties", {}),
                        sub_item.get("required", []),
                        ctx,
                        imports,
                        extra_defs=nested_defs,
                        strict=sub_item.get("additionalProperties") is False,
                    )
                    extra_defs.extend(nested_defs)
                    extra_defs.append(class_def)
                    sub_ann_list.append(item_class)
                    continue

                # general case: recurse to resolve type/annotation string
                if isinstance(sub_item, dict):
                    sub_type = sub_item.get("type", "string")
                    _, sub_str = get_python_type_and_string(f"{field}_{i}", sub_type, sub_item, ctx, imports,
                                                           parent_name=parent_name, extra_defs=extra_defs)
                    sub_ann_list.append(sub_str)
                else:
                    # unexpected shape; fallback to Any
                    imports.add("from typing import Any")
                    sub_ann_list.append("Any")

            imports.add("from typing import Tuple")
            py = f"Tuple[{', '.join(sub_ann_list)}]"
            return py, py
        # --- end tuple-style handling ---

        if isinstance(items, dict) and items.get("type") == "object" and "properties" in items:
            root = parent_name if parent_name else pascal_case(field or "Anon")
            item_class = to_class_name(root, f"{field.capitalize()}Item")
            nested_defs: List[str] = []
            class_def = _build_class_from_properties(item_class, items.get("properties", {}), items.get("required", []),
                                                     ctx, imports, extra_defs=nested_defs,
                                                     strict=items.get("additionalProperties") is False)
            extra_defs.extend(nested_defs)
            extra_defs.append(class_def)
            item_ann = item_class
            container_ann = build_container_annotation(item_ann)
            imports.add("from typing import List")
            return f"List[{item_class}]", container_ann
        if "$ref" in items:
            ref_model = pascal_case(items["$ref"].split("/")[-1])
            imports.add(f"from near_jsonrpc_models.{snake_case(ref_model)} import {ref_model}")
            item_ann = ref_model
            container_ann = build_container_annotation(item_ann)
            imports.add("from typing import List")
            return f"List[{ref_model}]", container_ann
        item_type = items.get("type", "string")
        imports.add("from typing import List")
        _, item_str = get_python_type_and_string(field, item_type, items, ctx, imports, parent_name=parent_name,
                                                 extra_defs=extra_defs)
        container_ann = build_container_annotation(item_str)
        return f"List[{item_str}]", container_ann

    # object/dict-like
    if json_type == "object":
        if "properties" in field_data:
            root = parent_name if parent_name else pascal_case(field or "Anon")
            field_pc = pascal_case(field)
            if root.endswith(field_pc):
                cls_name = f"{root}Payload"
            else:
                cls_name = to_class_name(root, field)
            nested_defs: List[str] = []
            strict_obj = field_data.get("additionalProperties") is False
            class_def = _build_class_from_properties(
                cls_name,
                field_data.get("properties", {}),
                field_data.get("required", []),
                ctx,
                imports,
                extra_defs=nested_defs,
                strict=strict_obj,
            )
            extra_defs.extend(nested_defs)
            extra_defs.append(class_def)
            return cls_name, cls_name

        if "patternProperties" in field_data:
            pattern_props = field_data["patternProperties"]
            _, value_schema = next(iter(pattern_props.items()))
            imports.add("from typing import Dict")
            _, val_str = get_python_type_and_string(field, value_schema.get("type", "string"), value_schema, ctx,
                                                    imports, parent_name=parent_name, extra_defs=extra_defs)
            return f"Dict[str, {val_str}]", f"Dict[str, {val_str}]"
        additional = field_data.get("additionalProperties", None)
        if additional is False:
            imports.add("from typing import Dict, Any")
            return "Dict[str, Any]", "Dict[str, Any]"
        if isinstance(additional, dict):
            imports.add("from typing import Dict")
            _, val_str = get_python_type_and_string(field, additional.get("type", "string"), additional, ctx, imports,
                                                    parent_name=parent_name, extra_defs=extra_defs)
            return f"Dict[str, {val_str}]", f"Dict[str, {val_str}]"
        imports.add("from typing import Dict")
        imports.add("from typing import Any")
        return "Dict[str, Any]", "Dict[str, Any]"

    return str, "str"



# -------------------------
# Build BaseModel from properties dict
# -------------------------
def _build_class_from_properties(class_name: str, props: Dict[str, Any], required_fields: List[str],
                                 ctx: GeneratorContext, imports: Set[str],
                                 extra_defs: Optional[List[str]] = None,
                                 strict: bool = False) -> str:
    if extra_defs is None:
        extra_defs = []
    base = "StrictBaseModel" if strict else "BaseModel"
    if strict:
        imports.add("from near_jsonrpc_models.strict_model import StrictBaseModel")
    else:
        imports.add("from pydantic import BaseModel")
    body = f"class {class_name}({base}):\n"
    if not props:
        body += "    pass\n\n"
        return body
    for field, fd in props.items():
        _, ann = get_python_type_and_string(field, fd.get("type"), fd, ctx, imports, parent_name=class_name,
                                            extra_defs=extra_defs)
        is_req = field in required_fields
        if fd.get("type") == "array":
            imports.add("from typing import List")

        has_default = "default" in fd
        default = fd.get("default", None)
        desc_field = fd.get("description")
        if desc_field:
            body += f"    # {desc_field.replace(chr(10), chr(10) + '    # ')}\n"

        nullable = _is_nullable(fd)

        if is_req:
            if has_default:
                body += f"    {field}: {ann} = {repr(default)}\n"
            else:
                body += f"    {field}: {ann}\n"
        else:
            if has_default:
                if default is None:
                    annotate = ensure_optional_annotation(ann)
                    body += f"    {field}: {annotate} = {repr(default)}\n"
                else:
                    if nullable:
                        annotate = ensure_optional_annotation(ann)
                    else:
                        annotate = ann
                    body += f"    {field}: {annotate} = {repr(default)}\n"
            else:
                if nullable:
                    annotate = ensure_optional_annotation(ann)
                else:
                    annotate = ann
                body += f"    {field}: {annotate} = None\n"

    body += "\n"
    return body


# -------------------------
# Top-level model generation
# -------------------------
def generate_model(schema_name: str, schema_data: Dict[str, Any], ctx: GeneratorContext) -> str:
    properties = schema_data.get("properties", {})
    required_fields = schema_data.get("required", [])
    description = schema_data.get("description", "")

    imports: Set[str] = {"from pydantic import BaseModel"}
    extra_class_defs: List[str] = []

    # used class names across generation context (schema names only)
    used_class_names: Set[str] = set(pascal_case(s) for s in ctx.schemas.keys())
    used_class_names.add(pascal_case(schema_name))

    # handle simple alias $ref or single allOf-$ref
    if isinstance(schema_data, dict):
        if "$ref" in schema_data and set(schema_data.keys()).issubset({"$ref", "description"}):
            ref = pascal_case(schema_data["$ref"].split("/")[-1])
            import_stmt = f"from near_jsonrpc_models.{snake_case(ref)} import {ref}"
            desc_block = f'"""{description}"""\n\n' if description else ""
            return f"{desc_block}{import_stmt}\n\n\n{pascal_case(schema_name)} = {ref}\n"
        if "allOf" in schema_data and isinstance(schema_data["allOf"], list) and len(schema_data["allOf"]) == 1 and isinstance(schema_data["allOf"][0], dict) and "$ref" in schema_data["allOf"][0] and set(schema_data.keys()).issubset({"allOf", "description"}):
            ref = pascal_case(schema_data["allOf"][0]["$ref"].split("/")[-1])
            import_stmt = f"from near_jsonrpc_models.{snake_case(ref)} import {ref}"
            desc_block = f'"""{description}"""\n\n' if description else ""
            return f"{desc_block}{import_stmt}\n\n\n{pascal_case(schema_name)} = {ref}\n"

    # top-level enum None / string enums handled similarly to previous implementations
    if schema_data.get("enum") == [None]:
        desc = f'"""{description}"""\n\n' if description else ""
        imports_local = {"from pydantic import RootModel", "from types import NoneType"}
        import_block = "\n".join(sorted(imports_local))
        class_def = f"class {pascal_case(schema_name)}(RootModel[NoneType]):\n    pass\n\n"
        return f"{desc}{import_block}\n\n\n{class_def}"

    if "enum" in schema_data and schema_data.get("type") == "string":
        imports_needed = {"from pydantic import RootModel", "from typing import Literal"}
        enum_name = pascal_case(schema_name)
        vals = schema_data["enum"]
        lit = ", ".join(repr(v) for v in vals)
        desc_block = f'"""{description}"""\n\n' if description else ""
        import_block = "\n".join(sorted(imports_needed))
        class_def = f"class {enum_name}(RootModel[Literal[{lit}]]):\n    pass\n\n"
        return f"{desc_block}{import_block}\n\n\n{class_def}"

    # oneOf handling (improved naming: prefer title then property-derived suffixes then name-enum; prefer $ref as suffix when option is a $ref)
    if "oneOf" in schema_data:
        all_string_enums = True
        val_to_desc: Dict[str, Optional[str]] = {}
        for opt in schema_data["oneOf"]:
            if isinstance(opt, dict) and opt.get("type") == "string" and isinstance(opt.get("enum"), list) and opt.get("enum"):
                for v in opt["enum"]:
                    if v not in val_to_desc:
                        val_to_desc[v] = opt.get("description")
            else:
                all_string_enums = False
                break
        if all_string_enums and val_to_desc:
            imports_needed = {"from pydantic import RootModel", "from typing import Literal"}
            desc_lines: List[str] = []
            if description:
                desc_lines.append(description)
            for v, d in val_to_desc.items():
                if d:
                    desc_lines.append(f"{v}: {d}")
            desc_block = f'"""{"".join(desc_lines)}"""\n\n' if desc_lines else ""
            lit_vals = ", ".join(repr(v) for v in val_to_desc.keys())
            import_block = "\n".join(sorted(imports_needed))
            class_def = f"class {pascal_case(schema_name)}(RootModel[Literal[{lit_vals}]]):\n    pass\n\n"
            return f"{desc_block}{import_block}\n\n\n{class_def}"

        # complex oneOf -> create option classes
        option_names: List[str] = []
        option_defs: List[Tuple[Optional[str], str]] = []
        imports.add("from pydantic import BaseModel")

        # compute property frequency among object options to pick distinctive names
        obj_options = [opt for opt in schema_data["oneOf"] if isinstance(opt, dict) and opt.get("type") == "object" and "properties" in opt]
        prop_freq: Dict[str, int] = {}
        for o in obj_options:
            for pn in o.get("properties", {}).keys():
                prop_freq[pn] = prop_freq.get(pn, 0) + 1

        idx = 0
        for opt in schema_data["oneOf"]:
            idx += 1

            # handle merged allOf
            if isinstance(opt, dict) and "allOf" in opt and isinstance(opt["allOf"], list):
                merged_props = dict(properties or {})
                merged_required = set(required_fields or [])
                strict_opt = False
                title = opt.get("title")
                ref_name = None
                for entry in opt["allOf"]:
                    if not isinstance(entry, dict):
                        continue
                    if "$ref" in entry:
                        ref_name = pascal_case(entry["$ref"].split("/")[-1])
                        imports.add(f"from near_jsonrpc_models.{snake_case(ref_name)} import {ref_name}")
                    if entry.get("type") == "object" and "properties" in entry:
                        for k, v in entry.get("properties", {}).items():
                            merged_props[k] = v
                        merged_required.update(entry.get("required", []))
                    if entry.get("additionalProperties") is False:
                        strict_opt = True
                if isinstance(opt.get("properties"), dict):
                    for k, v in opt.get("properties", {}).items():
                        merged_props[k] = v
                merged_required.update(opt.get("required", []))

                # Priority for suffix: title -> single-name enum -> props -> fallback OptionN
                name_suffix = _extract_single_name_enum(merged_props)
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                    from_title = True
                elif name_suffix:
                    suffix = name_suffix
                    from_title = False
                else:
                    prop_order = list(merged_props.keys())
                    suffix = _make_suffix_from_props(merged_props, prop_order)
                    if not suffix:
                        suffix = f"Option{idx}"
                    # preserve version-style property names like 'V0' as title-derived (prevents numeric suffixing)
                    from_title = _prop_is_version_name(prop_order[0]) if len(prop_order) == 1 else False

                combined = to_class_name(pascal_case(schema_name), suffix)
                # If the combined name would collide with a referenced model inside props, append Option
                if _needs_option_suffix(combined, merged_props):
                    combined = f"{combined}Option"
                combined = _ensure_unique_name(combined, used_class_names, from_title=from_title)
                local_nested: List[str] = []
                merged_required_list = list(merged_required)
                class_def = _build_class_from_properties(combined, merged_props, merged_required_list,
                                                         ctx, imports, extra_defs=local_nested, strict=strict_opt)
                if ref_name:
                    if f"class {combined}(" in class_def:
                        lines = class_def.splitlines(True)
                        if lines:
                            first_line = lines[0]
                            if "(BaseModel):" in first_line:
                                lines[0] = first_line.replace("(BaseModel):", f"({ref_name}):")
                            elif "(StrictBaseModel):" in first_line:
                                lines[0] = first_line.replace("(StrictBaseModel):", f"({ref_name}):")
                            else:
                                lines[0] = re.sub(r"\(.*\):", f"({ref_name}):", lines[0])
                            class_def = "".join(lines)
                class_def = _inject_class_docstring(class_def, combined, opt.get("description"))
                option_defs.extend((None, d) for d in local_nested)
                option_defs.append((combined, class_def))
                option_names.append(combined)
                continue

            # string enum -> RootModel alias
            if isinstance(opt, dict) and opt.get("type") == "string" and isinstance(opt.get("enum"), list) and opt.get("enum"):
                vals = opt["enum"]
                raw = vals[0] if vals else f"Option{idx}"
                suffix = pascal_case(str(raw))
                alias_name = to_class_name(pascal_case(schema_name), suffix)
                alias_name = _ensure_unique_name(alias_name, used_class_names)
                imports.add("from pydantic import RootModel")
                imports.add("from typing import Literal")
                seen = set()
                dedup_vals = []
                for v in vals:
                    if v not in seen:
                        dedup_vals.append(v)
                        seen.add(v)
                lit_vals = ", ".join(repr(v) for v in dedup_vals)
                alias_def = (f'"""{opt.get("description")}"""\n' if opt.get("description") else "") + f"class {alias_name}(RootModel[Literal[{lit_vals}]]):\n    pass\n\n"
                option_defs.append((alias_name, alias_def))
                option_names.append(alias_name)
                continue

            # $ref option -> wrapper (prefer using the $ref name as suffix)
            if isinstance(opt, dict) and "$ref" in opt:
                ref = pascal_case(opt["$ref"].split("/")[-1])
                title = opt.get("title")
                # build merged props so we can produce descriptive suffix if needed
                merged_props = dict(properties or {})
                if isinstance(opt.get("properties"), dict):
                    for k, v in opt.get("properties", {}).items():
                        merged_props[k] = v
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))

                # Prefer using the ref name as suffix when available
                if ref:
                    suffix = ref
                    from_title = False
                else:
                    name_suffix = _extract_single_name_enum(merged_props)
                    if isinstance(title, str) and title.strip():
                        suffix = pascal_case(title)
                        from_title = True
                    elif name_suffix:
                        suffix = name_suffix
                        from_title = False
                    else:
                        prop_order = list(merged_props.keys())
                        suffix = _make_suffix_from_props(merged_props, prop_order)
                        if not suffix:
                            suffix = ref or f"Option{idx}"
                        from_title = _prop_is_version_name(prop_order[0]) if len(prop_order) == 1 else False

                combined = to_class_name(pascal_case(schema_name), suffix)
                if _needs_option_suffix(combined, merged_props):
                    combined = f"{combined}Option"
                combined = _ensure_unique_name(combined, used_class_names, from_title=from_title)
                imports.add(f"from near_jsonrpc_models.{snake_case(ref)} import {ref}")
                local_nested: List[str] = []
                strict_opt = opt.get("additionalProperties") is False
                class_def = _build_class_from_properties(combined, merged_props, merged_required, ctx, imports, extra_defs=local_nested, strict=strict_opt)
                header_base = f"class {combined}("
                if header_base in class_def:
                    lines = class_def.splitlines(True)
                    first = lines[0]
                    if "(BaseModel):" in first:
                        lines[0] = first.replace("(BaseModel):", f"({ref}):")
                    elif "(StrictBaseModel):" in first:
                        lines[0] = first.replace("(StrictBaseModel):", f"({ref}):")
                    else:
                        lines[0] = re.sub(r"\(.*\):", f"({ref}):", lines[0])
                    class_def = "".join(lines)
                class_def = _inject_class_docstring(class_def, combined, opt.get("description"))
                option_defs.extend((None, d) for d in local_nested)
                option_defs.append((combined, class_def))
                option_names.append(combined)
                continue

            # inline object option -> prefer title -> name enum -> props-derived suffix (keeps 'BlockIdShardId' instead of 'BlockShardId')
            if isinstance(opt, dict) and opt.get("type") == "object" and "properties" in opt:
                props = opt.get("properties", {})
                title = opt.get("title")
                # priority: title -> single-name enum -> props -> OptionN
                name_suffix = _extract_single_name_enum(props)
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                    from_title = True
                elif name_suffix:
                    suffix = name_suffix
                    from_title = False
                else:
                    prop_order = list(props.keys())
                    suffix = _make_suffix_from_props(props, prop_order)
                    if not suffix:
                        suffix = f"Option{idx}"
                    # preserve version-like single-property names
                    from_title = _prop_is_version_name(prop_order[0]) if len(prop_order) == 1 else False
                combined = to_class_name(pascal_case(schema_name), suffix)
                if _needs_option_suffix(combined, props):
                    combined = f"{combined}Option"
                combined = _ensure_unique_name(combined, used_class_names, from_title=from_title)
                local_nested: List[str] = []
                strict_opt = opt.get("additionalProperties") is False
                merged_props = dict(properties or {})
                for k, v in (props or {}).items():
                    merged_props[k] = v
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))
                class_def = _build_class_from_properties(combined, merged_props, merged_required, ctx, imports, extra_defs=local_nested, strict=strict_opt)
                class_def = _inject_class_docstring(class_def, combined, opt.get("description"))
                option_defs.extend((None, d) for d in local_nested)
                option_defs.append((combined, class_def))
                option_names.append(combined)
                continue

            # fallback
            combined = f"{pascal_case(schema_name)}Variant{idx}"
            combined = _ensure_unique_name(combined, used_class_names)
            fallback_def = f"class {combined}(BaseModel):\n    value: Any\n\n"
            fallback_def = _inject_class_docstring(fallback_def, combined, opt.get("description") if isinstance(opt, dict) else None)
            option_defs.append((combined, fallback_def))
            option_names.append(combined)
            imports.add("from typing import Any")

        imports.add("from typing import Union")
        imports.add("from pydantic import RootModel")
        import_block = "\n".join(sorted(imports))
        defs_code = ""
        for _name, d in option_defs:
            defs_code += d
        desc_block = f'"""{description}"""\n\n' if description else ""
        union_inner = ", ".join(option_names)
        wrapper_def = f"class {pascal_case(schema_name)}(RootModel[Union[{union_inner}]]):\n    pass\n\n"
        return f"{desc_block}{import_block}\n\n\n{defs_code}{wrapper_def}"

    # anyOf handling (prefer title then name-enum then props-based suffixes; when option is a $ref prefer schema+Ref naming)
    if "anyOf" in schema_data:
        opts = schema_data["anyOf"]
        parts: List[str] = []
        local_defs: List[str] = []
        need_conint = False
        need_condecimal = False
        need_literal = False
        used_aliases: Set[str] = set()
        idx = 0

        def make_alias_for_primitive(opt: Dict[str, Any], idx_local: int) -> Optional[str]:
            nonlocal need_conint, need_condecimal, need_literal
            if not isinstance(opt, dict):
                return None
            t = opt.get("type")
            title = opt.get("title")
            desc = opt.get("description") or ""
            # prefer title when available for primitives; otherwise OptionN
            if isinstance(title, str) and title.strip():
                suffix = pascal_case(title)
                from_title = True
            else:
                suffix = f"Option{idx_local}"
                from_title = False
            alias_candidate = to_class_name(pascal_case(schema_name), suffix)
            alias_candidate = _ensure_unique_name(alias_candidate, used_class_names, from_title=from_title)
            if alias_candidate in used_aliases:
                alias_candidate = _ensure_unique_name(alias_candidate, used_class_names)
            used_aliases.add(alias_candidate)

            if t == "integer":
                minimum = opt.get("minimum")
                maximum = opt.get("maximum")
                fmt = opt.get("format")
                if fmt == "int32" and minimum is None and maximum is None:
                    minimum = _INT32_MIN
                    maximum = _INT32_MAX
                if fmt in ("uint64", "unint64"):
                    if minimum is None:
                        minimum = 0
                    if maximum is None:
                        maximum = _UINT64_MAX
                if fmt == "uint32":
                    if minimum is None:
                        minimum = 0
                    if maximum is None:
                        maximum = _UINT32_MAX
                if fmt == "uint16":
                    if minimum is None:
                        minimum = 0
                    if maximum is None:
                        maximum = _UINT16_MAX
                if fmt == "uint8":
                    if minimum is None:
                        minimum = 0
                    if maximum is None:
                        maximum = _UINT8_MAX
                if fmt == "uint":
                    if minimum is None:
                        minimum = 0
                    if maximum is None:
                        maximum = _UINT32_MAX
                if minimum is not None or maximum is not None:
                    need_conint = True
                    args = []
                    if minimum is not None:
                        args.append(f"ge={minimum}")
                    if maximum is not None:
                        args.append(f"le={maximum}")
                    local_defs.append((f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[conint({', '.join(args)})]):\n    pass\n\n")
                else:
                    local_defs.append((f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[int]):\n    pass\n\n")
                return alias_candidate

            if t == "number":
                minimum = opt.get("minimum")
                maximum = opt.get("maximum")
                args = []
                if minimum is not None:
                    args.append(f"ge={minimum}")
                if maximum is not None:
                    args.append(f"le={maximum}")
                if args:
                    need_condecimal = True
                    local_defs.append((f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[condecimal({', '.join(args)})]):\n    pass\n\n")
                else:
                    local_defs.append((f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[float]):\n    pass\n\n")
                return alias_candidate

            if t == "string":
                if "enum" in opt and isinstance(opt["enum"], list) and opt["enum"]:
                    need_literal = True
                    vals = opt["enum"]
                    seen = set()
                    dedup_vals = []
                    for v in vals:
                        if v not in seen:
                            dedup_vals.append(v)
                            seen.add(v)
                    lit_vals = ", ".join(repr(v) for v in dedup_vals)
                    local_defs.append((f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[Literal[{lit_vals}]]):\n    pass\n\n")
                    return alias_candidate
                local_defs.append((f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[str]):\n    pass\n\n")
                return alias_candidate

            if t == "boolean":
                local_defs.append((f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[bool]):\n    pass\n\n")
                return alias_candidate

            return None

        for opt in opts:
            idx += 1
            if isinstance(opt, dict) and "$ref" in opt:
                ref = pascal_case(opt["$ref"].split("/")[-1])
                title = opt.get("title")
                # build merged_props and check for single-name enum
                merged_props = dict(properties or {})
                if isinstance(opt.get("properties"), dict):
                    for k, v in opt.get("properties", {}).items():
                        merged_props[k] = v

                name_suffix = _extract_single_name_enum(merged_props)
                # priority: prefer ref name -> title -> name enum -> props -> ref
                if ref:
                    suffix = ref
                    from_title = False
                elif isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                    from_title = True
                elif name_suffix:
                    suffix = name_suffix
                    from_title = False
                else:
                    prop_order = list(merged_props.keys())
                    suffix = _make_suffix_from_props(merged_props, prop_order)
                    if not suffix:
                        suffix = ref
                    # preserve version-like single-property names
                    from_title = _prop_is_version_name(prop_order[0]) if len(prop_order) == 1 else False

                alias_candidate = to_class_name(pascal_case(schema_name), suffix)
                if _needs_option_suffix(alias_candidate, merged_props):
                    alias_candidate = f"{alias_candidate}Option"
                alias_candidate = _ensure_unique_name(alias_candidate, used_class_names, from_title=from_title)
                if alias_candidate in used_aliases:
                    alias_candidate = _ensure_unique_name(alias_candidate, used_class_names)
                used_aliases.add(alias_candidate)
                if ref:
                    imports.add(f"from near_jsonrpc_models.{snake_case(ref)} import {ref}")
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))
                local_nested: List[str] = []
                class_def = _build_class_from_properties(alias_candidate, merged_props, merged_required, ctx, imports, extra_defs=local_nested, strict=opt.get("additionalProperties") is False)
                if f"class {alias_candidate}(" in class_def and ref:
                    lines = class_def.splitlines(True)
                    first = lines[0]
                    if "(BaseModel):" in first:
                        lines[0] = first.replace("(BaseModel):", f"({ref}):")
                    elif "(StrictBaseModel):" in first:
                        lines[0] = first.replace("(StrictBaseModel):", f"({ref}):")
                    else:
                        lines[0] = re.sub(r"\(.*\):", f"({ref}):", lines[0])
                    class_def = "".join(lines)
                class_def = _inject_class_docstring(class_def, alias_candidate, opt.get("description"))
                local_defs.extend(local_nested)
                local_defs.append(class_def)
                parts.append(alias_candidate)
                continue

            alias = make_alias_for_primitive(opt, idx)
            if alias:
                parts.append(alias)
                continue

            if isinstance(opt, dict) and opt.get("type") == "object" and "properties" in opt:
                title = opt.get("title")
                props = opt.get("properties", {})
                # priority: title -> single-name enum -> props -> OptionN
                name_suffix = _extract_single_name_enum(props)
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                    from_title = True
                elif name_suffix:
                    suffix = name_suffix
                    from_title = False
                else:
                    prop_order = list(props.keys())
                    suffix = _make_suffix_from_props(props, prop_order)
                    if not suffix:
                        suffix = f"Option{idx}"
                    # preserve version-like single-property names
                    from_title = _prop_is_version_name(prop_order[0]) if len(prop_order) == 1 else False
                inline_name = to_class_name(pascal_case(schema_name), suffix)
                if _needs_option_suffix(inline_name, props):
                    inline_name = f"{inline_name}Option"
                inline_name = _ensure_unique_name(inline_name, used_class_names, from_title=from_title)
                nested: List[str] = []
                strict_opt = opt.get("additionalProperties") is False
                merged_props = dict(properties or {})
                for k, v in (opt.get("properties") or {}).items():
                    merged_props[k] = v
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))
                class_def = _build_class_from_properties(inline_name, merged_props, merged_required, ctx, imports, extra_defs=nested, strict=strict_opt)
                local_defs.extend(nested)
                local_defs.append(class_def)
                parts.append(inline_name)
                continue

            imports.add("from typing import Any")
            parts.append("Any")

        if parts:
            imports.add("from typing import Union")
            if need_literal:
                imports.add("from typing import Literal")
            if need_conint:
                imports.add("from pydantic import conint")
            if need_condecimal:
                imports.add("from pydantic import condecimal")

            def primitive_like(p: str) -> bool:
                p = str(p)
                return (
                    p in ("int", "float", "str", "bool", "Any")
                    or p.startswith("conint")
                    or p.startswith("condecimal")
                    or p.startswith("Literal[")
                    or (p and p[0].isupper() and p != pascal_case(schema_name))
                )

            ordered: List[str] = []
            for p in parts:
                if primitive_like(p) and p not in ordered:
                    ordered.append(p)
            for p in parts:
                if p not in ordered:
                    ordered.append(p)

            union_inner = ", ".join(ordered)
            imports.add("from pydantic import RootModel")
            import_block = "\n".join(sorted(imports))
            desc_block = f'"""{description}"""\n\n' if description else ""
            pre_defs = "".join(local_defs + extra_class_defs)
            wrapper_def = f"class {pascal_case(schema_name)}(RootModel[Union[{union_inner}]]):\n    pass\n\n"
            return f"{desc_block}{import_block}\n\n\n{pre_defs}{wrapper_def}"

    # primitive alias handling (top-level primitive schemas)
    if isinstance(schema_data, dict) and schema_data.get("type") in {"string", "integer", "number", "boolean"}:
        t = schema_data.get("type")
        fmt = schema_data.get("format")
        desc = f'"""{description}"""\n\n' if description else ""
        imports.add("from pydantic import RootModel")
        if t == "integer":
            minimum = schema_data.get("minimum")
            maximum = schema_data.get("maximum")
            if fmt == "int32" and minimum is None and maximum is None:
                minimum = _INT32_MIN
                maximum = _INT32_MAX
            if fmt in ("uint64", "unint64"):
                if minimum is None:
                    minimum = 0
                if maximum is None:
                    maximum = _UINT64_MAX
            if fmt == "uint32":
                if minimum is None:
                    minimum = 0
                if maximum is None:
                    maximum = _UINT32_MAX
            if fmt == "uint16":
                if minimum is None:
                    minimum = 0
                if maximum is None:
                    maximum = _UINT16_MAX
            if fmt == "uint8":
                if minimum is None:
                    minimum = 0
                if maximum is None:
                    maximum = _UINT8_MAX
            if fmt == "uint":
                if minimum is None:
                    minimum = 0
                if maximum is None:
                    maximum = _UINT32_MAX
            if minimum is not None or maximum is not None:
                imports.add("from pydantic import conint")
                class_def = f"class {pascal_case(schema_name)}(RootModel[conint(ge={minimum}, le={maximum})]):\n    pass\n\n"
            else:
                class_def = f"class {pascal_case(schema_name)}(RootModel[int]):\n    pass\n\n"
            import_block = "\n".join(sorted(imports))
            return f"{desc}{import_block}\n\n\n{class_def}"
        if t == "number":
            imports.add("from pydantic import RootModel")
            args = []
            if schema_data.get("minimum") is not None:
                args.append(f"ge={schema_data['minimum']}")
            if schema_data.get("maximum") is not None:
                args.append(f"le={schema_data['maximum']}")
            if args:
                class_def = f"class {pascal_case(schema_name)}(RootModel[condecimal({', '.join(args)})]):\n    pass\n\n"
            else:
                class_def = f"class {pascal_case(schema_name)}(RootModel[condecimal()]):\n    pass\n\n"
            import_block = "\n".join(sorted(imports))
            return f"{import_block}\n\n\n{class_def}"
        if t == "string":
            if fmt == "bytes" or fmt == "byte":
                class_def = f"class {pascal_case(schema_name)}(RootModel[str]):\n    pass\n\n"
                import_block = "\n".join(sorted(imports))
                return f"{desc}{import_block}\n\n\n{class_def}"
            if schema_data.get("minLength") is not None or schema_data.get("maxLength") is not None or schema_data.get("pattern") is not None:
                imports.add("from pydantic import constr")
                args = []
                if schema_data.get("minLength") is not None:
                    args.append(f"min_length={schema_data['minLength']}")
                if schema_data.get("maxLength") is not None:
                    args.append(f"max_length={schema_data['maxLength']}")
                if schema_data.get("pattern") is not None:
                    args.append(f"regex={repr(schema_data['pattern'])}")
                class_def = f"class {pascal_case(schema_name)}(RootModel[constr({', '.join(args)})]):\n    pass\n\n"
            else:
                class_def = f"class {pascal_case(schema_name)}(RootModel[str]):\n    pass\n\n"
            import_block = "\n".join(sorted(imports))
            return f"{desc}{import_block}\n\n\n{class_def}"
        if t == "boolean":
            class_def = f"class {pascal_case(schema_name)}(RootModel[bool]):\n    pass\n\n"
            import_block = "\n".join(sorted(imports))
            return f"{desc}{import_block}\n\n\n{class_def}"

    # fallback: object -> BaseModel/StrictBaseModel
    imports.add("from pydantic import BaseModel")
    top_strict = isinstance(schema_data, dict) and schema_data.get("additionalProperties") is False
    if top_strict:
        imports.add("from near_jsonrpc_models.strict_model import StrictBaseModel")

    base = "StrictBaseModel" if top_strict else "BaseModel"
    class_body = f"class {pascal_case(schema_name)}({base}):\n"
    validators: List[str] = []
    need_validator_import = False
    need_field_import = False

    if not properties:
        class_body += "    pass\n"

    for field, fd in properties.items():
        _, ann = get_python_type_and_string(field, fd.get("type"), fd, ctx, imports, parent_name=pascal_case(schema_name), extra_defs=extra_class_defs)

        has_default = "default" in fd
        default = fd.get("default", None)
        desc_field = fd.get("description")
        is_req = field in required_fields
        if desc_field:
            class_body += f"    # {desc_field.replace(chr(10), chr(10) + '    # ')}\n"
        if fd.get("type") == "array":
            imports.add("from typing import List")

        nullable = _is_nullable(fd)

        if has_default and _is_mutable_default(default):
            imports.add("from pydantic import Field")
            need_field_import = True
            if fd.get("type") == "array" and isinstance(fd.get("items"), dict) and ("$ref" in fd.get("items") or ("allOf" in fd.get("items") and any(isinstance(e, dict) and "$ref" in e for e in fd.get("items")["allOf"]))):
                items = fd.get("items")
                if "$ref" in items:
                    ref_model = pascal_case(items["$ref"].split("/")[-1])
                else:
                    ref_model = None
                    for e in items.get("allOf", []):
                        if isinstance(e, dict) and "$ref" in e:
                            ref_model = pascal_case(e["$ref"].split("/")[-1])
                            break
                    ref_model = ref_model or ann
                imports.add(f"from near_jsonrpc_models.{snake_case(ref_model)} import {ref_model}")
                factory = f"lambda: [{ref_model}(**item) for item in {repr(default)}]"
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory={factory})\n"
                else:
                    annotate = ann if not nullable else ensure_optional_annotation(ann)
                    class_body += f"    {field}: {annotate} = Field(default_factory={factory})\n"
            elif _field_is_ref_like(fd):
                ctor = ann
                factory = f"lambda: {ctor}(**{repr(default)})"
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory={factory})\n"
                else:
                    annotate = ann if not nullable else ensure_optional_annotation(ann)
                    class_body += f"    {field}: {annotate} = Field(default_factory={factory})\n"
            else:
                factory = f"lambda: {repr(default)}"
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory={factory})\n"
                else:
                    annotate = ann if not nullable else ensure_optional_annotation(ann)
                    class_body += f"    {field}: {annotate} = Field(default_factory={factory})\n"
        else:
            if has_default and _field_is_ref_like(fd) and not _is_mutable_default(default):
                imports.add("from pydantic import Field")
                need_field_import = True
                ctor = ann
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory=lambda: {ctor}({repr(default)}))\n"
                else:
                    if default is None or nullable:
                        annotate = ensure_optional_annotation(ann)
                    else:
                        annotate = ann
                    class_body += f"    {field}: {annotate} = Field(default_factory=lambda: {ctor}({repr(default)}))\n"
            else:
                if is_req:
                    if has_default:
                        class_body += f"    {field}: {ann} = {repr(default)}\n"
                    else:
                        class_body += f"    {field}: {ann}\n"
                else:
                    if has_default:
                        if default is None:
                            annotate = ensure_optional_annotation(ann)
                            class_body += f"    {field}: {annotate} = {repr(default)}\n"
                        else:
                            annotate = ensure_optional_annotation(ann) if nullable else ann
                            class_body += f"    {field}: {annotate} = {repr(default)}\n"
                    else:
                        if nullable:
                            annotate = ensure_optional_annotation(ann)
                        else:
                            annotate = ann
                        class_body += f"    {field}: {annotate} = None\n"

        if fd.get("type") == "object" and "patternProperties" in fd:
            pattern = next(iter(fd["patternProperties"].keys()))
            imports.add("from pydantic import field_validator")
            need_validator_import = True
            validators.append(generate_pattern_properties_validator(field, pattern))

        if ann == "datetime":
            imports.add("from pydantic import field_validator")
            need_validator_import = True
            validators.append(generate_datetime_string_parser_validator(field))

    if validators:
        class_body += "\n"
        for v in validators:
            class_body += v

    if need_validator_import:
        imports.add("from pydantic import field_validator")
    if need_field_import:
        imports.add("from pydantic import Field")

    import_block = "\n".join(sorted(imports))
    pre_defs = "".join(extra_class_defs)
    model_code = ""
    if description:
        model_code += f'"""{description}"""\n\n'
    model_code += f"{import_block}\n\n\n{pre_defs}{class_body}"
    return model_code


# -------------------------
# IO helpers
# -------------------------
def save_model_to_file(model_code: str, schema_name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{snake_case(schema_name)}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(model_code)


def find_classes_and_aliases(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                if isinstance(node.value, ast.Name):
                    names.append(target)
    return names


def generate_models_init_py(models_dir: str) -> None:
    lines: List[str] = [
        "# GENERATED FILE — DO NOT EDIT MANUALLY",
        "# This file is re-generated by the model generator.",
        "from typing import TYPE_CHECKING",
        "",
    ]
    py_files = [f for f in os.listdir(models_dir) if f.endswith(".py") and f != "__init__.py"]
    all_classes: List[str] = []
    lines.append("if TYPE_CHECKING:")
    for py_file in py_files:
        mod_name = py_file[:-3]
        file_path = os.path.join(models_dir, py_file)
        classes = find_classes_and_aliases(file_path)
        all_classes.extend(classes)
        for cls in classes:
            lines.append(f"    from .{mod_name} import {cls}")
    lines.append("")
    lines.append("__all__ = [")
    for cls in sorted(all_classes):
        lines.append(f"    {cls!r},")
    lines.append("]")
    lines.append("")
    lines.append("_CLASS_TO_MODULE = {")
    for py_file in py_files:
        mod_name = py_file[:-3]
        classes = find_classes_and_aliases(os.path.join(models_dir, py_file))
        for cls in classes:
            lines.append(f"    {cls!r}: {mod_name!r},")
    lines.append("}")
    lines.append("")
    lines.append("def __getattr__(name: str):")
    lines.append("    if name in _CLASS_TO_MODULE:")
    lines.append("        import importlib")
    lines.append("        module_name = _CLASS_TO_MODULE[name]")
    lines.append("        module = importlib.import_module(f'.{module_name}', __name__)")
    lines.append("        value = getattr(module, name)")
    lines.append("        globals()[name] = value")
    lines.append("        return value")
    lines.append("    raise AttributeError(name)")
    lines.append("")
    lines.append("def __dir__():")
    lines.append("    return sorted(list(globals().keys()) + list(__all__))")
    lines.append("")
    os.makedirs(models_dir, exist_ok=True)
    init_path = os.path.join(models_dir, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Generated __init__.py with {len(all_classes)} classes/aliases.")


def delete_types(models_dir: str = "near_jsonrpc_models") -> None:
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename == 'strict_model.py':
                continue
            file_path = os.path.join(models_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
            elif os.path.isdir(file_path):
                print(f"Skipping dir {file_path}")
    else:
        print(f"Folder {models_dir} does not exist.")


def generate_models(ctx, models_dir):
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    for schema_name, schema_data in ctx.schemas.items():
        model_code = generate_model(schema_name, schema_data, ctx)
        save_model_to_file(model_code, schema_name, models_dir)
    generate_models_init_py(models_dir=models_dir)
    print(f"✅ Models generated successfully")
