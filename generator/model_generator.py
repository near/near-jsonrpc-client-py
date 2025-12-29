import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from generator.context import GeneratorContext
from generator.helpers.name_resolvers import snake_case, pascal_case, _make_safe_class_name

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
            # sometimes null is represented as {"type": "null"} or enum contains null
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


# helper to inject a class-level docstring into a generated class definition
def _inject_class_docstring(class_def: str, class_name: str, description: Optional[str]) -> str:
    if not description:
        return class_def

    # sanitize triple quotes in description
    desc_sanitized = description.replace('"""', '\\"\\"\\"')

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
# validator helpers
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
# helper: ensure unique class/alias names within generation context
# -------------------------
def _ensure_unique_name(candidate: str, used: Set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate
    base = candidate
    i = 1
    # try readable suffix first
    if not base.endswith("Option") and not base.endswith("Variant"):
        candidate = f"{base}Option"
        if candidate not in used:
            used.add(candidate)
            return candidate
    # then numeric suffixes
    while True:
        candidate = f"{base}{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


# -------------------------
# type resolution
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

    # --- special: allOf single-$ref -> return referenced model type ---
    if "allOf" in field_data and isinstance(field_data["allOf"], list):
        # if there's exactly one entry and it's a $ref, treat it as that ref
        if len(field_data["allOf"]) == 1 and isinstance(field_data["allOf"][0], dict) and "$ref" in field_data["allOf"][
            0]:
            ref = field_data["allOf"][0]["$ref"]
            model_name = pascal_case(ref.split("/")[-1])
            imports.add(f"from models.{snake_case(model_name)} import {model_name}")
            return model_name, model_name
        # otherwise, fall through — more complex allOf merges are not collapsed here

    # Inline enum -> Literal (string enums)
    if "enum" in field_data and field_data.get("type") == "string" and "$ref" not in field_data:
        imports.add("from typing import Literal")
        vals = field_data["enum"]
        # if single-member enum, return a Literal annotation
        if len(vals) == 1:
            return str, f"Literal[{repr(vals[0])}]"
        return str, "Literal[" + ", ".join(repr(v) for v in vals) + "]"

    # anyOf pattern nullable + $ref -> Model | None
    if "anyOf" in field_data:
        ref_opt = next((o for o in field_data["anyOf"] if isinstance(o, dict) and "$ref" in o), None)
        null_opt = next((o for o in field_data["anyOf"] if
                         isinstance(o, dict) and (o.get("enum") == [None] or o.get("nullable") is True)), None)
        if ref_opt and null_opt:
            name = pascal_case(ref_opt["$ref"].split("/")[-1])
            imports.add(f"from models.{snake_case(name)} import {name}")
            union = f"{name} | None"
            return union, union

    # oneOf collapse of single-member string-enums -> return combined Literal
    if "oneOf" in field_data:
        flattenable = True
        vals = []
        for opt in field_data["oneOf"]:
            if isinstance(opt, dict) and opt.get("type") == "string" and "enum" in opt and isinstance(opt["enum"],
                                                                                                      list):
                vals.extend(opt["enum"])
            else:
                flattenable = False
                break
        if flattenable and vals:
            # dedupe preserving order
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
        imports.add(f"from models.{snake_case(name)} import {name}")
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

        # helper to build conlist annotation when needed
        def build_container_annotation(item_annotation: str) -> str:
            # item_annotation expected like "Type" or "List[Type]" etc — use as-is inside conlist
            if min_items is None and max_items is None:
                return f"List[{item_annotation}]"
            # use conlist when there are bounds
            imports.add("from pydantic import conlist")
            args = []
            if min_items is not None:
                args.append(f"min_length={min_items}")
            if max_items is not None:
                args.append(f"max_length={max_items}")
            # join args with comma if both present
            args_str = ", ".join(args)
            return f"conlist({item_annotation}, {args_str})" if args_str else f"conlist({item_annotation})"

        if isinstance(items, dict) and items.get("type") == "object" and "properties" in items:
            root = parent_name if parent_name else pascal_case(field or "Anon")
            item_class = _make_safe_class_name(root, f"{field.capitalize()}Item")
            nested_defs: List[str] = []
            class_def = _build_class_from_properties(item_class, items.get("properties", {}), items.get("required", []),
                                                     ctx, imports, extra_defs=nested_defs,
                                                     strict=items.get("additionalProperties") is False)
            # place nested defs into extra_defs so caller will emit them
            extra_defs.extend(nested_defs)
            extra_defs.append(class_def)
            item_ann = item_class
            container_ann = build_container_annotation(item_ann)
            imports.add("from typing import List")
            return f"List[{item_class}]", container_ann
        if "$ref" in items:
            ref_model = pascal_case(items["$ref"].split("/")[-1])
            imports.add(f"from models.{snake_case(ref_model)} import {ref_model}")
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
                cls_name = _make_safe_class_name(root, field)

            nested_defs: List[str] = []
            # determine strictness for this object
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
# Accepts extra_defs list to collect nested/alias defs
# -------------------------
def _build_class_from_properties(class_name: str, props: Dict[str, Any], required_fields: List[str],
                                 ctx: GeneratorContext, imports: Set[str],
                                 extra_defs: Optional[List[str]] = None,
                                 strict: bool = False) -> str:
    if extra_defs is None:
        extra_defs = []
    base = "StrictBaseModel" if strict else "BaseModel"
    if strict:
        imports.add("from models.strict_model import StrictBaseModel")
    else:
        imports.add("from pydantic import BaseModel")
    body = f"class {class_name}({base}):\n"
    if not props:
        body += "    pass\n\n"
        return body
    for field, fd in props.items():
        # pass the same extra_defs so nested classes are accumulated in caller's list
        _, ann = get_python_type_and_string(field, fd.get("type"), fd, ctx, imports, parent_name=class_name,
                                            extra_defs=extra_defs)
        is_req = field in required_fields
        if fd.get("type") == "array":
            imports.add("from typing import List")

        # Determine if schema provides an explicit default (even if it is None)
        has_default = "default" in fd
        default = fd.get("default", None)

        # include field description as a comment if present
        desc_field = fd.get("description")
        if desc_field:
            body += f"    # {desc_field.replace(chr(10), chr(10) + '    # ')}\n"

        # determine explicit nullability according to schema
        nullable = _is_nullable(fd)

        if is_req:
            # required fields: keep exact annotation; include default if specified
            if has_default:
                body += f"    {field}: {ann} = {repr(default)}\n"
            else:
                body += f"    {field}: {ann}\n"
        else:
            # not required (field may be omitted) — produce sensible mapping:
            # - if schema is nullable -> Optional[...] with default (if provided) or = None
            # - elif has_default -> use concrete type (no Optional) and assign default
            # - else -> make it Optional[...] = None so omission is allowed
            if has_default:
                if default is None:
                    # default explicitly None -> we must allow None
                    annotate = ensure_optional_annotation(ann)
                    body += f"    {field}: {annotate} = {repr(default)}\n"
                else:
                    # explicit default and not None -> prefer concrete type (no Optional) unless schema is nullable
                    if nullable:
                        annotate = ensure_optional_annotation(ann)
                    else:
                        annotate = ann
                    body += f"    {field}: {annotate} = {repr(default)}\n"
            else:
                # no default provided: make field optional (omittable).
                # Use Optional[...] = None. This allows omission at runtime.
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

    # always include BaseModel import so generated files are consistent
    imports: Set[str] = {"from pydantic import BaseModel"}
    extra_class_defs: List[str] = []  # top-level extra defs (will be emitted after imports)

    # prepare a set of used class names (avoid collisions with other schemas)
    used_class_names: Set[str] = set(pascal_case(s) for s in ctx.schemas.keys())
    # ensure the current schema name is considered used
    used_class_names.add(pascal_case(schema_name))

    # ---- handle simple $ref or single-$ref allOf alias at top-level (e.g. "ChunkHash": {"$ref": "#/components/schemas/CryptoHash"}) ----
    # treat as an alias to the referenced model (preserve optional description)
    if isinstance(schema_data, dict):
        # direct $ref alias
        if "$ref" in schema_data and set(schema_data.keys()).issubset({"$ref", "description"}):
            ref = pascal_case(schema_data["$ref"].split("/")[-1])
            import_stmt = f"from models.{snake_case(ref)} import {ref}"
            desc_block = f'"""{description}"""\n\n' if description else ""
            return f"{desc_block}{import_stmt}\n\n\n{pascal_case(schema_name)} = {ref}\n"
        # allOf single-$ref alias
        if "allOf" in schema_data and isinstance(schema_data["allOf"], list) and len(
                schema_data["allOf"]) == 1 and isinstance(schema_data["allOf"][0], dict) and "$ref" in \
                schema_data["allOf"][0] and set(schema_data.keys()).issubset({"allOf", "description"}):
            ref = pascal_case(schema_data["allOf"][0]["$ref"].split("/")[-1])
            import_stmt = f"from models.{snake_case(ref)} import {ref}"
            desc_block = f'"""{description}"""\n\n' if description else ""
            return f"{desc_block}{import_stmt}\n\n\n{pascal_case(schema_name)} = {ref}\n"

    # ---- handle simple $ref alias at top-level (e.g. "ChunkHash": {"$ref": "#/components/schemas/CryptoHash"}) ----
    # treat as an alias to the referenced model (preserve optional description)
    if isinstance(schema_data, dict) and "$ref" in schema_data and set(schema_data.keys()).issubset(
            {"$ref", "description"}):
        ref = pascal_case(schema_data["$ref"].split("/")[-1])
        import_stmt = f"from models.{snake_case(ref)} import {ref}"
        desc_block = f'"""{description}"""\n\n' if description else ""
        return f"{desc_block}{import_stmt}\n\n\n{pascal_case(schema_name)} = {ref}\n"

    # top-level enum None -> generate RootModel wrapper over NoneType
    if schema_data.get("enum") == [None]:
        desc = f'"""{description}"""\n\n' if description else ""
        imports_local = {"from pydantic import RootModel", "from types import NoneType"}
        import_block = "\n".join(sorted(imports_local))
        class_def = f"class {pascal_case(schema_name)}(RootModel[NoneType]):\n    pass\n\n"
        return f"{desc}{import_block}\n\n\n{class_def}"

    # top-level string enum -> produce a pydantic RootModel wrapper over Literal[...] (not Python Enum)
    if "enum" in schema_data and schema_data.get("type") == "string":
        imports_needed = {"from pydantic import RootModel", "from typing import Literal"}
        enum_name = pascal_case(schema_name)
        vals = schema_data["enum"]
        lit = ", ".join(repr(v) for v in vals)
        desc_block = f'"""{description}"""\n\n' if description else ""
        import_block = "\n".join(sorted(imports_needed))
        class_def = f"class {enum_name}(RootModel[Literal[{lit}]]):\n    pass\n\n"
        return f"{desc_block}{import_block}\n\n\n{class_def}"

    # oneOf handling (string enums or complex)
    if "oneOf" in schema_data:
        # detect if all options are string enums (single- or multi-member)
        all_string_enums = True
        val_to_desc: Dict[str, Optional[str]] = {}
        for opt in schema_data["oneOf"]:
            if isinstance(opt, dict) and opt.get("type") == "string" and isinstance(opt.get("enum"), list) and opt.get(
                    "enum"):
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

        # complex oneOf -> create option classes (incl. nested defs) and a Union
        option_names: List[str] = []
        option_defs: List[str] = []
        imports.add("from pydantic import BaseModel")

        # Precompute property frequencies across object-style options to pick distinctive suffixes
        obj_options = [opt for opt in schema_data["oneOf"] if
                       isinstance(opt, dict) and opt.get("type") == "object" and "properties" in opt]
        prop_freq: Dict[str, int] = {}
        for o in obj_options:
            for pn in o.get("properties", {}).keys():
                prop_freq[pn] = prop_freq.get(pn, 0) + 1

        idx = 0
        for opt in schema_data["oneOf"]:
            idx += 1

            # Handle options that are expressed as allOf (merge their parts)
            if isinstance(opt, dict) and "allOf" in opt and isinstance(opt["allOf"], list):
                # Merge entries in allOf into a single inline-like option
                merged_props = dict(properties or {})
                merged_required = set(required_fields or [])
                strict_opt = False
                title = opt.get("title")
                ref_name = None
                # track any $ref imports we encounter (we'll import them but still emit a wrapper class)
                for entry in opt["allOf"]:
                    if not isinstance(entry, dict):
                        continue
                    if "$ref" in entry:
                        ref_name = pascal_case(entry["$ref"].split("/")[-1])
                        imports.add(f"from models.{snake_case(ref_name)} import {ref_name}")
                    if entry.get("type") == "object" and "properties" in entry:
                        for k, v in entry.get("properties", {}).items():
                            merged_props[k] = v
                        merged_required.update(entry.get("required", []))
                    if entry.get("additionalProperties") is False:
                        strict_opt = True

                # Merge option's own properties and required (fix for wrapper cases where opt has properties)
                if isinstance(opt.get("properties"), dict):
                    for k, v in opt.get("properties", {}).items():
                        merged_props[k] = v
                merged_required.update(opt.get("required", []))

                # Determine suffix/name for this merged option similar to inline object handling
                suffix = None
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                if suffix is None:
                    # try to use an enum value from a "name" prop if present
                    name_prop = None
                    for entry in opt["allOf"]:
                        if isinstance(entry, dict) and "properties" in entry:
                            name_prop = entry["properties"].get("name")
                            if isinstance(name_prop, dict) and name_prop.get("enum") and isinstance(name_prop["enum"],
                                                                                                    list):
                                suffix = pascal_case(name_prop["enum"][0])
                                break
                    # fallback: use most distinctive property
                if suffix is None and merged_props:
                    candidates = sorted(merged_props.keys(), key=lambda n: (prop_freq.get(n, 0), n))
                    if candidates:
                        suffix = pascal_case(candidates[0])
                suffix = suffix or f"Option{idx}"
                combined = _make_safe_class_name(pascal_case(schema_name), suffix)
                combined = _ensure_unique_name(combined, used_class_names)
                local_nested: List[str] = []
                merged_required_list = list(merged_required)
                class_def = _build_class_from_properties(combined, merged_props, merged_required_list,
                                                         ctx, imports, extra_defs=local_nested, strict=strict_opt)
                # If a $ref was present in the allOf, make the wrapper inherit from that ref
                if ref_name:
                    # replace header to inherit from the referenced model instead of BaseModel/StrictBaseModel
                    if f"class {combined}(" in class_def:
                        lines = class_def.splitlines(True)
                        if lines:
                            first_line = lines[0]
                            if "(BaseModel):" in first_line:
                                lines[0] = first_line.replace("(BaseModel):", f"({ref_name}):")
                            elif "(StrictBaseModel):" in first_line:
                                lines[0] = first_line.replace("(StrictBaseModel):", f"({ref_name}):")
                            else:
                                import re
                                lines[0] = re.sub(r"\(.*\):", f"({ref_name}):", lines[0])
                            class_def = "".join(lines)
                class_def = _inject_class_docstring(class_def, combined, opt.get("description"))
                option_defs.extend((None, d) for d in local_nested)
                option_defs.append((combined, class_def))
                option_names.append(combined)
                continue

            # string enum -> create RootModel(Literal[...]) alias class with descriptive name
            if isinstance(opt, dict) and opt.get("type") == "string" and isinstance(opt.get("enum"), list) and opt.get(
                    "enum"):
                # derive a readable alias from the enum values (prefer first)
                vals = opt["enum"]
                raw = vals[0] if vals else f"Option{idx}"
                suffix = pascal_case(str(raw))
                alias_name = _make_safe_class_name(pascal_case(schema_name), suffix)
                alias_name = _ensure_unique_name(alias_name, used_class_names)
                imports.add("from pydantic import RootModel")
                imports.add("from typing import Literal")
                # deduplicate preserving order
                seen = set()
                dedup_vals = []
                for v in vals:
                    if v not in seen:
                        dedup_vals.append(v)
                        seen.add(v)
                lit_vals = ", ".join(repr(v) for v in dedup_vals)
                alias_def = (f'"""{opt.get("description")}"""\n' if opt.get(
                    "description") else "") + f"class {alias_name}(RootModel[Literal[{lit_vals}]]):\n    pass\n\n"
                option_defs.append((alias_name, alias_def))
                option_names.append(alias_name)
                continue

            # $ref -> wrapper subclass of referenced model (and inject parent's props)
            if isinstance(opt, dict) and "$ref" in opt:
                ref = pascal_case(opt["$ref"].split("/")[-1])
                # prefer a descriptive wrapper name combining parent schema and referenced model (e.g. RpcQueryResponseAccountView)
                title = opt.get("title")
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                else:
                    suffix = ref
                combined = _make_safe_class_name(pascal_case(schema_name), suffix)
                combined = _ensure_unique_name(combined, used_class_names)
                imports.add(f"from models.{snake_case(ref)} import {ref}")
                local_nested: List[str] = []
                # try to detect strictness of the option wrapper (if explicitly set)
                strict_opt = opt.get("additionalProperties") is False
                # Merge top-level properties into wrapper so shared properties are present
                merged_props = dict(properties or {})
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))
                # If the option itself provides properties, merge them too (opt could contain both $ref and properties)
                if isinstance(opt.get("properties"), dict):
                    for k, v in opt.get("properties", {}).items():
                        merged_props[k] = v
                class_def = _build_class_from_properties(combined, merged_props, merged_required, ctx, imports,
                                                         extra_defs=local_nested, strict=strict_opt)
                # replace header to inherit from the referenced model instead of BaseModel/StrictBaseModel
                header_base = f"class {combined}("
                if header_base in class_def:
                    lines = class_def.splitlines(True)
                    if lines:
                        first_line = lines[0]
                        if "(BaseModel):" in first_line:
                            lines[0] = first_line.replace("(BaseModel):", f"({ref}):")
                        elif "(StrictBaseModel):" in first_line:
                            lines[0] = first_line.replace("(StrictBaseModel):", f"({ref}):")
                        else:
                            import re
                            lines[0] = re.sub(r"\(.*\):", f"({ref}):", lines[0])
                        class_def = "".join(lines)
                class_def = _inject_class_docstring(class_def, combined, opt.get("description"))
                option_defs.extend((None, d) for d in local_nested)
                option_defs.append((combined, class_def))
                option_names.append(combined)
                continue

            # inline object option -> choose a descriptive suffix derived from the option's properties (no hardcoded field list)
            if isinstance(opt, dict) and opt.get("type") == "object" and "properties" in opt:
                props = opt.get("properties", {})
                # prefer explicit title for naming; fallback to heuristics then OptionN
                title = opt.get("title")
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                else:
                    suffix = None

                if suffix is None:
                    # If name property has enum -> use its enum value (keeps descriptive value)
                    name_prop = props.get("name")
                    if isinstance(name_prop, dict) and name_prop.get("enum") and isinstance(name_prop["enum"], list):
                        suffix = pascal_case(name_prop["enum"][0])

                # If only one property present, use its property name
                if suffix is None and len(props) == 1:
                    prop_name = next(iter(props.keys()))
                    suffix = pascal_case(prop_name)

                # Otherwise pick the most distinctive property name (lowest frequency across object options)
                if suffix is None and props:
                    # compute candidates sorted by (freq, name) to be deterministic: prefer lower freq, then lexicographic
                    candidates = sorted(props.keys(), key=lambda n: (prop_freq.get(n, 0), n))
                    if candidates:
                        suffix = pascal_case(candidates[0])

                suffix = suffix or f"Option{idx}"
                combined = _make_safe_class_name(pascal_case(schema_name), suffix)
                combined = _ensure_unique_name(combined, used_class_names)
                local_nested: List[str] = []
                strict_opt = opt.get("additionalProperties") is False

                # Merge top-level properties with option-specific properties so shared properties appear in each option class.
                merged_props = dict(properties or {})
                for k, v in (props or {}).items():
                    merged_props[k] = v
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))

                class_def = _build_class_from_properties(combined, merged_props, merged_required,
                                                         ctx, imports, extra_defs=local_nested, strict=strict_opt)
                # inject description for this inline option wrapper class
                class_def = _inject_class_docstring(class_def, combined, opt.get("description"))
                option_defs.extend((None, d) for d in local_nested)
                option_defs.append((combined, class_def))
                option_names.append(combined)
                continue

            # fallback wrapper (use Variant suffix to avoid generic OptionN names)
            combined = f"{pascal_case(schema_name)}Variant{idx}"
            combined = _ensure_unique_name(combined, used_class_names)
            # fallback uses BaseModel — we don't know strictness here
            fallback_def = f"class {combined}(BaseModel):\n    value: Any\n\n"
            # inject description if provided
            fallback_def = _inject_class_docstring(fallback_def, combined,
                                                   opt.get("description") if isinstance(opt, dict) else None)
            option_defs.append((combined, fallback_def))
            option_names.append(combined)
            imports.add("from typing import Any")

        imports.add("from typing import Union")
        # add RootModel import so unions are wrapped as Pydantic RootModel classes
        imports.add("from pydantic import RootModel")
        import_block = "\n".join(sorted(imports))
        defs_code = ""
        for _name, d in option_defs:
            defs_code += d
        desc_block = f'"""{description}"""\n\n' if description else ""
        union_inner = ", ".join(option_names)
        # emit a RootModel wrapper for the Union
        wrapper_def = f"class {pascal_case(schema_name)}(RootModel[Union[{union_inner}]]):\n    pass\n\n"
        return f"{desc_block}{import_block}\n\n\n{defs_code}{wrapper_def}"

    # anyOf handling (merge top-level properties into each option and produce Union of merged variants)
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

            # alias name from schemaName + title (preferred) or Option{idx}
            if isinstance(title, str) and title.strip():
                suffix = pascal_case(title)
            else:
                suffix = f"Option{idx_local}"
            alias_candidate = _make_safe_class_name(pascal_case(schema_name), suffix)
            alias_candidate = _ensure_unique_name(alias_candidate, used_class_names)
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
                    local_defs.append(
                        (
                            f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[conint({', '.join(args)})]):\n    pass\n\n"
                    )
                else:
                    local_defs.append(
                        (f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[int]):\n    pass\n\n")
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
                    local_defs.append(
                        (
                            f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[condecimal({', '.join(args)})]):\n    pass\n\n")
                else:
                    local_defs.append((
                                          f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[float]):\n    pass\n\n")
                return alias_candidate

            if t == "string":
                # if enum present -> produce RootModel(Literal[...]) alias (descriptive name)
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
                    local_defs.append(
                        (
                            f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[Literal[{lit_vals}]]):\n    pass\n\n"
                    )
                    return alias_candidate
                local_defs.append(
                    (f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[str]):\n    pass\n\n")
                return alias_candidate

            if t == "boolean":
                local_defs.append(
                    (f'"""{desc}"""\n' if desc else "") + f"class {alias_candidate}(RootModel[bool]):\n    pass\n\n")
                return alias_candidate

            return None

        for opt in opts:
            idx += 1
            # $ref option -> create a merged wrapper class that inherits the referenced model (if possible)
            if isinstance(opt, dict) and "$ref" in opt:
                ref = pascal_case(opt["$ref"].split("/")[-1])
                title = opt.get("title")
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                else:
                    suffix = ref
                alias_candidate = _make_safe_class_name(pascal_case(schema_name), suffix)
                alias_candidate = _ensure_unique_name(alias_candidate, used_class_names)
                if alias_candidate in used_aliases:
                    alias_candidate = _ensure_unique_name(alias_candidate, used_class_names)
                used_aliases.add(alias_candidate)

                imports.add(f"from models.{snake_case(ref)} import {ref}")
                # Build merged properties class and then change its parent to inherit from the referenced model
                merged_props = dict(properties or {})
                if isinstance(opt.get("properties"), dict):
                    for k, v in opt.get("properties", {}).items():
                        merged_props[k] = v
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))
                local_nested: List[str] = []
                class_def = _build_class_from_properties(alias_candidate, merged_props, merged_required, ctx, imports,
                                                         extra_defs=local_nested,
                                                         strict=opt.get("additionalProperties") is False)
                # swap base to referenced model
                if f"class {alias_candidate}(" in class_def:
                    lines = class_def.splitlines(True)
                    first = lines[0]
                    if "(BaseModel):" in first:
                        lines[0] = first.replace("(BaseModel):", f"({ref}):")
                    elif "(StrictBaseModel):" in first:
                        lines[0] = first.replace("(StrictBaseModel):", f"({ref}):")
                    else:
                        import re
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
                # Inline object option — merge top-level properties and option properties into a new class
                title = opt.get("title")
                if isinstance(title, str) and title.strip():
                    suffix = pascal_case(title)
                else:
                    suffix = f"Option{idx}"
                inline_name = _make_safe_class_name(pascal_case(schema_name), suffix)
                inline_name = _ensure_unique_name(inline_name, used_class_names)
                nested: List[str] = []
                strict_opt = opt.get("additionalProperties") is False

                merged_props = dict(properties or {})
                for k, v in (opt.get("properties") or {}).items():
                    merged_props[k] = v
                merged_required = list(set(required_fields or []) | set(opt.get("required", [])))

                class_def = _build_class_from_properties(inline_name, merged_props, merged_required, ctx, imports,
                                                         extra_defs=nested, strict=strict_opt)
                local_defs.extend(nested)
                local_defs.append(class_def)
                parts.append(inline_name)
                continue

            # fallback: unknown variant -> use Any
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

            # smart ordering:aliases/constrained types first
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
            # add RootModel import so unions are wrapped as Pydantic RootModel classes
            imports.add("from pydantic import RootModel")
            import_block = "\n".join(sorted(imports))
            desc_block = f'"""{description}"""\n\n' if description else ""
            pre_defs = "".join(local_defs + extra_class_defs)
            # emit a RootModel wrapper for the Union
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
            imports.add("from pydantic import condecimal")
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

            if schema_data.get("minLength") is not None or schema_data.get("maxLength") is not None or schema_data.get(
                    "pattern") is not None:
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

    # fallback: generate BaseModel / StrictBaseModel class depending on top-level additionalProperties
    imports.add("from pydantic import BaseModel")
    top_strict = isinstance(schema_data, dict) and schema_data.get("additionalProperties") is False
    if top_strict:
        imports.add("from models.strict_model import StrictBaseModel")

    base = "StrictBaseModel" if top_strict else "BaseModel"
    class_body = f"class {pascal_case(schema_name)}({base}):\n"
    validators: List[str] = []
    need_validator_import = False
    need_field_import = False

    if not properties:
        class_body += "    pass\n"

    for field, fd in properties.items():
        _, ann = get_python_type_and_string(field, fd.get("type"), fd, ctx, imports,
                                            parent_name=pascal_case(schema_name), extra_defs=extra_class_defs)

        # Determine if schema provides an explicit default (even if it is None)
        has_default = "default" in fd
        default = fd.get("default", None)

        desc_field = fd.get("description")
        is_req = field in required_fields
        if desc_field:
            class_body += f"    # {desc_field.replace(chr(10), chr(10) + '    # ')}\n"
        if fd.get("type") == "array":
            imports.add("from typing import List")

        # Determine explicit nullability
        nullable = _is_nullable(fd)

        # Handle mutable defaults (lists/dicts) with Field(default_factory=...)
        if has_default and _is_mutable_default(default):
            imports.add("from pydantic import Field")
            need_field_import = True

            # If the field references a model (direct $ref/allOf-$ref) or is an array of $ref items,
            # construct instances in the default factory instead of returning raw dict/list.
            if fd.get("type") == "array" and isinstance(fd.get("items"), dict) and ("$ref" in fd.get("items") or (
                    "allOf" in fd.get("items") and any(
                isinstance(e, dict) and "$ref" in e for e in fd.get("items")["allOf"]))):
                # list of model instances
                items = fd.get("items")
                if "$ref" in items:
                    ref_model = pascal_case(items["$ref"].split("/")[-1])
                else:
                    # find first $ref in allOf
                    ref_model = None
                    for e in items.get("allOf", []):
                        if isinstance(e, dict) and "$ref" in e:
                            ref_model = pascal_case(e["$ref"].split("/")[-1])
                            break
                    ref_model = ref_model or ann
                imports.add(f"from models.{snake_case(ref_model)} import {ref_model}")
                factory = f"lambda: [{ref_model}(**item) for item in {repr(default)}]"
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory={factory})\n"
                else:
                    # if not required, but has a mutable default, we still emit optional annotation if schema says nullable
                    annotate = ann if not nullable else ensure_optional_annotation(ann)
                    class_body += f"    {field}: {annotate} = Field(default_factory={factory})\n"
            elif _field_is_ref_like(fd):
                # single model instance from dict default
                ctor = ann
                factory = f"lambda: {ctor}(**{repr(default)})"
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory={factory})\n"
                else:
                    annotate = ann if not nullable else ensure_optional_annotation(ann)
                    class_body += f"    {field}: {annotate} = Field(default_factory={factory})\n"
            else:
                # plain mutable primitive default (list/dict/set) — keep literal
                factory = f"lambda: {repr(default)}"
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory={factory})\n"
                else:
                    annotate = ann if not nullable else ensure_optional_annotation(ann)
                    class_body += f"    {field}: {annotate} = Field(default_factory={factory})\n"

        else:
            # non-mutable default handling (including when schema explicitly sets default=None)
            if has_default and _field_is_ref_like(fd) and not _is_mutable_default(default):
                imports.add("from pydantic import Field")
                need_field_import = True
                ctor = ann
                if is_req:
                    class_body += f"    {field}: {ann} = Field(default_factory=lambda: {ctor}({repr(default)}))\n"
                else:
                    # choose annotation: if default is None -> optional; else if schema nullable -> optional; else concrete
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
                    # not required
                    if has_default:
                        if default is None:
                            annotate = ensure_optional_annotation(ann)
                            class_body += f"    {field}: {annotate} = {repr(default)}\n"
                        else:
                            # default exists and not None: if schema nullable -> optional else concrete
                            annotate = ensure_optional_annotation(ann) if nullable else ann
                            class_body += f"    {field}: {annotate} = {repr(default)}\n"
                    else:
                        # no default provided: make field omittable by emitting Optional[...] = None
                        if nullable:
                            annotate = ensure_optional_annotation(ann)
                        else:
                            annotate = ann
                        class_body += f"    {field}: {annotate} = None\n"

        # patternProperties validator
        if fd.get("type") == "object" and "patternProperties" in fd:
            pattern = next(iter(fd["patternProperties"].keys()))
            imports.add("from pydantic import field_validator")
            need_validator_import = True
            validators.append(generate_pattern_properties_validator(field, pattern))

        # datetime validator
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

    # assemble file: import block first, then any extra_class_defs (nested / aliases), then the main class
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
def save_model_to_file(model_code: str, schema_name: str, out_dir: str = "models") -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{snake_case(schema_name)}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(model_code)


def generate_models_init_py(ctx: GeneratorContext, models_dir: str = "models") -> None:
    schema_names = sorted(ctx.schemas.keys())
    lines: List[str] = []
    lines.append("# GENERATED FILE — DO NOT EDIT MANUALLY")
    lines.append("# This file is re-generated by the model generator.")
    lines.append("")
    lines.append("from typing import TYPE_CHECKING")
    lines.append("")
    lines.append("if TYPE_CHECKING:")
    for s in schema_names:
        cls = pascal_case(s)
        mod = snake_case(s)
        lines.append(f"    from .{mod} import {cls}  # pragma: no cover")
    lines.append("")
    lines.append("__all__ = [")
    for s in schema_names:
        lines.append(f"    {pascal_case(s)!r},")
    lines.append("]")
    lines.append("")
    lines.append("_CLASS_TO_MODULE = {")
    for s in schema_names:
        lines.append(f"    {pascal_case(s)!r}: {snake_case(s)!r},")
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


def delete_types(models_dir: str = "models") -> None:
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


def generate_models(ctx, models_dir="models"):
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # generate each model
    for schema_name, schema_data in ctx.schemas.items():
        model_code = generate_model(schema_name, schema_data, ctx)
        save_model_to_file(model_code, schema_name)

    # generate __init__.py
    generate_models_init_py(ctx, models_dir=models_dir)

    print(f"✅ Models generated successfully")
