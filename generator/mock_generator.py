# mock_generator.py
from pathlib import Path
import json
import random
from typing import Any, Dict, List, Optional, Set, Union

from generator.helpers.name_resolvers import snake_case

MAX_DEPTH = 30


class MockGenerator:
    @staticmethod
    def generate(ctx, target_dirs: List[Union[str, Path]]) -> None:
        schemas = getattr(ctx, "schemas", None) or {}
        if not schemas:
            print("No schemas found in GeneratorContext")
            return

        for sample_dir in target_dirs:
            clean_sample_json_files(sample_dir)

        target_paths = [Path(d) for d in target_dirs]
        for p in target_paths:
            p.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = 0
        for schema_name in sorted(schemas.keys()):
            try:
                sample = MockGenerator.generate_sample_for_schema_name(schema_name, ctx)
                file_name = f"{snake_case(schema_name)}.json"
                text = json.dumps(sample if sample is not None else None, indent=2, ensure_ascii=False)
                for dirp in target_paths:
                    (dirp / file_name).write_text(text, encoding="utf-8")
                success += 1
            except Exception as e:
                print(f"❌ Failed to generate {schema_name} test: {e}")
                failed += 1

        print(f"✨ Mock generation complete! success={success} failed={failed}")
        for d in target_paths:
            print(f" - {d.resolve()}")

    @staticmethod
    def generate_sample_for_schema_name(schema_name: str, ctx) -> Any:
        schemas = getattr(ctx, "schemas", {}) or {}
        schema = schemas.get(schema_name)
        if schema is None:
            return None
        return MockGenerator.generate_sample(schema, ctx.spec, 0, set())

    @staticmethod
    def resolve_ref(ref: Optional[str], spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if ref is None:
            return None
        prefix = "#/components/schemas/"
        name = ref[len(prefix):] if ref.startswith(prefix) else ref
        return (spec.get("components") or {}).get("schemas", {}).get(name)

    @staticmethod
    def resolve_ref_deep(schema: Optional[Dict[str, Any]], spec: Dict[str, Any], seen: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Resolve $ref chain until a non-$ref schema is reached.
        Safely handles non-dict schema inputs by returning them unchanged.
        """
        if schema is None:
            return None
        if not isinstance(schema, dict):
            return schema

        seen = set() if seen is None else set(seen)
        cur = schema
        while True:
            r = cur.get("$ref") or cur.get("ref")
            if not r or r in seen:
                break
            seen.add(r)
            next_schema = MockGenerator.resolve_ref(r, spec)
            if not next_schema:
                break
            if not isinstance(next_schema, dict):
                return next_schema
            cur = next_schema
        return cur

    @staticmethod
    def schema_accepts_null(schema: Optional[Dict[str, Any]], spec: Dict[str, Any], seen: Optional[Set[str]] = None) -> bool:
        """
        Return True if the (resolved) schema allows a null value.
        Checks:
          - type == "null"
          - nullable == True
          - enum contains None
          - anyOf/oneOf/allOf variants allow null
        Safely handles non-dict schema inputs.
        """
        if schema is None:
            return False
        if not isinstance(schema, dict):
            return False

        seen = set() if seen is None else set(seen)
        resolved = MockGenerator.resolve_ref_deep(schema, spec, seen)

        if not isinstance(resolved, dict):
            return False

        # direct checks
        if resolved.get("type") == "null":
            return True
        if resolved.get("nullable") is True:
            return True
        enum_v = resolved.get("enum")
        if isinstance(enum_v, list) and any(v is None for v in enum_v):
            return True

        # check variants
        for key in ("oneOf", "anyOf", "allOf"):
            variants = resolved.get(key) or []
            for v in variants:
                rv = MockGenerator.resolve_ref_deep(v, spec, set(seen))
                if MockGenerator.schema_accepts_null(rv, spec, set(seen)):
                    return True
        return False

    @staticmethod
    def generate_sample(schema: Optional[Dict[str, Any]], spec: Dict[str, Any], depth: int, seen: Set[str]) -> Any:
        if schema is None:
            return None
        if depth > MAX_DEPTH:
            return None

        # if schema itself is not a dict (e.g., items is a list element like {"$ref": ...}),
        # keep proceeding but guard in helper functions.
        resolved = MockGenerator.resolve_ref_deep(schema, spec, set(seen))

        # default
        if isinstance(schema, dict) and "default" in schema:
            return schema["default"]

        # allOf
        if isinstance(resolved, dict) and resolved.get("allOf"):
            return MockGenerator.generate_sample_all_of(resolved["allOf"], resolved, spec, depth + 1, set(seen))

        # oneOf / anyOf
        variants = (resolved.get("oneOf") if isinstance(resolved, dict) else None) or (resolved.get("anyOf") if isinstance(resolved, dict) else None)
        if variants:
            return MockGenerator.generate_sample_one_of_any_of(variants, resolved if isinstance(resolved, dict) else schema, spec, depth + 1, set(seen))

        # enum
        if isinstance(resolved, dict) and resolved.get("enum"):
            enum_list = resolved.get("enum")
            if enum_list:
                if any(v is None for v in enum_list):
                    return None
                return enum_list[0]

        # type null
        if isinstance(resolved, dict) and resolved.get("type") == "null":
            return None

        # array
        if (isinstance(resolved, dict) and resolved.get("type") == "array") or (isinstance(resolved, dict) and "items" in resolved):
            items = resolved.get("items") if isinstance(resolved, dict) else None
            # items may be dict or list (tuple-style)
            size = 1
            if isinstance(resolved, dict) and resolved.get("minItems") is not None:
                try:
                    size = max(0, int(resolved["minItems"]))
                except Exception:
                    size = 1
            elif isinstance(resolved, dict) and resolved.get("maxItems") is not None:
                try:
                    size = max(1, int(resolved["maxItems"]))
                except Exception:
                    size = 1

            result = []

            # tuple-style: items is a list of schemas
            if isinstance(items, list):
                for i in range(size):
                    if i < len(items):
                        sub_schema = items[i]
                    else:
                        # if requested size is larger than items list, reuse last schema
                        sub_schema = items[-1]
                    elem = MockGenerator.generate_sample(sub_schema, spec, depth + 1, set(seen))
                    if elem is None:
                        if MockGenerator.schema_accepts_null(sub_schema, spec, set(seen)):
                            result.append(None)
                        else:
                            result.append(MockGenerator.fallback_value(sub_schema))
                    else:
                        result.append(elem)
                return result

            # homogeneous array: items is a dict (or missing)
            if isinstance(items, dict):
                for i in range(size):
                    elem = MockGenerator.generate_sample(items, spec, depth + 1, set(seen))
                    if elem is None:
                        if MockGenerator.schema_accepts_null(items, spec, set(seen)):
                            result.append(None)
                        else:
                            result.append(MockGenerator.fallback_value(items))
                    else:
                        result.append(elem)
                return result

            # fallback: unknown items shape
            for i in range(size):
                result.append(None)
            return result

        # object-like
        is_object = isinstance(resolved, dict) and (
            (resolved.get("type") == "object")
            or ("properties" in resolved)
            or ("additionalProperties" in resolved)
            or ("patternProperties" in resolved)
        )
        if is_object:
            props = resolved.get("properties") or {}
            obj: Dict[str, Any] = {}
            for prop_name, prop_schema in props.items():
                child = MockGenerator.generate_sample(prop_schema, spec, depth + 1, set(seen))
                if child is None:
                    if MockGenerator.schema_accepts_null(prop_schema, spec, set(seen)):
                        obj[prop_name] = None
                    else:
                        obj[prop_name] = MockGenerator.fallback_value(prop_schema)
                else:
                    obj[prop_name] = child

            addp = resolved.get("additionalProperties")
            if isinstance(addp, dict):
                add_val = MockGenerator.generate_sample(addp, spec, depth + 1, set(seen))
                if add_val is None and MockGenerator.schema_accepts_null(addp, spec, set(seen)):
                    obj["additionalProp1"] = None
                else:
                    obj["additionalProp1"] = add_val if add_val is not None else MockGenerator.fallback_value(addp)

            pattern_props = resolved.get("patternProperties") or {}
            for pattern, pat_schema in pattern_props.items():
                if pattern == r"^\d+$":
                    key = str(random.randint(0, 999))
                elif pattern == r"^[a-zA-Z_][a-zA-Z0-9_]*$":
                    key = random.choice(["key_a", "item_1", "field_x"])
                else:
                    key = f"generated_key_{abs(hash(pattern)) % 1000}"
                v = MockGenerator.generate_sample(pat_schema, spec, depth + 1, set(seen))
                if v is None:
                    if MockGenerator.schema_accepts_null(pat_schema, spec, set(seen)):
                        obj[key] = None
                    else:
                        obj[key] = "s"
                else:
                    obj[key] = v
            return obj

        # primitives
        t = resolved.get("type") if isinstance(resolved, dict) else None
        if t == "string":
            fmt = resolved.get("format")
            if fmt == "date-time":
                return "2021-01-01T00:00:00Z"
            if fmt == "date":
                return "2021-01-01"
            if fmt == "uuid":
                return "00000000-0000-0000-0000-000000000000"
            if fmt == "email":
                return "user@example.com"
            if fmt == "uri":
                return "https://example.com"
            return "s"
        if t == "integer":
            minv = resolved.get("minimum") if isinstance(resolved, dict) else None
            try:
                if minv is not None:
                    return int(minv)
            except Exception:
                pass
            return 0
        if t == "number":
            minv = resolved.get("minimum") if isinstance(resolved, dict) else None
            try:
                if minv is not None:
                    return float(minv)
            except Exception:
                pass
            return 0.0
        if t == "boolean":
            return True

        if "const" in (resolved if isinstance(resolved, dict) else {}):
            return resolved["const"]

        return None

    @staticmethod
    def generate_sample_all_of(all_list: List[Dict[str, Any]], parent_schema: Dict[str, Any], spec: Dict[str, Any], depth: int, seen: Set[str]) -> Any:
        merged_from_subs: Dict[str, Any] = {}
        primitive_from_subs = None
        for sub in all_list:
            resolved_sub = MockGenerator.resolve_ref_deep(sub, spec, set(seen))
            sample = MockGenerator.generate_sample(resolved_sub, spec, depth + 1, set(seen))
            if sample is not None and not isinstance(sample, dict):
                primitive_from_subs = sample
            if isinstance(sample, dict):
                merged_from_subs.update(sample)

        parent_props = parent_schema.get("properties") or {}
        merged_parent_props: Dict[str, Any] = {}
        for pname, pschema in parent_props.items():
            val = MockGenerator.generate_sample(pschema, spec, depth + 1, set(seen))
            if val is None:
                if MockGenerator.schema_accepts_null(pschema, spec, set(seen)):
                    merged_parent_props[pname] = None
                else:
                    merged_parent_props[pname] = MockGenerator.fallback_value(pschema)
            else:
                merged_parent_props[pname] = val

        if not merged_from_subs and not merged_parent_props and primitive_from_subs is not None:
            return primitive_from_subs

        final: Dict[str, Any] = {}
        final.update(merged_from_subs)
        final.update(merged_parent_props)
        if final:
            return final
        else:
            return {}

    @staticmethod
    def generate_sample_one_of_any_of(variants: List[Dict[str, Any]], parent: Dict[str, Any], spec: Dict[str, Any], depth: int, seen: Set[str]) -> Any:
        def score(v: Dict[str, Any]) -> int:
            s = 0
            if v.get("$ref") or v.get("ref"):
                s += 10
            if v.get("enum"):
                s += 5
            if v.get("properties") or v.get("allOf") or v.get("oneOf"):
                s += 2
            return s

        ordered = sorted(variants, key=score, reverse=True)
        parent_props = parent.get("properties") or {}

        for v in ordered:
            resolved = MockGenerator.resolve_ref_deep(v, spec, set(seen))
            if isinstance(resolved, dict) and resolved.get("enum"):
                if any(x is None for x in resolved.get("enum")):
                    sample = None
                else:
                    sample = resolved["enum"][0]
            else:
                sample = MockGenerator.generate_sample(resolved, spec, depth + 1, set(seen))

            if sample is None:
                if not parent_props:
                    if MockGenerator.schema_accepts_null(v, spec, set(seen)):
                        return None
                    continue
                merged: Dict[str, Any] = {}
                for pn, ps in parent_props.items():
                    pv = MockGenerator.generate_sample(ps, spec, depth + 1, set(seen))
                    if pv is None:
                        if MockGenerator.schema_accepts_null(ps, spec, set(seen)):
                            merged[pn] = None
                        else:
                            merged[pn] = MockGenerator.fallback_value(ps)
                    else:
                        merged[pn] = pv
                merged["_variant"] = None
                return merged

            if not isinstance(sample, dict):
                if not parent_props:
                    return sample
                merged = {}
                for pn, ps in parent_props.items():
                    pv = MockGenerator.generate_sample(ps, spec, depth + 1, set(seen))
                    if pv is None:
                        if MockGenerator.schema_accepts_null(ps, spec, set(seen)):
                            merged[pn] = None
                        else:
                            merged[pn] = MockGenerator.fallback_value(ps)
                    else:
                        merged[pn] = pv
                merged["_variant"] = sample
                return merged

            if isinstance(sample, dict) and sample:
                merged = {}
                for pn, ps in parent_props.items():
                    pv = MockGenerator.generate_sample(ps, spec, depth + 1, set(seen))
                    if pv is None:
                        if MockGenerator.schema_accepts_null(ps, spec, set(seen)):
                            merged[pn] = None
                        else:
                            merged[pn] = MockGenerator.fallback_value(ps)
                    else:
                        merged[pn] = pv
                merged.update(sample)
                return merged

        # fallback: merge first variant with parent props
        first = MockGenerator.resolve_ref_deep(variants[0], spec, set(seen))
        merged_props: Dict[str, Any] = {}
        merged_props.update(parent.get("properties") or {})
        if isinstance(first, dict):
            merged_props.update(first.get("properties") or {})
        synthetic = {"type": "object", "properties": merged_props}
        fallback = MockGenerator.generate_sample(synthetic, spec, depth + 1, set(seen))
        return fallback if fallback is not None else {}

    @staticmethod
    def fallback_value(schema: Optional[Dict[str, Any]]) -> Any:
        if not schema:
            return None
        # if schema is not a dict, try to resolve basic ref/null handling
        if not isinstance(schema, dict):
            return None
        t = schema.get("type")
        if t == "array":
            return []
        if t == "object":
            return {}
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        if t == "boolean":
            return True
        if t == "null":
            return None
        return "s"


def clean_sample_json_files(samples_dir: Path) -> None:
    if not samples_dir.exists() or not samples_dir.is_dir():
        return
    deleted = 0
    for file in samples_dir.iterdir():
        if file.is_file() and file.suffix == ".json":
            file.unlink()
            deleted += 1
    print(f" Cleaned {deleted} json files from {samples_dir.resolve()}")
