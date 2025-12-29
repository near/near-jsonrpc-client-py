import re


def snake_case(schema_name: str) -> str:
    if "EXPERIMENTAL" in schema_name:
        schema_name = schema_name.replace("EXPERIMENTAL", "experimental")

    schema_name = re.sub(r'([a-z])([A-Z])', r'\1_\2', schema_name).lower()

    return schema_name


def _make_safe_class_name(root: str, suffix: str) -> str:
    if re.fullmatch(r'[A-Z0-9_]+', suffix):
        parts = suffix.lower().split('_')
        suffix_safe = ''.join(p.capitalize() for p in parts)
    else:
        suffix_safe = pascal_case(suffix)

    return f"{pascal_case(root)}{suffix_safe}"


def to_constant_name(value: str) -> str:
    # PascalCase or CamelCase
    if (
            re.match(r'^[A-Z][a-zA-Z0-9]*$', value)
            or re.match(r'^[a-z][a-zA-Z0-9]*$', value)
    ):
        return re.sub(
            r'([a-z0-9])([A-Z])',
            r'\1_\2',
            value
        ).upper()

    else:
        s = re.sub(r'[^A-Za-z0-9]+', '_', value)
        s = re.sub(r'_+', '_', s)
        s = s.strip('_')

        if not s:
            return "UNKNOWN"

        s = s.upper()

        # If starts with digit, prefix underscore
        if s[0].isdigit():
            s = f"_{s}"

        return s


def pascal_case(schema_name: str) -> str:
    if "EXPERIMENTAL" in schema_name:
        schema_name = schema_name.replace("EXPERIMENTAL", "experimental")

    if re.fullmatch(r'[A-Z0-9_]+', schema_name):
        schema_name = schema_name.lower()

    if re.match(r'^[A-Z][a-zA-Z0-9]*$', schema_name):
        return schema_name

    parts = schema_name.split('_')
    camel_case_name = ''.join(word[0].upper() + word[1:] if word else '' for word in parts)

    return camel_case_name
