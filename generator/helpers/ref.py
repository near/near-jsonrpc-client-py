from generator.context import GeneratorContext


def resolve_ref(ref: str, ctx: GeneratorContext) -> dict:
    if not ref.startswith("#/components/schemas/"):
        raise ValueError(f"Unsupported ref: {ref}")

    name = ref.split("/")[-1]
    return ctx.schemas[name]
