class GeneratorContext:
    def __init__(self, spec: dict):
        self.spec = spec
        self.schemas = spec["components"]["schemas"]
        self.paths = spec["paths"]
