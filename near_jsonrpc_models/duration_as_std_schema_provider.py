from pydantic import BaseModel
from pydantic import conint


class DurationAsStdSchemaProvider(BaseModel):
    nanos: conint(ge=-2147483648, le=2147483647)
    secs: int
