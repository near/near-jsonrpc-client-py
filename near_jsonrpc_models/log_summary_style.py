from pydantic import RootModel
from typing import Literal


class LogSummaryStyle(RootModel[Literal['plain', 'colored']]):
    pass

