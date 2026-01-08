"""Data structure for semver version and github tag or commit."""

from pydantic import BaseModel


class Version(BaseModel):
    build: str
    commit: str
    rustc_version: str = ''
    version: str
