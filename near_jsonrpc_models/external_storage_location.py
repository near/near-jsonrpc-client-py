"""Supported external storage backends and their minimal config."""

from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class ExternalStorageLocationS3Payload(BaseModel):
    # Location on S3.
    bucket: str
    # Data may only be available in certain locations.
    region: str

class ExternalStorageLocationS3(StrictBaseModel):
    S3: ExternalStorageLocationS3Payload

class ExternalStorageLocationFilesystemPayload(BaseModel):
    root_dir: str

class ExternalStorageLocationFilesystem(StrictBaseModel):
    """Local filesystem root for storing data."""
    Filesystem: ExternalStorageLocationFilesystemPayload

class ExternalStorageLocationGcsPayload(BaseModel):
    bucket: str

class ExternalStorageLocationGcs(StrictBaseModel):
    """Google Cloud Storage bucket name."""
    GCS: ExternalStorageLocationGcsPayload

class ExternalStorageLocation(RootModel[Union[ExternalStorageLocationS3, ExternalStorageLocationFilesystem, ExternalStorageLocationGcs]]):
    pass

