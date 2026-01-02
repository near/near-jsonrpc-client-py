"""Wasmer0: Wasmer 0.17.x VM. Gone now.Wasmtime: Wasmtime VM.Wasmer2: Wasmer 2.x VM.NearVm: NearVM."""

from pydantic import RootModel
from typing import Literal


class VMKind(RootModel[Literal['Wasmer0', 'Wasmtime', 'Wasmer2', 'NearVm']]):
    pass

