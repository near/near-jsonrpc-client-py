"""Serializable version of `near-vm-runner::FunctionCallError`.

Must never reorder/remove elements, can only add new variants at the end (but do that very
carefully). It describes stable serialization format, and only used by serialization logic."""

from near_jsonrpc_models.compilation_error import CompilationError
from near_jsonrpc_models.host_error import HostError
from near_jsonrpc_models.method_resolve_error import MethodResolveError
from near_jsonrpc_models.strict_model import StrictBaseModel
from near_jsonrpc_models.wasm_trap import WasmTrap
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class FunctionCallErrorWasmUnknownError(RootModel[Literal['WasmUnknownError', '_EVMError']]):
    pass

class FunctionCallErrorCompilationError(StrictBaseModel):
    """Wasm compilation error"""
    CompilationError: CompilationError

class FunctionCallErrorLinkErrorPayload(BaseModel):
    msg: str

class FunctionCallErrorLinkError(StrictBaseModel):
    """Wasm binary env link error

Note: this is only to deserialize old data, use execution error for new data"""
    LinkError: FunctionCallErrorLinkErrorPayload

class FunctionCallErrorMethodResolveError(StrictBaseModel):
    """Import/export resolve error"""
    MethodResolveError: MethodResolveError

class FunctionCallErrorWasmTrap(StrictBaseModel):
    """A trap happened during execution of a binary

Note: this is only to deserialize old data, use execution error for new data"""
    WasmTrap: WasmTrap

class FunctionCallErrorHostError(StrictBaseModel):
    """Note: this is only to deserialize old data, use execution error for new data"""
    HostError: HostError

class FunctionCallErrorExecutionError(StrictBaseModel):
    ExecutionError: str

class FunctionCallError(RootModel[Union[FunctionCallErrorWasmUnknownError, FunctionCallErrorCompilationError, FunctionCallErrorLinkError, FunctionCallErrorMethodResolveError, FunctionCallErrorWasmTrap, FunctionCallErrorHostError, FunctionCallErrorExecutionError]]):
    pass

