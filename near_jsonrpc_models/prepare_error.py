"""Error that can occur while preparing or executing Wasm smart-contract.Serialization: Error happened while serializing the module.Deserialization: Error happened while deserializing the module.InternalMemoryDeclared: Internal memory declaration has been found in the module.GasInstrumentation: Gas instrumentation failed.

This most likely indicates the module isn't valid.StackHeightInstrumentation: Stack instrumentation failed.

This  most likely indicates the module isn't valid.Instantiate: Error happened during instantiation.

This might indicate that `start` function trapped, or module isn't
instantiable and/or un-linkable.Memory: Error creating memory.TooManyFunctions: Contract contains too many functions.TooManyLocals: Contract contains too many locals.TooManyTables: Contract contains too many tables.TooManyTableElements: Contract contains too many table elements.FunctionBodyTooLarge: A function body in the contract exceeds the size limit.InstrumentedCodeTooLarge: The instrumented code exceeds the size limit.TooManyBlocksPerFunction: A function contains too many basic blocks.TooManyBlocksPerContract: A contract contains too many basic blocks."""

from pydantic import RootModel
from typing import Literal


class PrepareError(RootModel[Literal['Serialization', 'Deserialization', 'InternalMemoryDeclared', 'GasInstrumentation', 'StackHeightInstrumentation', 'Instantiate', 'Memory', 'TooManyFunctions', 'TooManyLocals', 'TooManyTables', 'TooManyTableElements', 'FunctionBodyTooLarge', 'InstrumentedCodeTooLarge', 'TooManyBlocksPerFunction', 'TooManyBlocksPerContract']]):
    pass

