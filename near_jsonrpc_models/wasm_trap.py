"""A kind of a trap happened during execution of a binaryUnreachable: An `unreachable` opcode was executed.IncorrectCallIndirectSignature: Call indirect incorrect signature trap.MemoryOutOfBounds: Memory out of bounds trap.CallIndirectOOB: Call indirect out of bounds trap.IllegalArithmetic: An arithmetic exception, e.g. divided by zero.MisalignedAtomicAccess: Misaligned atomic access trap.IndirectCallToNull: Indirect call to null.StackOverflow: Stack overflow.GenericTrap: Generic trap."""

from pydantic import RootModel
from typing import Literal


class WasmTrap(RootModel[Literal['Unreachable', 'IncorrectCallIndirectSignature', 'MemoryOutOfBounds', 'CallIndirectOOB', 'IllegalArithmetic', 'MisalignedAtomicAccess', 'IndirectCallToNull', 'StackOverflow', 'GenericTrap']]):
    pass

