from io import BytesIO
import numpy as np
from palimpzest.constants import PromptStrategy, QueryStrategy
from palimpzest.elements import DataRecord
from palimpzest.corelib import (
    Download,
    EquationImage,
    File,
    ImageFile,
    PDFFile,
    Schema,
    Table,
    TextFile,
    XLSFile,
)
from palimpzest.generators import DSPyGenerator
from palimpzest.profiler import (
    ApiStats,
    FilterLLMStats,
    FilterNonLLMStats,
    InduceLLMStats,
    InduceNonLLMStats,
)
from palimpzest.solver.query_strategies import (
    runBondedQuery,
    runConventionalQuery,
    runCodeGenQuery,
)
from palimpzest.solver.task_descriptors import TaskDescriptor


class Solver:
    """
    This class exposes a synthesize() method, which takes in a physical operator's
    high-level description of a task to be performed (e.g. to convert a record between
    two schemas, or to filter records adhering to a specific schema) and returns a
    function which executes that task.

    The functions returned by the Solver are responsible for marshalling input records
    and producing valid output records (where "validity" is task-specific).

    These functions are NOT responsible for managing the details of LLM output generation.
    That responsibility lies in the Generator class(es).
    """

    def __init__(self, verbose: bool = False):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._hardcodedFns = set()
        # TODO GV: As we add more hardcoded functions, we should find a more scalable way to manage them,
        # a simple idea could be to have a dictionary of hardcoded functions, where the key is a tuple of the input and output schema
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((TextFile, File))
        self._hardcodedFns.add((File, Download))
        self._hardcodedFns.add((XLSFile, File))
        self._hardcodedFns.add((Table, XLSFile))
        # self._hardcodedFns.add((ImageFile, File))
        # self._hardcodedFns.add((EquationImage, ImageFile))
        self._verbose = verbose

    def isSimpleConversion(self, td: TaskDescriptor) -> bool:
        """
        Returns true if the given task descriptor has a simple type conversion.
        """
        typeConversionDescriptor = (td.outputSchema, td.inputSchema)
        return typeConversionDescriptor in self._simpleTypeConversions

    def easyConversionAvailable(self, outputSchema: Schema, inputSchema: Schema):
        return (outputSchema, inputSchema) in self._simpleTypeConversions or (
            outputSchema,
            inputSchema,
        ) in self._hardcodedFns

    def synthesize(self, td: TaskDescriptor, shouldProfile: bool = False):
        """
        Return a function that implements the desired task as specified by some PhysicalOp.
        Right now, the two primary tasks that the Solver provides solutions for are:

        1. Induce operations
        2. Filter operations

        The shouldProfile parameter also determines whether or not PZ should compute
        profiling statistics for LLM invocations and attach them to each record.
        """
        # synthesize a function to induce from inputType to outputType
        if "InduceFromCandidateOp" in td.physical_op:
            typeConversionDescriptor = (td.outputSchema, td.inputSchema)
            if typeConversionDescriptor in self._simpleTypeConversions:
                return self._makeSimpleTypeConversionFn(td, shouldProfile)
            elif typeConversionDescriptor in self._hardcodedFns:
                return self._makeHardCodedTypeConversionFn(td, shouldProfile)
            else:
                return self._makeLLMTypeConversionFn(td, shouldProfile)

        # synthesize a function to filter records
        elif "FilterCandidateOp" in td.physical_op:
            return self._makeFilterFn(td, shouldProfile)

        else:
            raise Exception(
                "Cannot synthesize function for task descriptor: " + str(td)
            )
