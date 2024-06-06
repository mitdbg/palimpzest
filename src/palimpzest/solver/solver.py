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
    ConvertLLMStats,
    ConvertNonLLMStats,
)


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

    def _makeFilterFn(self, td: TaskDescriptor, shouldProfile: bool = False):
        # compute record schema and type
        doc_schema = str(td.inputSchema)
        doc_type = td.inputSchema.className()

        # if filter has a function, simply return a wrapper around that function
        if td.filter.filterFn is not None:

            def nonLLMFilter(candidate: DataRecord):
                start_time = time.time()
                result = td.filter.filterFn(candidate)
                fn_call_duration_secs = time.time() - start_time

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    candidate._stats[td.op_id] = FilterNonLLMStats(
                        fn_call_duration_secs=fn_call_duration_secs,
                        filter=str(td.filter.filterFn),
                    )

                # set _passed_filter attribute and return record
                setattr(candidate, "_passed_filter", result)
                print(f"ran filter function on {candidate}")

                return candidate

            return nonLLMFilter

        # otherwise, the filter requires an LLM invocation to run
        def llmFilter(candidate: DataRecord):
            # do not filter candidate if it doesn't match inputSchema
            if not candidate.schema == td.inputSchema:
                return False

            # create generator
            generator = None
            if td.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
                generator = DSPyGenerator(
                    td.model.value,
                    td.prompt_strategy,
                    doc_schema,
                    doc_type,
                    self._verbose,
                )
            # TODO
            elif td.prompt_strategy == PromptStrategy.ZERO_SHOT:
                raise Exception("not implemented yet")
            # TODO
            elif td.prompt_strategy == PromptStrategy.FEW_SHOT:
                raise Exception("not implemented yet")
            # TODO
            elif td.prompt_strategy == PromptStrategy.CODE_GEN_BOOL:
                raise Exception("not implemented yet")

            # invoke LLM to generate filter decision (True or False)
            text_content = candidate._asJSON(include_bytes=False)
            try:
                response, _, gen_stats = generator.generate(
                    context=text_content,
                    question=td.filter.filterCondition,
                    plan_idx=td.plan_idx,
                )

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    candidate._stats[td.op_id] = FilterLLMStats(
                        gen_stats=gen_stats, filter=td.filter.filterCondition
                    )

                # set _passed_filter attribute and return record
                setattr(candidate, "_passed_filter", "true" in response.lower())
            except Exception as e:
                # If there is an exception consider the record as not passing the filter
                print(f"Error invoking LLM for filter: {e}")
                setattr(candidate, "_passed_filter", False)

            return candidate

        return llmFilter

    def synthesize(self, td: TaskDescriptor, shouldProfile: bool = False):
        """
        Return a function that implements the desired task as specified by some PhysicalOp.
        Right now, the two primary tasks that the Solver provides solutions for are:

        1. Convert operations
        2. Filter operations

        The shouldProfile parameter also determines whether or not PZ should compute
        profiling statistics for LLM invocations and attach them to each record.
        """
        # synthesize a function to induce from inputType to outputType
        if "ConvertFromCandidateOp" in td.physical_op:
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
