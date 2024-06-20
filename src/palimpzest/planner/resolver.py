from palimpzest.constants import Model, PromptStrategy, QueryStrategy
from palimpzest.operators import PhysicalOperator

import palimpzest as pz
import palimpzest.operators as pz_ops
import palimpzest.corelib.schemas as schemas

from typing import Optional


import multiprocessing



def resolveLogicalConvertOp(
    logical_convert_op: pz_ops.ConvertScan,
    model: Optional[Model] = None,
    prompt_strategy: Optional[PromptStrategy] = None,
    query_strategy: Optional[QueryStrategy] = None,
    token_budget: Optional[float] = 1.0,
    shouldProfile: bool = False,
    sentinel: bool = False,
) -> PhysicalOperator:
    """
    Given the logical operator for a convert, determine which (set of) physical operation(s)
    the PhysicalPlanner can use to implement that logical operation.
    """
    if sentinel:
        shouldProfile = True

    # get input and output schemas for convert
    inputSchema = logical_convert_op.inputSchema
    outputSchema = logical_convert_op.outputSchema

    # TODO: test schema equality
    # use simple convert if the input and output schemas are the same
    if inputSchema == outputSchema:
        op = pz_ops.SimpleTypeConvert(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            targetCacheId=logical_convert_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

    # TODO: replace all these elif's with iteration over self.physical_ops
    #       - Q: what happens if we ever have two hard-coded conversions w/same input and output schema?
    #            - e.g. imagine if we had ConvertDownloadToFileTypeA and ConvertDownloadToFileTypeB
    # if input and output schema are covered by a hard-coded convert; use that
    elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.TextFile):
        op = pz_ops.ConvertFileToText(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            targetCacheId=logical_convert_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

    elif isinstance(inputSchema, schemas.ImageFile) and isinstance(outputSchema, schemas.EquationImage):
        op = pz_ops.ConvertImageToEquation(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            targetCacheId=logical_convert_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

    elif isinstance(inputSchema, schemas.Download) and isinstance(outputSchema, schemas.File):
        op = pz_ops.ConvertDownloadToFile(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            targetCacheId=logical_convert_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

    elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.XLSFile):
        op = pz_ops.ConvertFileToXLS(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            targetCacheId=logical_convert_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

    elif isinstance(inputSchema, schemas.XLSFile) and isinstance(outputSchema, schemas.Table):
        op = pz_ops.ConvertXLSToTable(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            cardinality=logical_convert_op.cardinality,
            targetCacheId=logical_convert_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

    elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.PDFFile):
        op = pz_ops.ConvertFileToPDF(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            pdfprocessor=pz.DataDirectory().current_config.get("pdfprocessing"),
            targetCacheId=logical_convert_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

    # otherwise, create convert op for the given set of hyper-parameters
    else:
        assert prompt_strategy is not None, "Prompt strategy must be specified for LLMConvert"
        op = pz_ops.LLMConvert(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            model=model,
            prompt_strategy=prompt_strategy,
            query_strategy=query_strategy,
            token_budget=token_budget,
            cardinality=logical_convert_op.cardinality,
            image_conversion=logical_convert_op.image_conversion,
            desc=logical_convert_op.desc,
            targetCacheId=logical_convert_op.targetCacheId,
        )

    return op

def resolveLogicalFilterOp(
    logical_filter_op: pz_ops.FilteredScan,
    useParallelOps: bool = False,
    model: Optional[Model] = None,
    prompt_strategy: Optional[PromptStrategy] = None,
    shouldProfile: bool = False,
    sentinel: bool = False,
) -> PhysicalOperator:
    """
    Given the logical operator for a filter, determine which (set of) physical operation(s)
    the PhysicalPlanner can use to implement that logical operation.
    """
    if sentinel:
        shouldProfile = True

    use_llm_filter = logical_filter_op.filter.filterFn is None
    op = (
        pz_ops.LLMFilter(
            inputSchema=logical_filter_op.inputSchema,
            outputSchema=logical_filter_op.outputSchema,
            filter=logical_filter_op.filter,
            model=model,
            prompt_strategy=prompt_strategy,
            targetCacheId=logical_filter_op.targetCacheId,
            shouldProfile=shouldProfile,
            max_workers=multiprocessing.cpu_count() if useParallelOps else 1,
        )
        if use_llm_filter
        else pz_ops.NonLLMFilter(
            inputSchema=logical_filter_op.inputSchema,
            outputSchema=logical_filter_op.outputSchema,
            filter=logical_filter_op.filter,
            targetCacheId=logical_filter_op.targetCacheId,
            shouldProfile=shouldProfile,
            max_workers=multiprocessing.cpu_count() if useParallelOps else 1,
        )
    )

    return op

def resolveLogicalApplyAggFuncOp(
    logical_apply_agg_fn_op: pz_ops.ApplyAggregateFunction,
    shouldProfile: bool = False,
) -> PhysicalOperator:
    """
    Given the logical operator for a group by, determine which (set of) physical operation(s)
    the PhysicalPlanner can use to implement that logical operation.
    """

    # TODO: use an Enum to list possible funcDesc(s)
    physicalOp = None
    if logical_apply_agg_fn_op.aggregationFunction.funcDesc == "COUNT":
        physicalOp = pz_ops.ApplyCountAggregateOp
    elif logical_apply_agg_fn_op.aggregationFunction.funcDesc == "AVERAGE":
        physicalOp = pz_ops.ApplyAverageAggregateOp

    op = physicalOp(
        inputSchema=logical_apply_agg_fn_op.inputSchema,
        aggFunction=logical_apply_agg_fn_op.aggregationFunction,
        targetCacheId=logical_apply_agg_fn_op.targetCacheId,
        shouldProfile=shouldProfile,
    )

    return op