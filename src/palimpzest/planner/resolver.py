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
) -> PhysicalOperator:
    """
    Given the logical operator for a convert, determine which (set of) physical operation(s)
    the PhysicalPlanner can use to implement that logical operation.
    """
    # get input and output schemas for convert
    inputSchema = logical_convert_op.inputSchema
    outputSchema = logical_convert_op.outputSchema

    parameters = {
        'inputSchema': inputSchema,
        'outputSchema': outputSchema,
        'targetCacheId': logical_convert_op.targetCacheId,
        'shouldProfile': shouldProfile,
    }

    # TODO: test schema equality
    # use simple convert if the input and output schemas are the same
    if inputSchema == outputSchema:
        op_class = pz_ops.SimpleTypeConvert

    # TODO: replace all these elif's with iteration over self.physical_ops
    #       - Q: what happens if we ever have two hard-coded conversions w/same input and output schema?
    #            - e.g. imagine if we had ConvertDownloadToFileTypeA and ConvertDownloadToFileTypeB
    # if input and output schema are covered by a hard-coded convert; use that
    elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.TextFile):
        op_class = pz_ops.ConvertFileToText

    elif isinstance(inputSchema, schemas.ImageFile) and isinstance(outputSchema, schemas.EquationImage):
        op_class = pz_ops.ConvertImageToEquation

    elif isinstance(inputSchema, schemas.Download) and isinstance(outputSchema, schemas.File):
        op_class = pz_ops.ConvertDownloadToFile

    elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.XLSFile):
        op_class = pz_ops.ConvertFileToXLS

    elif isinstance(inputSchema, schemas.XLSFile) and isinstance(outputSchema, schemas.Table):
        op = pz_ops.ConvertXLSToTable
        parameters["cardinality"] = logical_convert_op.cardinality

    elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.PDFFile):
        op_class = pz_ops.ConvertFileToPDF

    # otherwise, create convert op for the given set of hyper-parameters
    else:
        assert prompt_strategy is not None, "Prompt strategy must be specified for LLMConvert"
        op_class = pz_ops.LLMConvert
        parameters["model"] = model
        parameters["prompt_strategy"] = prompt_strategy
        parameters["query_strategy"] = query_strategy
        parameters["token_budget"] = token_budget
        parameters["cardinality"] = logical_convert_op.cardinality
        parameters["image_conversion"] = logical_convert_op.image_conversion
        parameters["desc"] = logical_convert_op.desc

    return op_class(**parameters)

def resolveLogicalFilterOp(
    logical_filter_op: pz_ops.FilteredScan,
    useParallelOps: bool = False,
    model: Optional[Model] = None,
    prompt_strategy: Optional[PromptStrategy] = None,
    shouldProfile: bool = False,
) -> PhysicalOperator:
    """
    Given the logical operator for a filter, determine which (set of) physical operation(s)
    the PhysicalPlanner can use to implement that logical operation.
    """

    parameters = {
        'inputSchema': logical_filter_op.inputSchema,
        'outputSchema': logical_filter_op.outputSchema,
        'filter': logical_filter_op.filter,
        'targetCacheId': logical_filter_op.targetCacheId,
        'shouldProfile': shouldProfile,
        'max_workers': multiprocessing.cpu_count() if useParallelOps else 1,
    }
    if logical_filter_op.filter.filterFn is None:
        op_class = pz_ops.LLMFilter
        parameters["model"] = model
        parameters["prompt_strategy"] = prompt_strategy
    else:
        op_class = pz_ops.NonLLMFilter

    return op_class(**parameters)

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