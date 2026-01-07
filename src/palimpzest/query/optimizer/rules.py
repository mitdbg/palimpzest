import logging
import os
from copy import deepcopy
from itertools import combinations

from palimpzest.constants import AggFunc, Model, PromptStrategy
from palimpzest.core.data.context_manager import ContextManager
from palimpzest.core.lib.schemas import (
    AUDIO_FIELD_TYPES,
    AUDIO_LIST_FIELD_TYPES,
    IMAGE_FIELD_TYPES,
    IMAGE_LIST_FIELD_TYPES,
)
from palimpzest.prompts import CONTEXT_SEARCH_PROMPT
from palimpzest.query.operators.aggregate import (
    ApplyGroupByOp,
    AverageAggregateOp,
    CountAggregateOp,
    MaxAggregateOp,
    MinAggregateOp,
    SemanticAggregate,
    SumAggregateOp,
)
from palimpzest.query.operators.compute import SmolAgentsCompute
from palimpzest.query.operators.convert import LLMConvertBonded, NonLLMConvert
from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineConvert, CritiqueAndRefineFilter
from palimpzest.query.operators.distinct import DistinctOp
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.join import EmbeddingJoin, NestedLoopsJoin, RelationalJoin
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.logical import (
    Aggregate,
    BaseScan,
    ComputeOperator,
    ContextScan,
    ConvertScan,
    Distinct,
    FilteredScan,
    GroupByAggregate,
    JoinOp,
    LimitScan,
    Project,
    SearchOperator,
    TopKScan,
)
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsConvert, MixtureOfAgentsFilter
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.project import ProjectOp
from palimpzest.query.operators.rag import RAGConvert, RAGFilter
from palimpzest.query.operators.scan import ContextScanOp, MarshalAndScanDataOp
from palimpzest.query.operators.search import (
    SmolAgentsSearch,  # SmolAgentsCustomManagedSearch,  # SmolAgentsManagedSearch
)
from palimpzest.query.operators.split import SplitConvert, SplitFilter
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer.primitives import Expression, Group, LogicalExpression, PhysicalExpression
from palimpzest.utils.model_helpers import resolve_reasoning_settings

logger = logging.getLogger(__name__)


class Rule:
    """
    The abstract base class for transformation and implementation rules.
    """

    @classmethod
    def get_rule_id(cls):
        return cls.__name__

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        raise NotImplementedError("Calling this method from an abstract base class.")

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **kwargs: dict) -> set[Expression]:
        raise NotImplementedError("Calling this method from an abstract base class.")


class TransformationRule(Rule):
    """
    Base class for transformation rules which convert a logical expression to another logical expression.
    The substitute method for a TransformationRule should return all new expressions and all new groups
    which are created during the substitution.
    """

    @classmethod
    def is_exploration_rule(cls) -> bool:
        """Returns True if this rule is an exploration rule and False otherwise. Default is False."""
        return False

    @classmethod
    def substitute(
        cls, logical_expression: LogicalExpression, groups: dict[int, Group], expressions: dict[int, Expression], **kwargs
    ) -> tuple[set[LogicalExpression], set[Group]]:
        """
        This function applies the transformation rule to the logical expression, which
        potentially creates new intermediate expressions and groups.

        The function returns a tuple containing:
        - the set of all new logical expressions created when applying the transformation
        - the set of all new groups created when applying the transformation
        - the next group id (after creating any new groups)
        """
        raise NotImplementedError("Calling this method from an abstract base class.")


class ReorderConverts(TransformationRule):
    """
    This rule is an exploration rule that returns new logical expressions by re-ordering a sequence of ConvertScans.
    """

    @classmethod
    def is_exploration_rule(cls) -> bool:
        return True

    @classmethod
    def matches_pattern(cls, logical_expression: Expression) -> bool:
        is_match = isinstance(logical_expression.operator, ConvertScan)
        logger.debug(f"ReorderConverts matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(
        cls, logical_expression: LogicalExpression, groups: dict[int, Group], expressions: dict[int, Expression], **kwargs: dict
    ) -> tuple[set[LogicalExpression], set[Group]]:
        logger.debug(f"Substituting ReorderConverts for {logical_expression}")

        # initialize the sets of new logical expressions and groups to be returned
        new_logical_expressions, new_groups = set(), set()

        # for each input group, if this convert does not depend on an operator in that group:
        # then swap the group with this convert
        convert_operator: ConvertScan = logical_expression.operator
        for input_group_id in logical_expression.input_group_ids:
            input_group = groups[input_group_id]

            # if the convert's dependencies aren't contained within the input group's fields,
            # then we can not push it down into this group
            if any([field not in input_group.fields for field in convert_operator.depends_on]):
                continue

            # iterate over logical expressions
            for expr in input_group.logical_expressions:
                # if the expression operator is not a convert, we cannot swap
                if not isinstance(expr.operator, ConvertScan):
                    continue

                # if this convert depends on a field generated by the expression we're trying to swap with, we can't swap
                if any([field in expr.generated_fields for field in convert_operator.depends_on]):
                    continue

                # create new logical expression with convert pushed down to the input group's logical expression
                new_input_group_ids = deepcopy(expr.input_group_ids)
                new_input_fields = deepcopy(expr.input_fields)
                new_depends_on_field_names = deepcopy(logical_expression.depends_on_field_names)
                new_generated_fields = deepcopy(logical_expression.generated_fields)
                new_convert_expr = LogicalExpression(
                    convert_operator,
                    input_group_ids=new_input_group_ids,
                    input_fields=new_input_fields,
                    depends_on_field_names=new_depends_on_field_names,
                    generated_fields=new_generated_fields,
                    group_id=None,
                )

                # add new_convert_expr to set of new expressions
                new_logical_expressions.add(new_convert_expr)

                # get or compute the group_id and group for this new expression
                group_id, group = None, None

                # if the expression already exists, lookup the group_id and group
                if new_convert_expr.expr_id in expressions:
                    group_id = expressions[new_convert_expr.expr_id].group_id
                    new_convert_expr.set_group_id(group_id)
                    group = groups[group_id]

                # otherwise, lookup or create expression's group and add it to the new expressions
                else:
                    # first, compute the fields for the group
                    all_fields = {**new_input_fields, **new_generated_fields}

                    # next, compute the properties; the properties will be identical to those of the input group
                    # EXCEPT for the filters which will change as a result of our swap
                    new_group_properties = deepcopy(input_group.properties)

                    # if the expression we're swapping with is a map,
                    # we need to remove its model fields from the input group properties
                    if sorted(expr.operator.input_schema.model_fields.keys()) == sorted(expr.operator.output_schema.model_fields.keys()):
                        model_fields_dict = {
                            k: {"annotation": v.annotation, "default": v.default, "description": v.description}
                            for k, v in expr.operator.output_schema.model_fields.items()
                        }
                        new_group_properties["maps"].remove(model_fields_dict)

                    # finally, if this expression is a map, add its model fields to the new group's properties
                    if sorted(convert_operator.input_schema.model_fields.keys()) == sorted(convert_operator.output_schema.model_fields.keys()):
                        model_fields_dict = {
                            k: {"annotation": v.annotation, "default": v.default, "description": v.description}
                            for k, v in convert_operator.output_schema.model_fields.items()
                        }
                        if "maps" in new_group_properties:
                            new_group_properties["maps"].add(model_fields_dict)
                        else:
                            new_group_properties["maps"] = set([model_fields_dict])

                    # create group for this new convert expression
                    group = Group(
                        logical_expressions=[new_convert_expr],
                        fields=all_fields,
                        properties=new_group_properties,
                    )
                    group_id = group.group_id
                    new_convert_expr.set_group_id(group_id)

                    # if the group already exists, add the expression to that group
                    if group_id in groups:
                        group = groups[group_id]
                        group.logical_expressions.add(new_convert_expr)

                    # otherwise, add this new group to groups and to the set of new groups
                    else:
                        groups[group_id] = group
                        new_groups.add(group)

                # create final new logical expression with expr's operator pulled up
                new_expr = LogicalExpression(
                    expr.operator.copy(),
                    input_group_ids=[group_id] + [g_id for g_id in logical_expression.input_group_ids if g_id != input_group_id],
                    input_fields=group.fields,
                    depends_on_field_names=expr.depends_on_field_names,
                    generated_fields=expr.generated_fields,
                    group_id=logical_expression.group_id,
                )

                # add newly created expression to set of returned expressions
                new_logical_expressions.add(new_expr)

        logger.debug(f"Done substituting ReorderConverts for {logical_expression}")

        return new_logical_expressions, new_groups


class PushDownFilter(TransformationRule):
    """
    If this operator is a filter, push down the filter and replace it with the
    most expensive operator in the input group.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: Expression) -> bool:
        is_match = isinstance(logical_expression.operator, FilteredScan)
        logger.debug(f"PushDownFilter matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(
        cls, logical_expression: LogicalExpression, groups: dict[int, Group], expressions: dict[int, Expression], **kwargs: dict
    ) -> tuple[set[LogicalExpression], set[Group]]:
        logger.debug(f"Substituting PushDownFilter for {logical_expression}")

        # initialize the sets of new logical expressions and groups to be returned
        new_logical_expressions, new_groups = set(), set()

        # for each input group, if this filter does not depend on an operator
        # in that group: then swap the group with this filter
        filter_operator: FilteredScan = logical_expression.operator
        for input_group_id in logical_expression.input_group_ids:
            input_group = groups[input_group_id]

            # if the filter's dependencies aren't contained within the input group's fields,
            # then we can not push it down into this group
            if any([field not in input_group.fields for field in filter_operator.depends_on]):
                continue

            # iterate over logical expressions
            # NOTE: we previously deepcopy'ed the logical expression to avoid modifying the original;
            #       I think I've fixed this internally, but I'm leaving this NOTE as a reminder in case
            #       we see a regression / bug in the future
            for expr in input_group.logical_expressions:
                # if the expression operator is not a convert or a filter, we cannot swap
                if not (isinstance(expr.operator, (ConvertScan, FilteredScan, JoinOp))):
                    continue

                # if this filter depends on a field generated by the expression we're trying to swap with, we can't swap
                if any([field in expr.generated_fields for field in filter_operator.depends_on]):
                    continue

                # create new logical expression with filter pushed down to the input group's logical expression
                new_input_group_ids = deepcopy(expr.input_group_ids)
                new_input_fields = deepcopy(expr.input_fields)
                new_depends_on_field_names = deepcopy(logical_expression.depends_on_field_names)
                new_generated_fields = deepcopy(logical_expression.generated_fields)
                new_filter_expr = LogicalExpression(
                    filter_operator,
                    input_group_ids=new_input_group_ids,
                    input_fields=new_input_fields,
                    depends_on_field_names=new_depends_on_field_names,
                    generated_fields=new_generated_fields,
                    group_id=None,
                )

                # add new_filter_expr to set of new expressions
                new_logical_expressions.add(new_filter_expr)

                # get or compute the group_id and group for this new expression
                group_id, group = None, None

                # if the expression already exists, lookup the group_id and group
                if new_filter_expr.expr_id in expressions:
                    group_id = expressions[new_filter_expr.expr_id].group_id
                    new_filter_expr.set_group_id(group_id)
                    group = groups[group_id]

                # otherwise, lookup or create expression's group and add it to the new expressions
                else:
                    # first, compute the fields for the group
                    all_fields = {**new_input_fields, **new_generated_fields}

                    # next, compute the properties; the properties will be identical to those of the input group
                    # EXCEPT for the filters which will change as a result of our swap
                    new_group_properties = deepcopy(input_group.properties)

                    # if the expression we're swapping with is a FilteredScan,
                    # we need to remove its filter from the input group properties
                    if isinstance(expr.operator, FilteredScan):
                        filter_str = expr.operator.filter.get_filter_str()
                        new_group_properties["filters"].remove(filter_str)

                    # finally, add the pushed-down filter to the new group's properties
                    filter_str = filter_operator.filter.get_filter_str()
                    if "filters" in new_group_properties:
                        new_group_properties["filters"].add(filter_str)
                    else:
                        new_group_properties["filters"] = set([filter_str])

                    # create group for this new filter expression
                    group = Group(
                        logical_expressions=[new_filter_expr],
                        fields=all_fields,
                        properties=new_group_properties,
                    )
                    group_id = group.group_id
                    new_filter_expr.set_group_id(group_id)

                    # if the group already exists, add the expression to that group
                    if group_id in groups:
                        group = groups[group_id]
                        group.logical_expressions.add(new_filter_expr)

                    # otherwise, add this new group to groups and to the set of new groups
                    else:
                        groups[group_id] = group
                        new_groups.add(group)

                # create final new logical expression with expr's operator pulled up
                new_expr = LogicalExpression(
                    expr.operator.copy(),
                    input_group_ids=[group_id] + [g_id for g_id in logical_expression.input_group_ids if g_id != input_group_id],
                    input_fields=group.fields,
                    depends_on_field_names=expr.depends_on_field_names,
                    generated_fields=expr.generated_fields,
                    group_id=logical_expression.group_id,
                )

                # add newly created expression to set of returned expressions
                new_logical_expressions.add(new_expr)

        logger.debug(f"Done substituting PushDownFilter for {logical_expression}")

        return new_logical_expressions, new_groups


class ImplementationRule(Rule):
    """
    Base class for implementation rules which convert a logical expression to a physical expression.
    """

    @classmethod
    def _get_image_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have an image (or list[image]) type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in IMAGE_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _get_list_image_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have a list[image] type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in IMAGE_LIST_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _get_audio_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have an audio (or list[audio]) type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in AUDIO_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _get_list_audio_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have a list[audio] type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in AUDIO_LIST_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_image_only_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes only image input(s) and False otherwise."""
        return all([
            field.annotation in IMAGE_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_image_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes image input(s) and False otherwise."""
        return any([
            field.annotation in IMAGE_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_audio_only_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes only audio input(s) and False otherwise."""
        return all([
            field.annotation in AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_audio_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes audio input(s) and False otherwise."""
        return any([
            field.annotation in AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_text_only_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes only text input(s) and False otherwise."""
        return all([
            field.annotation not in IMAGE_FIELD_TYPES + AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_text_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes text input(s) and False otherwise."""
        return any([
            field.annotation not in IMAGE_FIELD_TYPES + AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    # TODO: support powerset of text + image + audio (+ video) multi-modal operations
    @classmethod
    def _is_text_image_multimodal_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes text and image inputs and False otherwise."""
        return cls._is_image_operation(logical_expression) and cls._is_text_operation(logical_expression)

    @classmethod
    def _is_text_audio_multimodal_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes text and audio inputs and False otherwise."""
        return cls._is_audio_operation(logical_expression) and cls._is_text_operation(logical_expression)

    @classmethod
    def _model_matches_input(cls, model: Model, logical_expression: LogicalExpression) -> bool:
        """Returns True if the model is capable of processing the input and False otherwise."""
        # compute how many image fields are in the input, and whether any fields are list[image] fields
        num_image_fields = len(cls._get_image_fields(logical_expression))
        has_list_image_field = len(cls._get_list_image_fields(logical_expression)) > 0
        num_audio_fields = len(cls._get_audio_fields(logical_expression))
        has_list_audio_field = len(cls._get_list_audio_fields(logical_expression)) > 0

        # corner-case: for now, all operators use text or vision models for processing inputs to __call__
        if model.is_embedding_model():
            return False

        # corner-case: Llama vision models cannot handle multiple image inputs (at least using Together)
        if model.is_llama_model() and model.is_vision_model() and (num_image_fields > 1 or has_list_image_field):
            return False

        # corner-case: Gemini models cannot handle multiple audio inputs
        if model.is_vertex_model() and model.is_audio_model() and (num_audio_fields > 1 or has_list_audio_field):
            return False

        # text-only input and text supporting model
        if cls._is_text_only_operation(logical_expression) and model.is_text_model():
            return True

        # image-only input and image supporting model
        if cls._is_image_only_operation(logical_expression) and model.is_vision_model():
            return True

        # audio-only input and audio supporting model
        if cls._is_audio_only_operation(logical_expression) and model.is_audio_model():
            return True

        # multi-modal input and multi-modal supporting model
        if cls._is_text_image_multimodal_operation(logical_expression) and model.is_text_image_multimodal_model():  # noqa: SIM103
            return True

        # multi-modal input and multi-modal supporting model
        if cls._is_text_audio_multimodal_operation(logical_expression) and model.is_text_audio_multimodal_model():  # noqa: SIM103
            return True

        return False

    @classmethod
    def _get_fixed_op_kwargs(cls, logical_expression: LogicalExpression, runtime_kwargs: dict) -> dict:
        """Get the fixed set of physical op kwargs provided by the logical expression and the runtime keyword arguments."""
        # get logical operator 
        logical_op = logical_expression.operator

        # set initial set of parameters for physical op
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": runtime_kwargs["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "unique_logical_op_id": logical_op.get_unique_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
                "api_base": runtime_kwargs["api_base"],
            }
        )

        return op_kwargs

    @classmethod
    def _perform_substitution(
        cls,
        logical_expression: LogicalExpression,
        physical_op_class: type[PhysicalOperator],
        runtime_kwargs: dict,
        variable_op_kwargs: list[dict] | dict | None = None,
    ) -> set[PhysicalExpression]:
        """
        This performs basic substitution logic which proceeds in four steps:

            1. The basic kwargs for the physical operator are computed using the logical operator
               and runtime kwargs.
            2. If variable kwargs are provided, then they are merged with the basic kwargs and one
               instance of the physical operator is created for each dictionary of variable kwargs.
            3. A physical expression is created for each physical operator instance.
            4. The unique set of physical expressions is returned.

        Args:
            logical_expression (LogicalExpression): The logical expression containing a logical operator.
            physical_op_class (type[PhysicalOperator]): The class of the physical operator we wish to construct.
            runtime_kwargs (dict): Keyword arguments which are provided at runtime.
            variable_op_kwargs (list[dict] | dict | None): A (list of) variable kwargs to customize each
                physical operator instance.

        Returns:
            set[PhysicalExpression]: The unique set of physical expressions produced by initializing the
                physical_op_class with the provided keyword arguments.
        """
        # get physical operator kwargs which are fixed for each instance of the physical operator
        fixed_op_kwargs = cls._get_fixed_op_kwargs(logical_expression, runtime_kwargs)

        # make variable_op_kwargs a list of dictionaries
        if variable_op_kwargs is None:
            variable_op_kwargs = [{}]
        elif isinstance(variable_op_kwargs, dict):
            variable_op_kwargs = [variable_op_kwargs]

        # construct physical operators for each set of kwargs
        physical_expressions = []
        for var_op_kwargs in variable_op_kwargs:
            # get kwargs for this physical operator instance
            op_kwargs = {**fixed_op_kwargs, **var_op_kwargs}

            # construct the physical operator
            op = physical_op_class(**op_kwargs)

            # construct physical expression and add to list of expressions
            expression = PhysicalExpression.from_op_and_logical_expr(op, logical_expression)
            physical_expressions.append(expression)

        return set(physical_expressions)


class NonLLMConvertRule(ImplementationRule):
    """
    Substitute a logical expression for a UDF ConvertScan with a NonLLMConvert physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, ConvertScan) and logical_expression.operator.udf is not None
        logger.debug(f"NonLLMConvertRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting NonLLMConvertRule for {logical_expression}")
        return cls._perform_substitution(logical_expression, NonLLMConvert, runtime_kwargs)


class LLMConvertBondedRule(ImplementationRule):
    """
    Substitute a logical expression for a ConvertScan with a bonded convert physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, ConvertScan) and logical_expression.operator.udf is None
        logger.debug(f"LLMConvertBondedRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting LLMConvertBondedRule for {logical_expression}")

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)]
        variable_op_kwargs = []
        for model in models:
            use_reasoning_prompt, reasoning_effort = resolve_reasoning_settings(model, runtime_kwargs["reasoning_effort"])
            prompt_strategy = PromptStrategy.MAP if use_reasoning_prompt else PromptStrategy.MAP_NO_REASONING
            variable_op_kwargs.append(
                {
                    "model": model,
                    "prompt_strategy": prompt_strategy,
                    "reasoning_effort": reasoning_effort,
                }
            )

        return cls._perform_substitution(logical_expression, LLMConvertBonded, runtime_kwargs, variable_op_kwargs)


class RAGRule(ImplementationRule):
    """
    Implementation rule for the RAG operators.
    """

    num_chunks_per_fields = [1, 2, 4]
    chunk_sizes = [1000, 2000, 4000]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_map_match = isinstance(logical_op, ConvertScan) and cls._is_text_only_operation(logical_expression) and logical_op.udf is None
        is_filter_match = isinstance(logical_op, FilteredScan) and cls._is_text_only_operation(logical_expression) and logical_op.filter.filter_fn is None
        logger.debug(f"RAGRule matches_pattern: {is_map_match or is_filter_match} for {logical_expression}")
        return is_map_match or is_filter_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting RAGRule for {logical_expression}")
        # select physical operator class based on whether this is a map or filter operation
        phys_op_cls = RAGConvert if isinstance(logical_expression.operator, ConvertScan) else RAGFilter

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)]
        variable_op_kwargs = []
        for model in models:
             use_reasoning_prompt, reasoning_effort = resolve_reasoning_settings(model, runtime_kwargs["reasoning_effort"])
             prompt_strategy = (
                 PromptStrategy.MAP if use_reasoning_prompt else PromptStrategy.MAP_NO_REASONING
                 if phys_op_cls is RAGConvert
                 else PromptStrategy.FILTER if use_reasoning_prompt else PromptStrategy.FILTER_NO_REASONING
             )
             variable_op_kwargs.extend(
                 [
                    {
                        "model": model,
                        "prompt_strategy": prompt_strategy,
                        "num_chunks_per_field": num_chunks_per_field,
                        "chunk_size": chunk_size,
                        "reasoning_effort": reasoning_effort,
                    }
                    for num_chunks_per_field in cls.num_chunks_per_fields
                    for chunk_size in cls.chunk_sizes
                 ]
             )

        return cls._perform_substitution(logical_expression, phys_op_cls, runtime_kwargs, variable_op_kwargs)


class MixtureOfAgentsRule(ImplementationRule):
    """
    Implementation rule for the MixtureOfAgents operators.
    """

    num_proposer_models = [1, 2, 3]
    temperatures = [0.0, 0.4, 0.8]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_map_match = isinstance(logical_op, ConvertScan) and logical_op.udf is None
        is_filter_match = isinstance(logical_op, FilteredScan) and logical_op.filter.filter_fn is None
        logger.debug(f"MixtureOfAgentsRule matches_pattern: {is_map_match or is_filter_match} for {logical_expression}")
        return is_map_match or is_filter_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting MixtureOfAgentsRule for {logical_expression}")
        # select physical operator class based on whether this is a map or filter operation
        phys_op_cls = MixtureOfAgentsConvert if isinstance(logical_expression.operator, ConvertScan) else MixtureOfAgentsFilter

        # create variable physical operator kwargs for each model which can implement this logical_expression
        _, reasoning_effort = resolve_reasoning_settings(None, runtime_kwargs["reasoning_effort"])
        proposer_model_set = {model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)}
        aggregator_model_set = {model for model in runtime_kwargs["available_models"] if model.is_text_model()}
        variable_op_kwargs = [
            {
                "proposer_models": list(proposer_models),
                "temperatures": [temp] * len(proposer_models),
                "aggregator_model": aggregator_model,
                "reasoning_effort": reasoning_effort,
            }
            for k in cls.num_proposer_models
            for temp in cls.temperatures
            for proposer_models in combinations(proposer_model_set, k)
            for aggregator_model in aggregator_model_set
        ]

        return cls._perform_substitution(logical_expression, phys_op_cls, runtime_kwargs, variable_op_kwargs)


class CritiqueAndRefineRule(ImplementationRule):
    """
    Implementation rule for the CritiqueAndRefine operators.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_map_match = isinstance(logical_op, ConvertScan) and logical_op.udf is None
        is_filter_match = isinstance(logical_op, FilteredScan) and logical_op.filter.filter_fn is None
        logger.debug(f"CritiqueAndRefineRule matches_pattern: {is_map_match or is_filter_match} for {logical_expression}")
        return is_map_match or is_filter_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting CritiqueAndRefineRule for {logical_expression}")
        # select physical operator class based on whether this is a map or filter operation
        phys_op_cls = CritiqueAndRefineConvert if isinstance(logical_expression.operator, ConvertScan) else CritiqueAndRefineFilter

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)]
        variable_op_kwargs = []
        for model in models:
            use_reasoning_prompt, reasoning_effort = resolve_reasoning_settings(model, runtime_kwargs["reasoning_effort"])
            prompt_strategy = (
                PromptStrategy.MAP if use_reasoning_prompt else PromptStrategy.MAP_NO_REASONING
                if phys_op_cls is CritiqueAndRefineConvert
                else PromptStrategy.FILTER if use_reasoning_prompt else PromptStrategy.FILTER_NO_REASONING
            )
            variable_op_kwargs.extend(
                [
                    {
                        "model": model,
                        "critic_model": critic_model,
                        "refine_model": refine_model,
                        "prompt_strategy": prompt_strategy,
                        "reasoning_effort": reasoning_effort,
                    }
                    for critic_model in models
                    for refine_model in models
                ]
            )

        return cls._perform_substitution(logical_expression, phys_op_cls, runtime_kwargs, variable_op_kwargs)


class SplitRule(ImplementationRule):
    """
    Implementation rule for the Split operators.
    """
    num_chunks = [2, 4, 6]
    min_size_to_chunk = [1000, 4000]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_map_match = isinstance(logical_op, ConvertScan) and cls._is_text_only_operation(logical_expression) and logical_op.udf is None
        is_filter_match = isinstance(logical_op, FilteredScan) and cls._is_text_only_operation(logical_expression) and logical_op.filter.filter_fn is None
        logger.debug(f"SplitRule matches_pattern: {is_map_match or is_filter_match} for {logical_expression}")
        return is_map_match or is_filter_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting SplitRule for {logical_expression}")
        # select physical operator class based on whether this is a map or filter operation
        phys_op_cls = SplitConvert if isinstance(logical_expression.operator, ConvertScan) else SplitFilter

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)]
        _, reasoning_effort = resolve_reasoning_settings(None, runtime_kwargs["reasoning_effort"])
        variable_op_kwargs = [
            {
                "model": model,
                "min_size_to_chunk": min_size_to_chunk,
                "num_chunks": num_chunks,
                "reasoning_effort": reasoning_effort,
            }
            for model in models
            for min_size_to_chunk in cls.min_size_to_chunk
            for num_chunks in cls.num_chunks
        ]

        return cls._perform_substitution(logical_expression, phys_op_cls, runtime_kwargs, variable_op_kwargs)


class TopKRule(ImplementationRule):
    """
    Substitute a logical expression for a TopKScan with a TopK physical implementation.
    """
    k_budgets = [1, 3, 5, 10, 15, 20, 25]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, TopKScan)
        logger.debug(f"TopKRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting TopKRule for {logical_expression}")

        # create variable physical operator kwargs for each model which can implement this logical_expression
        ks = cls.k_budgets if logical_expression.operator.k == -1 else [logical_expression.operator.k]
        variable_op_kwargs = [{"k": k} for k in ks]
        return cls._perform_substitution(logical_expression, TopKOp, runtime_kwargs, variable_op_kwargs)


class NonLLMFilterRule(ImplementationRule):
    """
    Substitute a logical expression for a FilteredScan with a non-llm filter physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_match = isinstance(logical_op, FilteredScan) and logical_op.filter.filter_fn is not None
        logger.debug(f"NonLLMFilterRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting NonLLMFilterRule for {logical_expression}")
        return cls._perform_substitution(logical_expression, NonLLMFilter, runtime_kwargs)


class LLMFilterRule(ImplementationRule):
    """
    Substitute a logical expression for a FilteredScan with an llm filter physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_match = isinstance(logical_op, FilteredScan) and logical_op.filter.filter_fn is None
        logger.debug(f"LLMFilterRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting LLMFilterRule for {logical_expression}")

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)]
        variable_op_kwargs = []
        for model in models:
            use_reasoning_prompt, reasoning_effort = resolve_reasoning_settings(model, runtime_kwargs["reasoning_effort"])
            prompt_strategy = PromptStrategy.FILTER if use_reasoning_prompt else PromptStrategy.FILTER_NO_REASONING
            variable_op_kwargs.append(
                {
                    "model": model,
                    "prompt_strategy": prompt_strategy,
                    "reasoning_effort": reasoning_effort,
                }
            )

        return cls._perform_substitution(logical_expression, LLMFilter, runtime_kwargs, variable_op_kwargs)


class RelationalJoinRule(ImplementationRule):
    """
    Substitute a logical expression for a JoinOp with a RelationalJoin physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, JoinOp) and logical_expression.operator.condition == ""
        logger.debug(f"RelationalJoinRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting RelationalJoinRule for {logical_expression}")
        return cls._perform_substitution(logical_expression, RelationalJoin, runtime_kwargs)


class NestedLoopsJoinRule(ImplementationRule):
    """
    Substitute a logical expression for a JoinOp with an (LLM) NestedLoopsJoin physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, JoinOp) and logical_expression.operator.condition != ""
        logger.debug(f"NestedLoopsJoinRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting NestedLoopsJoinRule for {logical_expression}")

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)]
        variable_op_kwargs = []
        for model in models:
            use_reasoning_prompt, reasoning_effort = resolve_reasoning_settings(model, runtime_kwargs["reasoning_effort"])
            prompt_strategy = PromptStrategy.JOIN if use_reasoning_prompt else PromptStrategy.JOIN_NO_REASONING
            variable_op_kwargs.append(
                {
                    "model": model,
                    "prompt_strategy": prompt_strategy,
                    "join_parallelism": runtime_kwargs["join_parallelism"],
                    "reasoning_effort": reasoning_effort,
                    "retain_inputs": not runtime_kwargs["is_validation"],
                }
            )

        return cls._perform_substitution(logical_expression, NestedLoopsJoin, runtime_kwargs, variable_op_kwargs)


class EmbeddingJoinRule(ImplementationRule):
    """
    Substitute a logical expression for a JoinOp with an EmbeddingJoin physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, JoinOp) and logical_expression.operator.condition != "" and not cls._is_audio_operation(logical_expression)
        logger.debug(f"EmbeddingJoinRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting EmbeddingJoinRule for {logical_expression}")

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression)]
        variable_op_kwargs = []
        for model in models:
            use_reasoning_prompt, reasoning_effort = resolve_reasoning_settings(model, runtime_kwargs["reasoning_effort"])
            prompt_strategy = PromptStrategy.JOIN if use_reasoning_prompt else PromptStrategy.JOIN_NO_REASONING
            variable_op_kwargs.append(
                {
                    "model": model,
                    "prompt_strategy": prompt_strategy,
                    "join_parallelism": runtime_kwargs["join_parallelism"],
                    "reasoning_effort": reasoning_effort,
                    "retain_inputs": not runtime_kwargs["is_validation"],
                    "num_samples": 10, # TODO: iterate over different choices of num_samples
                }
            )

        return cls._perform_substitution(logical_expression, EmbeddingJoin, runtime_kwargs, variable_op_kwargs)

class SemanticAggregateRule(ImplementationRule):
    """
    Substitute a logical expression for a SemanticAggregate with an llm physical implementation.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, Aggregate) and logical_expression.operator.agg_str is not None
        logger.debug(f"SemanticAggregateRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting SemanticAggregateRule for {logical_expression}")

        # create variable physical operator kwargs for each model which can implement this logical_expression
        models = [model for model in runtime_kwargs["available_models"] if cls._model_matches_input(model, logical_expression) and not model.is_llama_model()]
        variable_op_kwargs = []
        for model in models:
            use_reasoning_prompt, reasoning_effort = resolve_reasoning_settings(model, runtime_kwargs["reasoning_effort"])
            prompt_strategy = PromptStrategy.AGG if use_reasoning_prompt else PromptStrategy.AGG_NO_REASONING
            variable_op_kwargs.append(
                {
                    "model": model,
                    "prompt_strategy": prompt_strategy,
                    "reasoning_effort": reasoning_effort,
                }
            )

        return cls._perform_substitution(logical_expression, SemanticAggregate, runtime_kwargs, variable_op_kwargs)


class AggregateRule(ImplementationRule):
    """
    Substitute the logical expression for an aggregate with its physical counterpart.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, Aggregate) and logical_expression.operator.agg_func is not None
        logger.debug(f"AggregateRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting AggregateRule for {logical_expression}")

        # get the physical op class based on the aggregation function
        physical_op_class = None
        if logical_expression.operator.agg_func == AggFunc.COUNT:
            physical_op_class = CountAggregateOp
        elif logical_expression.operator.agg_func == AggFunc.AVERAGE:
            physical_op_class = AverageAggregateOp
        elif logical_expression.operator.agg_func == AggFunc.SUM:
            physical_op_class = SumAggregateOp
        elif logical_expression.operator.agg_func == AggFunc.MIN:
            physical_op_class = MinAggregateOp
        elif logical_expression.operator.agg_func == AggFunc.MAX:
            physical_op_class = MaxAggregateOp
        else:
            raise Exception(f"Cannot support aggregate function: {logical_expression.operator.agg_func}")

        # perform the substitution
        return cls._perform_substitution(logical_expression, physical_op_class, runtime_kwargs)


class AddContextsBeforeComputeRule(ImplementationRule):
    """
    Searches the ContextManager for additional contexts which may be useful for the given computation.

    TODO: track cost of generating search query
    """
    k = 1
    SEARCH_GENERATOR_PROMPT = CONTEXT_SEARCH_PROMPT

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, ComputeOperator)
        logger.debug(f"AddContextsBeforeComputeRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting AddContextsBeforeComputeRule for {logical_expression}")

        # load an LLM to generate a short search query
        model = None
        if os.getenv("OPENAI_API_KEY"):
            model = "openai/gpt-4o-mini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = "anthropic/claude-3-5-sonnet-20241022"
        elif os.getenv("GEMINI_API_KEY"):
            model = "vertex_ai/gemini-2.0-flash"
        elif os.getenv("TOGETHER_API_KEY"):
            model = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"

        # importing litellm here because importing above causes deprecation warning
        import litellm

        # retrieve any additional context which may be useful
        cm = ContextManager()
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": cls.SEARCH_GENERATOR_PROMPT.format(instruction=logical_expression.operator.instruction)}]
        )
        query = response.choices[0].message.content
        variable_op_kwargs = {"additional_contexts": cm.search_context(query, k=cls.k, where={"materialized": True})}
        return cls._perform_substitution(logical_expression, SmolAgentsCompute, runtime_kwargs, variable_op_kwargs)


class BasicSubstitutionRule(ImplementationRule):
    """
    For logical operators with a single physical implementation, substitute the
    logical expression with its physical counterpart.
    """

    LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP = {
        BaseScan: MarshalAndScanDataOp,
        # ComputeOperator: SmolAgentsCompute,
        SearchOperator: SmolAgentsSearch, # SmolAgentsManagedSearch, # SmolAgentsCustomManagedSearch
        ContextScan: ContextScanOp,
        Distinct: DistinctOp,
        LimitScan: LimitScanOp,
        Project: ProjectOp,
        GroupByAggregate: ApplyGroupByOp,
    }

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op_class = logical_expression.operator.__class__
        is_match = logical_op_class in cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP
        logger.debug(f"BasicSubstitutionRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting BasicSubstitutionRule for {logical_expression}")
        physical_op_class = cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP[logical_expression.operator.__class__]
        return cls._perform_substitution(logical_expression, physical_op_class, runtime_kwargs)
