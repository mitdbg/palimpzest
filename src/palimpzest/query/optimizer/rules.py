import logging
from copy import deepcopy
from itertools import combinations

from palimpzest.constants import AggFunc, Cardinality, Model, PromptStrategy
from palimpzest.query.operators.aggregate import ApplyGroupByOp, AverageAggregateOp, CountAggregateOp
from palimpzest.query.operators.code_synthesis_convert import CodeSynthesisConvertSingle
from palimpzest.query.operators.convert import LLMConvertBonded, NonLLMConvert
from palimpzest.query.operators.critique_and_refine_convert import CriticAndRefineConvert
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.logical import (
    Aggregate,
    BaseScan,
    CacheScan,
    ConvertScan,
    FilteredScan,
    GroupByAggregate,
    LimitScan,
    MapScan,
    Project,
    RetrieveScan,
)
from palimpzest.query.operators.map import MapOp
from palimpzest.query.operators.mixture_of_agents_convert import MixtureOfAgentsConvert
from palimpzest.query.operators.project import ProjectOp
from palimpzest.query.operators.rag_convert import RAGConvert
from palimpzest.query.operators.retrieve import RetrieveOp
from palimpzest.query.operators.scan import CacheScanDataOp, MarshalAndScanDataOp
from palimpzest.query.operators.token_reduction_convert import TokenReducedConvertBonded
from palimpzest.query.optimizer.primitives import Expression, Group, LogicalExpression, PhysicalExpression
from palimpzest.utils.model_helpers import get_models, get_vision_models

logger = logging.getLogger(__name__)


class Rule:
    """
    The abstract base class for transformation and implementation rules.
    """

    @classmethod
    def get_rule_id(cls):
        return cls.__name__

    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        raise NotImplementedError("Calling this method from an abstract base class.")

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **kwargs) -> set[Expression]:
        raise NotImplementedError("Calling this method from an abstract base class.")


class TransformationRule(Rule):
    """
    Base class for transformation rules which convert a logical expression to another logical expression.
    The substitute method for a TransformationRule should return all new expressions and all new groups
    which are created during the substitution.
    """

    @staticmethod
    def substitute(
        logical_expression: LogicalExpression, groups: dict[int, Group], expressions: dict[int, Expression], **kwargs
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


class PushDownFilter(TransformationRule):
    """
    If this operator is a filter, push down the filter and replace it with the
    most expensive operator in the input group.
    """

    @staticmethod
    def matches_pattern(logical_expression: Expression) -> bool:
        is_match = isinstance(logical_expression.operator, FilteredScan)
        logger.debug(f"PushDownFilter matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @staticmethod
    def substitute(
        logical_expression: LogicalExpression, groups: dict[int, Group], expressions: dict[int, Expression], **kwargs
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
                if not (isinstance(expr.operator, (ConvertScan, FilteredScan))):
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
                if new_filter_expr.get_expr_id() in expressions:
                    group_id = expressions[new_filter_expr.get_expr_id()].group_id
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
                    input_group_ids=[group_id]
                    + [g_id for g_id in logical_expression.input_group_ids if g_id != input_group_id],
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

    pass


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
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting NonLLMConvertRule for {logical_expression}")

        logical_op = logical_expression.operator

        # get initial set of parameters for physical op
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # construct multi-expression
        op = NonLLMConvert(**op_kwargs)
        expression = PhysicalExpression(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            depends_on_field_names=logical_expression.depends_on_field_names,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )

        deduped_physical_expressions = set([expression])
        logger.debug(f"Done substituting NonLLMConvertRule for {logical_expression}")

        return deduped_physical_expressions


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
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting LLMConvertBondedRule for {logical_expression}")

        logical_op = logical_expression.operator

        # get initial set of parameters for physical op
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())
        pure_text_models = {model for model in text_models if model not in vision_models}
        pure_vision_models = {model for model in vision_models if model not in text_models}

        # compute attributes about this convert operation
        is_image_conversion = any(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        num_image_fields = sum(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        list_image_field = any(
            [
                field.is_image_field and hasattr(field, "element_type")
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )

        physical_expressions = []
        for model in physical_op_params["available_models"]:
            # skip this model if:
            # 1. this is a pure vision model and we're not doing an image conversion, or
            # 2. this is a pure text model and we're doing an image conversion, or
            # 3. this is a vision model hosted by Together (i.e. LLAMA3_V) and there is more than one image field
            first_criteria = model in pure_vision_models and not is_image_conversion
            second_criteria = model in pure_text_models and is_image_conversion
            third_criteria = model == Model.LLAMA3_V and (num_image_fields > 1 or list_image_field)
            if first_criteria or second_criteria or third_criteria:
                continue

            # construct multi-expression
            op = LLMConvertBonded(
                model=model,
                prompt_strategy=PromptStrategy.COT_QA_IMAGE if is_image_conversion else PromptStrategy.COT_QA,
                **op_kwargs,
            )
            expression = PhysicalExpression(
                operator=op,
                input_group_ids=logical_expression.input_group_ids,
                input_fields=logical_expression.input_fields,
                depends_on_field_names=logical_expression.depends_on_field_names,
                generated_fields=logical_expression.generated_fields,
                group_id=logical_expression.group_id,
            )
            physical_expressions.append(expression)

        deduped_physical_expressions = set(physical_expressions)
        logger.debug(f"Done substituting LLMConvertBondedRule for {logical_expression}")

        return deduped_physical_expressions


class TokenReducedConvertBondedRule(ImplementationRule):
    """
    Substitute a logical expression for a ConvertScan with a bonded token reduced physical implementation.
    """

    token_budgets = [0.1, 0.5, 0.9]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_image_conversion = any(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        is_match = isinstance(logical_op, ConvertScan) and not is_image_conversion and logical_op.udf is None
        logger.debug(f"TokenReducedConvertBondedRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting TokenReducedConvertBondedRule for {logical_expression}")

        logical_op = logical_expression.operator

        # get initial set of parameters for physical op
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())
        pure_vision_models = {model for model in vision_models if model not in text_models}

        physical_expressions = []
        for model in physical_op_params["available_models"]:
            for token_budget in cls.token_budgets:
                # skip this model if this is a pure image model
                if model in pure_vision_models:
                    continue

                # construct multi-expression
                op = TokenReducedConvertBonded(
                    model=model,
                    prompt_strategy=PromptStrategy.COT_QA,
                    token_budget=token_budget,
                    **op_kwargs,
                )
                expression = PhysicalExpression(
                    operator=op,
                    input_group_ids=logical_expression.input_group_ids,
                    input_fields=logical_expression.input_fields,
                    depends_on_field_names=logical_expression.depends_on_field_names,
                    generated_fields=logical_expression.generated_fields,
                    group_id=logical_expression.group_id,
                )
                physical_expressions.append(expression)

        logger.debug(f"Done substituting TokenReducedConvertBondedRule for {logical_expression}")
        deduped_physical_expressions = set(physical_expressions)

        return deduped_physical_expressions


class CodeSynthesisConvertRule(ImplementationRule):
    """
    Base rule for code synthesis convert operators; the physical convert class
    (CodeSynthesisConvertSingle) is provided by sub-class rules.

    NOTE: we provide the physical convert class(es) in their own sub-classed rules to make
    it easier to allow/disallow groups of rules at the Optimizer level.
    """

    physical_convert_class = None  # overriden by sub-classes

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_image_conversion = any(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        is_match = (
            isinstance(logical_op, ConvertScan)
            and not is_image_conversion
            and logical_op.cardinality != Cardinality.ONE_TO_MANY
            and logical_op.udf is None
        )
        logger.debug(f"CodeSynthesisConvertRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting CodeSynthesisConvertRule for {logical_expression}")

        logical_op = logical_expression.operator

        # get initial set of parameters for physical op
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # construct multi-expression
        op = cls.physical_convert_class(
            exemplar_generation_model=physical_op_params["champion_model"],
            code_synth_model=physical_op_params["code_champion_model"],
            fallback_model=physical_op_params["fallback_model"],
            prompt_strategy=PromptStrategy.COT_QA,
            **op_kwargs,
        )
        expression = PhysicalExpression(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            depends_on_field_names=logical_expression.depends_on_field_names,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )
        deduped_physical_expressions = set([expression])
        logger.debug(f"Done substituting CodeSynthesisConvertRule for {logical_expression}")

        return deduped_physical_expressions


class CodeSynthesisConvertSingleRule(CodeSynthesisConvertRule):
    """
    Substitute a logical expression for a ConvertScan with a (single) code synthesis physical implementation.
    """

    physical_convert_class = CodeSynthesisConvertSingle


class RAGConvertRule(ImplementationRule):
    """
    Substitute a logical expression for a ConvertScan with a RAGConvert physical implementation.
    """

    num_chunks_per_fields = [1, 2, 4]
    chunk_sizes = [1000, 2000, 4000]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_image_conversion = any(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        is_match = isinstance(logical_op, ConvertScan) and not is_image_conversion and logical_op.udf is None
        logger.debug(f"RAGConvertRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting RAGConvertRule for {logical_expression}")

        logical_op = logical_expression.operator

        # get initial set of parameters for physical op
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())
        pure_vision_models = {model for model in vision_models if model not in text_models}

        physical_expressions = []
        for model in physical_op_params["available_models"]:
            # skip this model if this is a pure image model
            if model in pure_vision_models:
                continue

            for num_chunks_per_field in cls.num_chunks_per_fields:
                for chunk_size in cls.chunk_sizes:
                    # construct multi-expression
                    op = RAGConvert(
                        model=model,
                        prompt_strategy=PromptStrategy.COT_QA,
                        num_chunks_per_field=num_chunks_per_field,
                        chunk_size=chunk_size,
                        **op_kwargs,
                    )
                    expression = PhysicalExpression(
                        operator=op,
                        input_group_ids=logical_expression.input_group_ids,
                        input_fields=logical_expression.input_fields,
                        depends_on_field_names=logical_expression.depends_on_field_names,
                        generated_fields=logical_expression.generated_fields,
                        group_id=logical_expression.group_id,
                    )
                    physical_expressions.append(expression)

        logger.debug(f"Done substituting RAGConvertRule for {logical_expression}")
        deduped_physical_expressions = set(physical_expressions)

        return deduped_physical_expressions


class MixtureOfAgentsConvertRule(ImplementationRule):
    """
    Implementation rule for the MixtureOfAgentsConvert operator.
    """

    num_proposer_models = [1, 2, 3]
    temperatures = [0.0, 0.4, 0.8]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_match = isinstance(logical_op, ConvertScan) and logical_op.udf is None
        logger.debug(f"MixtureOfAgentsConvertRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting MixtureOfAgentsConvertRule for {logical_expression}")

        logical_op = logical_expression.operator

        # get initial set of parameters for physical op
        op_kwargs: dict = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())

        # construct set of proposer models and set of aggregator models
        num_image_fields = sum(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        list_image_field = any(
            [
                field.is_image_field and hasattr(field, "element_type")
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        proposer_model_set, is_image_conversion = text_models, False
        if num_image_fields > 1 or list_image_field:
            proposer_model_set = [model for model in vision_models if model != Model.LLAMA3_V]
            is_image_conversion = True
        elif num_image_fields == 1:
            proposer_model_set = vision_models
            is_image_conversion = True
        aggregator_model_set = text_models

        # filter un-available models out of sets
        proposer_model_set = {model for model in proposer_model_set if model in physical_op_params["available_models"]}
        aggregator_model_set = {
            model for model in aggregator_model_set if model in physical_op_params["available_models"]
        }

        # construct MixtureOfAgentsConvert operations for various numbers of proposer models
        # and for every combination of proposer models and aggregator model
        physical_expressions = []
        for k in cls.num_proposer_models:
            for temp in cls.temperatures:
                for proposer_models in combinations(proposer_model_set, k):
                    for aggregator_model in aggregator_model_set:
                        # construct multi-expression
                        op = MixtureOfAgentsConvert(
                            proposer_models=list(proposer_models),
                            temperatures=[temp] * len(proposer_models),
                            aggregator_model=aggregator_model,
                            proposer_prompt=op_kwargs.get("prompt"),
                            proposer_prompt_strategy=PromptStrategy.COT_MOA_PROPOSER_IMAGE
                            if is_image_conversion
                            else PromptStrategy.COT_MOA_PROPOSER,
                            aggregator_prompt_strategy=PromptStrategy.COT_MOA_AGG,
                            **op_kwargs,
                        )
                        expression = PhysicalExpression(
                            operator=op,
                            input_group_ids=logical_expression.input_group_ids,
                            input_fields=logical_expression.input_fields,
                            depends_on_field_names=logical_expression.depends_on_field_names,
                            generated_fields=logical_expression.generated_fields,
                            group_id=logical_expression.group_id,
                        )
                        physical_expressions.append(expression)

        logger.debug(f"Done substituting MixtureOfAgentsConvertRule for {logical_expression}")
        deduped_physical_expressions = set(physical_expressions)

        return deduped_physical_expressions


class CriticAndRefineConvertRule(ImplementationRule):
    """
    Implementation rule for the CriticAndRefineConvert operator.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_match = isinstance(logical_op, ConvertScan) and logical_op.udf is None
        logger.debug(f"CriticAndRefineConvertRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting CriticAndRefineConvertRule for {logical_expression}")

        logical_op = logical_expression.operator

        # Get initial parameters for physical operator
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())
        pure_text_models = {model for model in text_models if model not in vision_models}
        pure_vision_models = {model for model in vision_models if model not in text_models}

        # compute attributes about this convert operation
        is_image_conversion = any(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        num_image_fields = sum(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        list_image_field = any(
            [
                field.is_image_field and hasattr(field, "element_type")
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )

        # identify models which can be used for this convert operation
        models = []
        for model in physical_op_params["available_models"]:
            # skip this model if:
            # 1. this is a pure vision model and we're not doing an image conversion, or
            # 2. this is a pure text model and we're doing an image conversion, or
            # 3. this is a vision model hosted by Together (i.e. LLAMA3_V) and there is more than one image field
            first_criteria = model in pure_vision_models and not is_image_conversion
            second_criteria = model in pure_text_models and is_image_conversion
            third_criteria = model == Model.LLAMA3_V and (num_image_fields > 1 or list_image_field)
            if first_criteria or second_criteria or third_criteria:
                continue

            models.append(model)

        # TODO: heuristic(s) to narrow the space of critic and refine models we consider using class attributes
        # construct CriticAndRefineConvert operations for every combination of model, critic model, and refinement model
        physical_expressions = []
        for model in models:
            for critic_model in models:
                for refine_model in models:
                    # construct multi-expression
                    op = CriticAndRefineConvert(
                        model=model,
                        prompt_strategy=PromptStrategy.COT_QA_IMAGE if is_image_conversion else PromptStrategy.COT_QA,
                        critic_model=critic_model,
                        refine_model=refine_model,
                        **op_kwargs,
                    )
                    expression = PhysicalExpression(
                        operator=op,
                        input_group_ids=logical_expression.input_group_ids,
                        input_fields=logical_expression.input_fields,
                        depends_on_field_names=logical_expression.depends_on_field_names,
                        generated_fields=logical_expression.generated_fields,
                        group_id=logical_expression.group_id,
                    )
                    physical_expressions.append(expression)

        logger.debug(f"Done substituting CriticAndRefineConvertRule for {logical_expression}")
        deduped_physical_expressions = set(physical_expressions)

        return deduped_physical_expressions


class RetrieveRule(ImplementationRule):
    """
    Substitute a logical expression for a RetrieveScan with a Retrieve physical implementation.
    """
    k_budgets = [1, 3, 5, 10, 15, 20, 25]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, RetrieveScan)
        logger.debug(f"RetrieveRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting RetrieveRule for {logical_expression}")

        logical_op = logical_expression.operator

        physical_expressions = []
        ks = cls.k_budgets if logical_op.k == -1 else [logical_op.k]
        for k in ks:
            # get initial set of parameters for physical op
            op_kwargs = logical_op.get_logical_op_params()
            op_kwargs.update(
                {
                    "verbose": physical_op_params["verbose"],
                    "logical_op_id": logical_op.get_logical_op_id(),
                    "logical_op_name": logical_op.logical_op_name(),
                    "k": k,
                }
            )

            # construct multi-expression
            op = RetrieveOp(**op_kwargs)
            expression = PhysicalExpression(
                operator=op,
                input_group_ids=logical_expression.input_group_ids,
                input_fields=logical_expression.input_fields,
                depends_on_field_names=logical_expression.depends_on_field_names,
                generated_fields=logical_expression.generated_fields,
                group_id=logical_expression.group_id,
            )

            physical_expressions.append(expression)

        logger.debug(f"Done substituting RetrieveRule for {logical_expression}")
        deduped_physical_expressions = set(physical_expressions)

        return deduped_physical_expressions


class NonLLMFilterRule(ImplementationRule):
    """
    Substitute a logical expression for a FilteredScan with a non-llm filter physical implementation.
    """

    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        is_match = (
            isinstance(logical_expression.operator, FilteredScan)
            and logical_expression.operator.filter.filter_fn is not None
        )
        logger.debug(f"NonLLMFilterRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting NonLLMFilterRule for {logical_expression}")

        logical_op = logical_expression.operator
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )
        op = NonLLMFilter(**op_kwargs)

        expression = PhysicalExpression(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            depends_on_field_names=logical_expression.depends_on_field_names,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )
        logger.debug(f"Done substituting NonLLMFilterRule for {logical_expression}")
        deduped_physical_expressions = set([expression])

        return deduped_physical_expressions


class LLMFilterRule(ImplementationRule):
    """
    Substitute a logical expression for a FilteredScan with an llm filter physical implementation.
    """

    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        is_match = (
            isinstance(logical_expression.operator, FilteredScan)
            and logical_expression.operator.filter.filter_condition is not None
        )
        logger.debug(f"LLMFilterRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting LLMFilterRule for {logical_expression}")

        logical_op = logical_expression.operator
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())
        pure_text_models = {model for model in text_models if model not in vision_models}
        pure_vision_models = {model for model in vision_models if model not in text_models}

        # compute attributes about this filter operation
        is_image_filter = any(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        num_image_fields = sum(
            [
                field.is_image_field
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )
        list_image_field = any(
            [
                field.is_image_field and hasattr(field, "element_type")
                for field_name, field in logical_expression.input_fields.items()
                if field_name.split(".")[-1] in logical_expression.depends_on_field_names
            ]
        )

        physical_expressions = []
        for model in physical_op_params["available_models"]:
            # skip this model if:
            # 1. this is a pure vision model and we're not doing an image filter, or
            # 2. this is a pure text model and we're doing an image filter, or
            # 3. this is a vision model hosted by Together (i.e. LLAMA3_V) and there is more than one image field
            first_criteria = model in pure_vision_models and not is_image_filter
            second_criteria = model in pure_text_models and is_image_filter
            third_criteria = model == Model.LLAMA3_V and (num_image_fields > 1 or list_image_field)
            if first_criteria or second_criteria or third_criteria:
                continue

            # construct multi-expression
            op = LLMFilter(
                model=model,
                prompt_strategy=PromptStrategy.COT_BOOL_IMAGE if is_image_filter else PromptStrategy.COT_BOOL,
                **op_kwargs,
            )
            expression = PhysicalExpression(
                operator=op,
                input_group_ids=logical_expression.input_group_ids,
                input_fields=logical_expression.input_fields,
                depends_on_field_names=logical_expression.depends_on_field_names,
                generated_fields=logical_expression.generated_fields,
                group_id=logical_expression.group_id,
            )
            physical_expressions.append(expression)

        logger.debug(f"Done substituting LLMFilterRule for {logical_expression}")
        deduped_physical_expressions = set(physical_expressions)

        return deduped_physical_expressions


class AggregateRule(ImplementationRule):
    """
    Substitute the logical expression for an aggregate with its physical counterpart.
    """

    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, Aggregate)
        logger.debug(f"AggregateRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting AggregateRule for {logical_expression}")

        logical_op = logical_expression.operator
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )

        op = None
        if logical_op.agg_func == AggFunc.COUNT:
            op = CountAggregateOp(**op_kwargs)
        elif logical_op.agg_func == AggFunc.AVERAGE:
            op = AverageAggregateOp(**op_kwargs)
        else:
            raise Exception(f"Cannot support aggregate function: {logical_op.agg_func}")

        expression = PhysicalExpression(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            depends_on_field_names=logical_expression.depends_on_field_names,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )

        logger.debug(f"Done substituting AggregateRule for {logical_expression}")
        deduped_physical_expressions = set([expression])

        return deduped_physical_expressions


class BasicSubstitutionRule(ImplementationRule):
    """
    For logical operators with a single physical implementation, substitute the
    logical expression with its physical counterpart.
    """

    LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP = {
        BaseScan: MarshalAndScanDataOp,
        CacheScan: CacheScanDataOp,
        LimitScan: LimitScanOp,
        Project: ProjectOp,
        GroupByAggregate: ApplyGroupByOp,
        MapScan: MapOp,
    }

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op_class = logical_expression.operator.__class__
        is_match = logical_op_class in cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP
        logger.debug(f"BasicSubstitutionRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> set[PhysicalExpression]:
        logger.debug(f"Substituting BasicSubstitutionRule for {logical_expression}")

        logical_op = logical_expression.operator
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": physical_op_params["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
            }
        )
        physical_op_class = cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP[logical_op.__class__]
        op = physical_op_class(**op_kwargs)

        expression = PhysicalExpression(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            depends_on_field_names=logical_expression.depends_on_field_names,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )

        logger.debug(f"Done substituting BasicSubstitutionRule for {logical_expression}")
        deduped_physical_expressions = set([expression])

        return deduped_physical_expressions
