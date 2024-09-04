
from palimpzest.operators.join import NonLLMJoin
from copy import deepcopy
from itertools import combinations
from typing import Dict, Set, Tuple

from palimpzest.constants import AggFunc, Cardinality, PromptStrategy
from palimpzest.operators.aggregate import ApplyGroupByOp, AverageAggregateOp, CountAggregateOp
from palimpzest.operators.code_synthesis_convert import CodeSynthesisConvertSingle
from palimpzest.operators.convert import LLMConvertBonded, LLMConvertConventional, NonLLMConvert
from palimpzest.operators.datasource import CacheScanDataOp, MarshalAndScanDataOp
from palimpzest.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.operators.limit import LimitScanOp
from palimpzest.operators.logical import (
    Aggregate,
    BaseScan,
    CacheScan,
    ConvertScan,
    FilteredScan,
    GroupByAggregate,
    LimitScan,
    RetrieveScan,
)
from palimpzest.operators.mixture_of_agents_convert import MixtureOfAgentsConvert
from palimpzest.operators.retrieve import RetrieveOp
from palimpzest.operators.token_reduction_convert import TokenReducedConvertBonded, TokenReducedConvertConventional
from palimpzest.optimizer.primitives import Expression, Group, LogicalExpression, PhysicalExpression
from palimpzest.utils.model_helpers import get_models, get_vision_models


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
    def substitute(logical_expression: LogicalExpression, **kwargs) -> Set[Expression]:
        raise NotImplementedError("Calling this method from an abstract base class.")


class TransformationRule(Rule):
    """
    Base class for transformation rules which convert a logical expression to another logical expression.
    The substitute method for a TransformationRule should return all new expressions and all new groups
    which are created during the substitution.
    """

    @staticmethod
    def substitute(
        logical_expression: LogicalExpression, groups: Dict[int, Group], expressions: Dict[int, Expression], **kwargs
    ) -> Tuple[Set[LogicalExpression], Set[Group]]:
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
        return isinstance(logical_expression.operator, FilteredScan)

    @staticmethod
    def substitute(
        logical_expression: LogicalExpression, groups: Dict[int, Group], expressions: Dict[int, Expression], **kwargs
    ) -> Tuple[Set[LogicalExpression], Set[Group]]:
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
            logical_exprs = input_group.logical_expressions.copy()
            for expr in logical_exprs:
                # if the expression operator is not a convert or a filter, we cannot swap
                if not (isinstance(expr.operator, (ConvertScan, FilteredScan))):
                    continue

                # if this filter depends on a field generated by the expression we're trying to swap with, we can't swap
                if any([field in expr.generated_fields for field in filter_operator.depends_on]):
                    continue

                # create new logical expression with filter pushed down to the input group's logical expression
                new_input_group_ids = expr.input_group_ids.copy()
                new_input_fields = expr.input_fields.copy()
                new_generated_fields = logical_expression.generated_fields.copy()
                new_filter_expr = LogicalExpression(
                    filter_operator,
                    input_group_ids=new_input_group_ids,
                    input_fields=new_input_fields,
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
                    all_fields = new_input_fields.union(new_generated_fields)

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
                    expr.operator,
                    input_group_ids=[group_id]
                    + [g_id for g_id in logical_expression.input_group_ids if g_id != input_group_id],
                    input_fields=group.fields,
                    generated_fields=expr.generated_fields,
                    group_id=logical_expression.group_id,
                )

                # add newly created expression to set of returned expressions
                new_logical_expressions.add(new_expr)

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
        return isinstance(logical_expression.operator, ConvertScan) and logical_expression.operator.udf is not None

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
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
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )

        return set([expression])


class LLMConvertRule(ImplementationRule):
    """
    Base rule for bonded and conventional LLM convert operators; the physical convert class
    (LLMConvertBonded or LLMConvertConventional) is provided by sub-class rules.

    NOTE: we provide the physical convert class(es) in their own sub-classed rules to make
    it easier to allow/disallow groups of rules at the Optimizer level.
    """

    # overridden by sub-classes
    physical_convert_class = None

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        return isinstance(logical_expression.operator, ConvertScan) and logical_expression.operator.udf is None

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
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

        physical_expressions = []
        for model in physical_op_params["available_models"]:
            # skip this model if:
            # 1. this is a pure image model and we're not doing an image conversion, or
            # 2. this is a pure text model and we're doing an image conversion
            is_image_conversion = op_kwargs['image_conversion']
            if (model in pure_text_models and is_image_conversion) or (model in pure_vision_models and not is_image_conversion):
                continue

            # construct multi-expression
            op = cls.physical_convert_class(
                model=model,
                prompt_strategy=PromptStrategy.DSPY_COT_QA,
                **op_kwargs,
            )
            expression = PhysicalExpression(
                operator=op,
                input_group_ids=logical_expression.input_group_ids,
                input_fields=logical_expression.input_fields,
                generated_fields=logical_expression.generated_fields,
                group_id=logical_expression.group_id,
            )
            physical_expressions.append(expression)

        return set(physical_expressions)


class LLMConvertBondedRule(LLMConvertRule):
    """
    Substitute a logical expression for a ConvertScan with a bonded convert physical implementation.
    """

    physical_convert_class = LLMConvertBonded


class LLMConvertConventionalRule(LLMConvertRule):
    """
    Substitute a logical expression for a ConvertScan with a conventional convert physical implementation.
    """

    physical_convert_class = LLMConvertConventional


class TokenReducedConvertRule(ImplementationRule):
    """
    Base rule for bonded and conventional token reduced convert operators; the physical convert class
    (TokenReducedConvertBonded or TokenReducedConvertConventional) is provided by sub-class rules.

    NOTE: we provide the physical convert class(es) in their own sub-classed rules to make
    it easier to allow/disallow groups of rules at the Optimizer level.
    """

    physical_convert_class = None  # overriden by sub-classes
    token_budgets = [0.1, 0.5, 0.9]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        return isinstance(logical_op, ConvertScan) and not logical_op.image_conversion and logical_op.udf is None

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
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
                op = cls.physical_convert_class(
                    model=model,
                    prompt_strategy=PromptStrategy.DSPY_COT_QA,
                    token_budget=token_budget,
                    **op_kwargs,
                )
                expression = PhysicalExpression(
                    operator=op,
                    input_group_ids=logical_expression.input_group_ids,
                    input_fields=logical_expression.input_fields,
                    generated_fields=logical_expression.generated_fields,
                    group_id=logical_expression.group_id,
                )
                physical_expressions.append(expression)

        return set(physical_expressions)


class TokenReducedConvertBondedRule(TokenReducedConvertRule):
    """
    Substitute a logical expression for a ConvertScan with a bonded token reduced physical implementation.
    """

    physical_convert_class = TokenReducedConvertBonded


class TokenReducedConvertConventionalRule(TokenReducedConvertRule):
    """
    Substitute a logical expression for a ConvertScan with a conventional token reduced physical implementation.
    """

    physical_convert_class = TokenReducedConvertConventional


class CodeSynthesisConvertRule(ImplementationRule):
    """
    Base rule for code synthesis convert operators; the physical convert class
    (TokenReducedConvertBonded or TokenReducedConvertConventional) is provided by sub-class rules.

    NOTE: we provide the physical convert class(es) in their own sub-classed rules to make
    it easier to allow/disallow groups of rules at the Optimizer level.
    """

    physical_convert_class = None  # overriden by sub-classes

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        return (
            isinstance(logical_op, ConvertScan)
            and not logical_op.image_conversion
            and logical_op.cardinality != Cardinality.ONE_TO_MANY
            and logical_op.udf is None
        )

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
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
            conventional_fallback_model=physical_op_params["conventional_fallback_model"],
            prompt_strategy=PromptStrategy.DSPY_COT_QA,
            **op_kwargs,
        )
        expression = PhysicalExpression(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )

        return set([expression])


class CodeSynthesisConvertSingleRule(CodeSynthesisConvertRule):
    """
    Substitute a logical expression for a ConvertScan with a (single) code synthesis physical implementation.
    """

    physical_convert_class = CodeSynthesisConvertSingle


class MixtureOfAgentsConvertRule(ImplementationRule):
    """
    Implementation rule for the MixtureOfAgentsConvert operator.
    """
    num_proposer_models = [1, 2, 3]
    temperatures = [0.0, 0.4, 0.8]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        return isinstance(logical_op, ConvertScan) and logical_op.udf is None

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
        logical_op = logical_expression.operator

        # get initial set of parameters for physical op
        op_kwargs: dict = logical_op.get_logical_op_params()
        op_kwargs.update({
            "verbose": physical_op_params['verbose'],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_name": logical_op.logical_op_name(),
        })

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())

        # construct set of proposer models and set of aggregator models
        is_image_conversion = op_kwargs['image_conversion']
        proposer_model_set = vision_models if is_image_conversion else text_models
        aggregator_model_set = text_models

        # filter un-available models out of sets
        proposer_model_set = {model for model in proposer_model_set if model in physical_op_params['available_models']}
        aggregator_model_set = {model for model in aggregator_model_set if model in physical_op_params['available_models']}

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
                            **op_kwargs,
                        )
                        expression = PhysicalExpression(
                            operator=op,
                            input_group_ids=logical_expression.input_group_ids,
                            input_fields=logical_expression.input_fields,
                            generated_fields=logical_expression.generated_fields,
                            group_id=logical_expression.group_id,
                        )
                        physical_expressions.append(expression)

        return set(physical_expressions)


class RetrieveRule(ImplementationRule):
    """
    Substitute a logical expression for a RetrieveScan with a Retrieve physical implementation.
    """
    k_budgets = [1, 3, 5, 10]

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        return (
            isinstance(logical_expression.operator, RetrieveScan)
        )

    @classmethod
    def substitute(
        cls, logical_expression: LogicalExpression, **physical_op_params
    ) -> Set[PhysicalExpression]:
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
                generated_fields=logical_expression.generated_fields,
                group_id=logical_expression.group_id,
            )

            physical_expressions.append(expression)

        return set(physical_expressions)


class NonLLMFilterRule(ImplementationRule):
    """
    Substitute a logical expression for a FilteredScan with a non-llm filter physical implementation.
    """

    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        return (
            isinstance(logical_expression.operator, FilteredScan)
            and logical_expression.operator.filter.filter_fn is not None
        )

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
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
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )
        return set([expression])


class LLMFilterRule(ImplementationRule):
    """
    Substitute a logical expression for a FilteredScan with an llm filter physical implementation.
    """

    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        return (
            isinstance(logical_expression.operator, FilteredScan)
            and logical_expression.operator.filter.filter_condition is not None
        )

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
        logical_op = logical_expression.operator
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update({
            "verbose": physical_op_params["verbose"],
            "logical_op_id": logical_op.get_logical_op_id(),
            "logical_op_name": logical_op.logical_op_name(),
        })

        # NOTE: when comparing pz.Model(s), equality is determined by the string (i.e. pz.Model.value)
        #       thus, Model.GPT_4o and Model.GPT_4o_V map to the same value; this allows us to use set logic
        #
        # identify models which can be used strictly for text or strictly for images
        vision_models = set(get_vision_models())
        text_models = set(get_models())
        pure_text_models = {model for model in text_models if model not in vision_models}
        pure_vision_models = {model for model in vision_models if model not in text_models}

        physical_expressions = []
        for model in physical_op_params["available_models"]:
            # skip this model if:
            # 1. this is a pure image model and we're not doing an image conversion, or
            # 2. this is a pure text model and we're doing an image conversion
            is_image_filter = op_kwargs['image_filter']
            if (model in pure_text_models and is_image_filter) or (model in pure_vision_models and not is_image_filter):
                continue

            # construct multi-expression
            op = LLMFilter(
                model=model,
                prompt_strategy=PromptStrategy.DSPY_COT_BOOL,
                **op_kwargs,
            )
            expression = PhysicalExpression(
                operator=op,
                input_group_ids=logical_expression.input_group_ids,
                input_fields=logical_expression.input_fields,
                generated_fields=logical_expression.generated_fields,
                group_id=logical_expression.group_id,
            )
            physical_expressions.append(expression)

        return set(physical_expressions)


class AggregateRule(ImplementationRule):
    """
    Substitute the logical expression for an aggregate with its physical counterpart.
    """

    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        return isinstance(logical_expression.operator, Aggregate)

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
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
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )
        return set([expression])

class JoinRule(ImplementationRule):
    """
    Substitute the logical expression for a join with its physical counterpart.
    """
    @staticmethod
    def matches_pattern(logical_expression: LogicalExpression) -> bool:
        return isinstance(logical_expression.operator, Join)

    @staticmethod
    def substitute(logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
        logical_op = logical_expression.operator
        op_kwargs = logical_op.get_op_params()
        op_kwargs.update({
            "verbose": physical_op_params['verbose'],
            "logical_op_id": logical_op.get_op_id(),
        })

        op = NonLLMJoin(**op_kwargs)

        expression = PhysicalExpression(
            operator=op,
            input_group_ids=logical_expression.input_group_ids,
            input_fields=logical_expression.input_fields,
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )
        return set([expression])


class BasicSubstitutionRule(ImplementationRule):
    """
    For logical operators with a single physical implementation, substitute the
    logical expression with its physical counterpart.
    """

    LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP = {
        BaseScan: MarshalAndScanDataOp,
        CacheScan: CacheScanDataOp,
        LimitScan: LimitScanOp,
        GroupByAggregate: ApplyGroupByOp,
    }

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op_class = logical_expression.operator.__class__
        return logical_op_class in cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **physical_op_params) -> Set[PhysicalExpression]:
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
            generated_fields=logical_expression.generated_fields,
            group_id=logical_expression.group_id,
        )
        return set([expression])
