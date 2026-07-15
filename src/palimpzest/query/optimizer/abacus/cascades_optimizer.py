from __future__ import annotations

import logging
from copy import deepcopy

from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import get_schema_field_names
from palimpzest.core.models import ExecutionStats
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType
from palimpzest.query.operators.logical import (
    ComputeOperator,
    ConvertScan,
    Distinct,
    FilteredScan,
    JoinOp,
    LimitScan,
    Project,
    SearchOperator,
)
from palimpzest.query.optimizer.cost_model import BaseCostModel, SampleBasedCostModel
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.plan import PhysicalPlan

from palimpzest.query.optimizer.abacus import (
    IMPLEMENTATION_RULES,
    TRANSFORMATION_RULES,
)
from palimpzest.validator.validator import Validator

from palimpzest.query.plan import SentinelPlan
from .primitives import Group, LogicalExpression
from .rules import (
    CritiqueAndRefineRule,
    LLMConvertBondedRule,
    MixtureOfAgentsRule,
    RAGRule,
    SplitRule,
)
from .tasks import (
    ApplyRule,
    ExploreGroup,
    OptimizeGroup,
    OptimizeLogicalExpression,
    OptimizePhysicalExpression,
)

from ..optimizer import Optimizer

logger = logging.getLogger(__name__)



class CascadesOptimizer(Optimizer):
    """

    This optimizer is modeled after the Cascades framework for top-down query optimization:
    - Thesis describing Cascades implementation (Chapters 1-3):
      https://15721.courses.cs.cmu.edu/spring2023/papers/17-optimizer2/xu-columbia-thesis1998.pdf

    - Andy Pavlo lecture with walkthrough example: https://www.youtube.com/watch?v=PXS49-tFLcI

    - Original Paper: https://www.cse.iitb.ac.in/infolab/Data/Courses/CS632/2015/Papers/Cascades-graefe.pdf
    """

    def __init__(self, *args, **kwargs):
        # Init base Optimizer class
        super().__init__(*args, **kwargs)

        # mapping from each group id to its Group object
        self.groups = {}

        # mapping from each expression to its Expression object
        self.expressions = {}

        # the stack of tasks to perform during optimization
        self.tasks_stack = []

        # the lists of implementation and transformation rules that the optimizer can apply
        self.implementation_rules = IMPLEMENTATION_RULES
        self.transformation_rules = TRANSFORMATION_RULES

        # remove transformation rules for optimization strategies which do not require them
        if self.optimizer_strategy.no_transformation():
            self.transformation_rules = []

        # prune implementation rules based on boolean flags
        if not self.allow_bonded_query:
            self.implementation_rules = [
                rule
                for rule in self.implementation_rules
                if rule not in [LLMConvertBondedRule]
            ]

        if not self.allow_rag_reduction:
            self.implementation_rules = [
                rule
                for rule in self.implementation_rules
                if not issubclass(rule, RAGRule)
            ]

        if not self.allow_mixtures:
            self.implementation_rules = [
                rule
                for rule in self.implementation_rules
                if not issubclass(rule, MixtureOfAgentsRule)
            ]

        if not self.allow_critic:
            self.implementation_rules = [
                rule
                for rule in self.implementation_rules
                if not issubclass(rule, CritiqueAndRefineRule)
            ]

        if not self.allow_split_merge:
            self.implementation_rules = [
                rule
                for rule in self.implementation_rules
                if not issubclass(rule, SplitRule)
            ]

        logger.info(f"Initialized Optimizer with verbose={self.verbose}")
        logger.debug(f"Initialized Optimizer with params: {self.__dict__}")

    def update_strategy(self, optimizer_strategy: OptimizationStrategyType):
        super().update_strategy(optimizer_strategy)
        # remove transformation rules for optimization strategies which do not require them
        if optimizer_strategy.no_transformation():
            self.transformation_rules = []

    def construct_group_tree(
        self, dataset: Dataset
    ) -> tuple[int, dict[str, FieldInfo], dict[str, set[str]]]:
        logger.debug(f"Constructing group tree for dataset: {dataset}")
        ### convert node --> Group ###
        # create the op for the given node
        op = dataset._operator

        # compute the input group id(s) and field(s) for this node
        if len(dataset._sources) == 0:
            input_group_ids, input_group_fields, input_group_properties = ([], {}, {})
        elif len(dataset._sources) == 1:
            input_group_id, input_group_fields, input_group_properties = (
                self.construct_group_tree(dataset._sources[0])
            )
            input_group_ids = [input_group_id]
        elif len(dataset._sources) == 2:
            (
                left_input_group_id,
                left_input_group_fields,
                left_input_group_properties,
            ) = self.construct_group_tree(dataset._sources[0])
            (
                right_input_group_id,
                right_input_group_fields,
                right_input_group_properties,
            ) = self.construct_group_tree(dataset._sources[1])
            input_group_ids = [left_input_group_id, right_input_group_id]
            input_group_fields = {**left_input_group_fields, **right_input_group_fields}
            input_group_properties = deepcopy(left_input_group_properties)
            for k, v in right_input_group_properties.items():
                if k in input_group_properties:
                    input_group_properties[k].update(v)
                else:
                    input_group_properties[k] = deepcopy(v)
        else:
            raise NotImplementedError(
                "Constructing group trees for datasets with more than 2 sources is not supported."
            )

        # compute the fields added by this operation and all fields
        input_group_short_field_names = list(
            map(lambda full_field: full_field.split(".")[-1], input_group_fields.keys())
        )
        new_fields = {
            field_name: op.output_schema.model_fields[field_name.split(".")[-1]]
            for field_name in get_schema_field_names(op.output_schema, id=dataset.id)
            if (field_name not in input_group_short_field_names)
            or (hasattr(op, "udf") and op.udf is not None)
        }
        all_fields = {**input_group_fields, **new_fields}

        # compute the set of (short) field names this operation depends on
        depends_on_field_names = (
            {}
            if dataset.is_root
            else {field_name.split(".")[-1] for field_name in op.depends_on}
        )

        # NOTE: group_id is computed as the unique (sorted) set of fields and properties;
        #       If an operation does not modify the fields (or modifies them in a way that
        #       can create an idential field set to an earlier group) then we must add an
        #       id from the operator to disambiguate the two groups.
        # compute all properties including this operations'
        all_properties = deepcopy(input_group_properties)
        if isinstance(op, ConvertScan) and sorted(
            op.input_schema.model_fields.keys()
        ) == sorted(op.output_schema.model_fields.keys()):
            model_fields_dict = {
                k: {
                    "annotation": v.annotation,
                    "default": v.default,
                    "description": v.description,
                }
                for k, v in op.output_schema.model_fields.items()
            }
            if "maps" in all_properties:
                all_properties["maps"].add(model_fields_dict)
            else:
                all_properties["maps"] = set([model_fields_dict])

        elif isinstance(op, FilteredScan):
            # NOTE: we could use op.get_full_op_id() here, but storing filter strings makes
            #       debugging a bit easier as you can read which filters are in the Group
            op_filter_str = op.filter.get_filter_str()
            if "filters" in all_properties:
                all_properties["filters"].add(op_filter_str)
            else:
                all_properties["filters"] = set([op_filter_str])

        elif isinstance(op, JoinOp):
            unique_join_str = (
                str(sorted(op.on)) if op.condition is None else op.condition
            )
            if "joins" in all_properties:
                all_properties["joins"].add(unique_join_str)
            else:
                all_properties["joins"] = set([unique_join_str])

        elif isinstance(op, LimitScan):
            op_limit_str = op.get_logical_op_id()
            if "limits" in all_properties:
                all_properties["limits"].add(op_limit_str)
            else:
                all_properties["limits"] = set([op_limit_str])

        elif isinstance(op, Project):
            op_project_str = op.get_logical_op_id()
            if "projects" in all_properties:
                all_properties["projects"].add(op_project_str)
            else:
                all_properties["projects"] = set([op_project_str])

        elif isinstance(op, Distinct):
            op_distinct_str = op.get_logical_op_id()
            if "distincts" in all_properties:
                all_properties["distincts"].add(op_distinct_str)
            else:
                all_properties["distincts"] = set([op_distinct_str])

        # TODO: temporary fix; perhaps use op_ids to identify group?
        elif isinstance(op, ComputeOperator):
            op_instruction = op.instruction
            if "instructions" in all_properties:
                all_properties["instructions"].add(op_instruction)
            else:
                all_properties["instructions"] = set([op_instruction])

        elif isinstance(op, SearchOperator):
            op_search_query = op.search_query
            if "search_queries" in all_properties:
                all_properties["search_queries"].add(op_search_query)
            else:
                all_properties["search_queries"] = set([op_search_query])

        # construct the logical expression and group
        logical_expression = LogicalExpression(
            operator=op,
            input_group_ids=input_group_ids,
            input_fields=input_group_fields,
            depends_on_field_names=depends_on_field_names,
            generated_fields=new_fields,
            group_id=None,
        )
        group = Group(
            logical_expressions=[logical_expression],
            fields=all_fields,
            properties=all_properties,
        )
        logical_expression.set_group_id(group.group_id)

        # add the expression and group to the optimizer's expressions and groups and return
        self.expressions[logical_expression.expr_id] = logical_expression
        self.groups[group.group_id] = group
        logger.debug(f"Constructed group tree for dataset: {dataset}")
        logger.debug(f"Group: {group.group_id}, {all_fields}, {all_properties}")

        return group.group_id, all_fields, all_properties

    def convert_query_plan_to_group_tree(self, dataset: Dataset) -> str:
        logger.debug(f"Converting query plan to group tree for dataset: {dataset}")

        # compute depends_on field for every node
        short_to_full_field_name = {}
        for node in dataset:
            # update mapping from short to full field names
            short_field_names = get_schema_field_names(node.schema)
            full_field_names = get_schema_field_names(node.schema, id=node.id)
            for short_field_name, full_field_name in zip(
                short_field_names, full_field_names
            ):
                # set mapping automatically if this is a new field
                if short_field_name not in short_to_full_field_name or (
                    hasattr(node._operator, "udf") and node._operator.udf is not None
                ):
                    short_to_full_field_name[short_field_name] = full_field_name

            # if the node is a root Dataset, then skip
            if node.is_root:
                continue

            # If the node already has depends_on specified, then resolve each field name to a full (unique) field name
            if len(node._operator.depends_on) > 0:
                node._operator.depends_on = list(
                    map(
                        lambda field: short_to_full_field_name[field],
                        node._operator.depends_on,
                    )
                )
                continue

            # otherwise, make the node depend on all upstream nodes
            node._operator.depends_on = set()
            upstream_nodes = node.get_upstream_datasets()
            for upstream_node in upstream_nodes:
                upstream_field_names = get_schema_field_names(
                    upstream_node.schema, id=upstream_node.id
                )
                node._operator.depends_on.update(upstream_field_names)
            node._operator.depends_on = list(node._operator.depends_on)

        # construct tree of groups
        final_group_id, _, _ = self.construct_group_tree(dataset)

        logger.debug(f"Converted query plan to group tree for dataset: {dataset}")
        logger.debug(f"Final group id: {final_group_id}")

        return final_group_id

    def heuristic_optimization(self, group_id: int) -> None:
        """
        Apply universally desirable transformations (e.g. filter/projection push-down).
        """
        pass

    def search_optimization_space(self, group_id: int) -> None:
        logger.debug(f"Searching optimization space for group_id: {group_id}")

        # begin the search for an optimal plan with a task to optimize the final group
        initial_task = OptimizeGroup(group_id)
        self.tasks_stack.append(initial_task)

        # TODO: conditionally stop when X number of tasks have been executed to limit exhaustive search
        while len(self.tasks_stack) > 0:
            task = self.tasks_stack.pop(-1)

            new_tasks = []
            if isinstance(task, (OptimizeGroup, ExploreGroup)):
                new_tasks = task.perform(self.groups)
            elif isinstance(task, OptimizeLogicalExpression):
                new_tasks = task.perform(
                    self.transformation_rules, self.implementation_rules
                )
            elif isinstance(task, ApplyRule):
                context = {
                    "costed_full_op_ids": self.cost_model.get_costed_full_op_ids()
                }
                new_tasks = task.perform(
                    self.groups,
                    self.expressions,
                    context=context,
                    **self.get_physical_op_params(),
                )
            elif isinstance(task, OptimizePhysicalExpression):
                context = {
                    "optimizer_strategy": self.optimizer_strategy,
                    "execution_strategy": self.execution_strategy,
                }
                new_tasks = task.perform(
                    self.cost_model, self.groups, self.policy, context=context
                )

            self.tasks_stack.extend(new_tasks)

        logger.debug(f"Done searching optimization space for group_id: {group_id}")


class NaiveOptimizer(CascadesOptimizer):
    """
    The difference between the NaiveOptimizer and the AbacusOptimizer is that the NaiveOptimizer does not use a cost model to guide the search for an optimal plan. Instead, it exhaustively searches the space of logical and physical plans, applying all applicable transformation and implementation rules to each group in the group tree.
    The cost used to select the optimal operators is based on the naive_cost function, which is a simple heuristic defined for each operator.
    This optimizer uses still the same Cascades group-style optimization framework as the AbacusOptimizer, but it does not perform sampling to update the cost model.
    """

    def optimize(self, dataset: Dataset) -> list[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        """
        logger.info(f"Optimizing query plan: {dataset}")
        # compute the initial group tree for the user plan
        dataset_copy = dataset.copy()
        final_group_id = self.convert_query_plan_to_group_tree(dataset_copy)

        # TODO
        # # do heuristic based pre-optimization
        # self.heuristic_optimization(final_group_id)

        # search the optimization space by applying logical and physical transformations to the initial group tree
        self.search_optimization_space(final_group_id)
        logger.info(f"Getting optimal plans for final group id: {final_group_id}")

        return self.strategy.get_optimal_plans(
            self.groups, final_group_id, self.policy, self.use_final_op_quality
        )


class AbacusOptimizer(CascadesOptimizer):
    """
    The AbacusOptimizer is a specialized version of the CascadesOptimizer that implements
    the Abacus optimization strategy. This means it uses sentinel plans and MAB planning.
    """

    def __init__(
        self, sentinel_execution_strategy: SentinelExecutionStrategy, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sentinel_execution_strategy = sentinel_execution_strategy

    def _create_sentinel_plan(
        self, dataset: Dataset, train_dataset: dict[str, Dataset] | None
    ) -> SentinelPlan:
        """
        Generates and returns a SentinelPlan for the given dataset.
        """
        # use a fresh optimizer so sentinel planning does not inherit mutable search state
        sentinel_optimizer = NaiveOptimizer(
            cost_model=SampleBasedCostModel(),
            **self.optimizer_config.to_optimizer_kwargs(
                optimizer_strategy=OptimizationStrategyType.SENTINEL
            ),
        )

        # create copy of dataset, but change its root Dataset(s) to the validation Dataset(s)
        dataset = dataset.copy()
        if train_dataset is not None:
            dataset._set_root_datasets(train_dataset)
            dataset._generate_unique_logical_op_ids()

        # get the sentinel plan for the given dataset
        sentinel_plans = sentinel_optimizer.optimize(dataset)
        sentinel_plan = sentinel_plans[0]

        return sentinel_plan

    def optimize(
        self,
        dataset: Dataset,
        validator: Validator,
        execution_stats: ExecutionStats,
        train_dataset: dict[str, Dataset] | None,
    ) -> list[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        """
        logger.info(f"Optimizing query plan: {dataset}")

        assert (
            validator is not None
        ), "Validator must be provided for AbacusOptimizer to generate sample execution data."

        # create sentinel plan
        sentinel_plan = self._create_sentinel_plan(dataset, train_dataset)

        # generate sample execution data
        if train_dataset is not None:
            sentinel_plan_stats = (
                self.sentinel_execution_strategy.execute_sentinel_plan(
                    sentinel_plan, train_dataset, validator
                )
            )

        else:
            train_dataset = dataset._get_root_datasets()
            sentinel_plan_stats = (
                self.sentinel_execution_strategy.execute_sentinel_plan(
                    sentinel_plan, train_dataset, validator
                )
            )

        # TODO I think a cleaner design should avoid passing the execution stats object to the optimizer and have the optimizer create its own "OptimizationStats" object.
        # update the execution stats to account for the work done in optimization
        execution_stats.add_plan_stats(sentinel_plan_stats)
        execution_stats.finish_optimization()

        # construct the CostModel with any sample execution data we've gathered
        self.cost_model = SampleBasedCostModel(sentinel_plan_stats, self.verbose)

        # compute the initial group tree for the user plan
        dataset_copy = dataset.copy()
        final_group_id = self.convert_query_plan_to_group_tree(dataset_copy)

        # TODO
        # # do heuristic based pre-optimization
        # self.heuristic_optimization(final_group_id)

        # search the optimization space by applying logical and physical transformations to the initial group tree
        self.search_optimization_space(final_group_id)
        logger.info(f"Getting optimal plans for final group id: {final_group_id}")

        return self.strategy.get_optimal_plans(
            self.groups, final_group_id, self.policy, self.use_final_op_quality
        )
