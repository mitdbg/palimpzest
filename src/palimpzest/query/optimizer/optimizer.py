from __future__ import annotations

import logging
from copy import deepcopy

from palimpzest.constants import Model
from palimpzest.core.data.datareaders import DataReader
from palimpzest.core.lib.fields import Field
from palimpzest.policy import Policy
from palimpzest.query.operators.logical import (
    Aggregate,
    BaseScan,
    ConvertScan,
    FilteredScan,
    GroupByAggregate,
    LimitScan,
    LogicalOperator,
    MapScan,
    Project,
    RetrieveScan,
)
from palimpzest.query.optimizer import (
    IMPLEMENTATION_RULES,
    TRANSFORMATION_RULES,
)
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer_strategy import (
    OptimizationStrategyType,
    OptimizerStrategyRegistry,
)
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.query.optimizer.primitives import Group, LogicalExpression
from palimpzest.query.optimizer.rules import (
    CodeSynthesisConvertRule,
    CriticAndRefineConvertRule,
    LLMConvertBondedRule,
    MixtureOfAgentsConvertRule,
    RAGConvertRule,
    TokenReducedConvertBondedRule,
)
from palimpzest.query.optimizer.tasks import (
    ApplyRule,
    ExpandGroup,
    OptimizeGroup,
    OptimizeLogicalExpression,
    OptimizePhysicalExpression,
)
from palimpzest.sets import Dataset, Set
from palimpzest.utils.hash_helpers import hash_for_serialized_dict
from palimpzest.utils.model_helpers import get_champion_model, get_code_champion_model, get_fallback_model

logger = logging.getLogger(__name__)


def get_node_uid(node: Dataset | DataReader) -> str:
    """Helper function to compute the universal identifier for a node in the query plan."""
    # NOTE: technically, hash_for_serialized_dict(node.serialize()) would be valid for both DataReader and Dataset;
    #       for the moment, I want to be explicit in Dataset about what constitutes a unique Dataset object, but
    #       in ther future we may be able to remove universal_identifier() from Dataset and just use this function
    return node.universal_identifier() if isinstance(node, Dataset) else hash_for_serialized_dict(node.serialize())


class Optimizer:
    """
    The optimizer is responsible for searching the space of possible physical plans
    for a user's initial (logical) plan and selecting the one which is closest to
    optimizing the user's policy objective.

    This optimizer is modeled after the Cascades framework for top-down query optimization:
    - Thesis describing Cascades implementation (Chapters 1-3):
      https://15721.courses.cs.cmu.edu/spring2023/papers/17-optimizer2/xu-columbia-thesis1998.pdf

    - Andy Pavlo lecture with walkthrough example: https://www.youtube.com/watch?v=PXS49-tFLcI

    - Original Paper: https://www.cse.iitb.ac.in/infolab/Data/Courses/CS632/2015/Papers/Cascades-graefe.pdf

    Notably, this optimization framework has served as the backbone of Microsoft SQL Server, CockroachDB,
    and a few other important DBMS systems.

    NOTE: the optimizer currently assumes that field names are unique across schemas; we do try to enforce
          this by rewriting field names underneath-the-hood to be "{schema_name}.{field_name}", but this still
          does not solve a situation in which -- for example -- a user uses the pz.URL schema twice in the same
          program. In order to address that situation, we will need to augment our renaming scheme.
    """

    def __init__(
        self,
        policy: Policy,
        cost_model: CostModel,
        cache: bool = False,
        verbose: bool = False,
        available_models: list[Model] | None = None,
        allow_bonded_query: bool = True,
        allow_code_synth: bool = False,
        allow_token_reduction: bool = False,
        allow_rag_reduction: bool = False,
        allow_mixtures: bool = True,
        allow_critic: bool = False,
        optimization_strategy_type: OptimizationStrategyType = OptimizationStrategyType.PARETO,
        use_final_op_quality: bool = False,  # TODO: make this func(plan) -> final_quality
    ):
        # store the policy
        if available_models is None or len(available_models) == 0:
            available_models = []
        self.policy = policy

        # store the cost model
        self.cost_model = cost_model

        # store the set of physical operators for which our cost model has cost estimates
        self.costed_phys_op_ids = cost_model.get_costed_phys_op_ids()

        # mapping from each group id to its Group object
        self.groups = {}

        # mapping from each expression to its Expression object
        self.expressions = {}

        # the stack of tasks to perform during optimization
        self.tasks_stack = []

        # the lists of implementation and transformation rules that the optimizer can apply
        self.implementation_rules = IMPLEMENTATION_RULES
        self.transformation_rules = TRANSFORMATION_RULES

        self.strategy = OptimizerStrategyRegistry.get_strategy(optimization_strategy_type.value)

        # if we are doing SENTINEL / NONE optimization; remove transformation rules
        if optimization_strategy_type in [OptimizationStrategyType.SENTINEL, OptimizationStrategyType.NONE]:
            self.transformation_rules = []

        # if we are not performing optimization, set available models to be single model
        # and remove all optimizations (except for bonded queries)
        if optimization_strategy_type == OptimizationStrategyType.NONE:
            self.allow_bonded_query = True
            self.allow_code_synth = False
            self.allow_token_reduction = False
            self.allow_rag_reduction = False
            self.allow_mixtures = False
            self.allow_critic = False
            self.available_models = [available_models[0]]

        # store optimization hyperparameters
        self.cache = cache
        self.verbose = verbose
        self.available_models = available_models
        self.allow_bonded_query = allow_bonded_query
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.allow_rag_reduction = allow_rag_reduction
        self.allow_mixtures = allow_mixtures
        self.allow_critic = allow_critic
        self.optimization_strategy_type = optimization_strategy_type
        self.use_final_op_quality = use_final_op_quality

        # prune implementation rules based on boolean flags
        if not self.allow_bonded_query:
            self.implementation_rules = [
                rule
                for rule in self.implementation_rules
                if rule not in [LLMConvertBondedRule, TokenReducedConvertBondedRule]
            ]

        if not self.allow_code_synth:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, CodeSynthesisConvertRule)
            ]

        if not self.allow_token_reduction:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, TokenReducedConvertBondedRule)
            ]

        if not self.allow_rag_reduction:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, RAGConvertRule)
            ]

        if not self.allow_mixtures:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, MixtureOfAgentsConvertRule)
            ]

        if not self.allow_critic:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, CriticAndRefineConvertRule)
            ]

        logger.info(f"Initialized Optimizer with verbose={self.verbose}")
        logger.debug(f"Initialized Optimizer with params: {self.__dict__}")

    def update_cost_model(self, cost_model: CostModel):
        self.cost_model = cost_model
        self.costed_phys_op_ids = cost_model.get_costed_phys_op_ids()

    def get_physical_op_params(self):
        return {
            "verbose": self.verbose,
            "available_models": self.available_models,
            "champion_model": get_champion_model(self.available_models),
            "code_champion_model": get_code_champion_model(self.available_models),
            "fallback_model": get_fallback_model(self.available_models),
        }

    def deepcopy_clean(self):
        optimizer = Optimizer(
            policy=self.policy,
            cost_model=CostModel(),
            cache=self.cache,
            verbose=self.verbose,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            allow_rag_reduction=self.allow_rag_reduction,
            allow_mixtures=self.allow_mixtures,
            allow_critic=self.allow_critic,
            optimization_strategy_type=self.optimization_strategy_type,
            use_final_op_quality=self.use_final_op_quality,
        )
        return optimizer

    def update_strategy(self, optimizer_strategy_type: OptimizationStrategyType):
        self.optimization_strategy_type = optimizer_strategy_type
        self.strategy = OptimizerStrategyRegistry.get_strategy(optimizer_strategy_type.value)

    def construct_group_tree(self, dataset_nodes: list[Set]) -> tuple[list[int], dict[str, Field], dict[str, set[str]]]:
        # get node, output_schema, and input_schema (if applicable)
        logger.debug(f"Constructing group tree for dataset_nodes: {dataset_nodes}")

        node = dataset_nodes[-1]
        output_schema = node.schema
        input_schema = dataset_nodes[-2].schema if len(dataset_nodes) > 1 else None

        ### convert node --> Group ###
        uid = get_node_uid(node)

        # create the op for the given node
        op: LogicalOperator | None = None

        # TODO: add cache scan when we add caching back to PZ
        # if self.cache:
        #     op = CacheScan(datareader=node, output_schema=output_schema)
        if isinstance(node, DataReader):
            op = BaseScan(datareader=node, output_schema=output_schema)
        elif node._filter is not None:
            op = FilteredScan(
                input_schema=input_schema,
                output_schema=output_schema,
                filter=node._filter,
                depends_on=node._depends_on,
                target_cache_id=uid,
            )
        elif node._group_by is not None:
            op = GroupByAggregate(
                input_schema=input_schema,
                output_schema=output_schema,
                group_by_sig=node._group_by,
                target_cache_id=uid,
            )
        elif node._agg_func is not None:
            op = Aggregate(
                input_schema=input_schema,
                output_schema=output_schema,
                agg_func=node._agg_func,
                target_cache_id=uid,
            )
        elif node._limit is not None:
            op = LimitScan(
                input_schema=input_schema,
                output_schema=output_schema,
                limit=node._limit,
                target_cache_id=uid,
            )
        elif node._project_cols is not None:
            op = Project(
                input_schema=input_schema,
                output_schema=output_schema,
                project_cols=node._project_cols,
                target_cache_id=uid,
            )
        elif node._index is not None:
            op = RetrieveScan(
                input_schema=input_schema,
                output_schema=output_schema,
                index=node._index,
                search_func=node._search_func,
                search_attr=node._search_attr,
                output_attr=node._output_attr,
                k=node._k,
                target_cache_id=uid,
            )
        elif output_schema != input_schema:
            op = ConvertScan(
                input_schema=input_schema,
                output_schema=output_schema,
                cardinality=node._cardinality,
                udf=node._udf,
                depends_on=node._depends_on,
                target_cache_id=uid,
            )
        elif output_schema == input_schema and node._udf is not None:
            op = MapScan(
                input_schema=input_schema,
                output_schema=output_schema,
                udf=node._udf,
                target_cache_id=uid,
            )
        # some legacy plans may have a useless convert; for now we simply skip it
        elif output_schema == input_schema:
            return self.construct_group_tree(dataset_nodes[:-1]) if len(dataset_nodes) > 1 else ([], {}, {})
        else:
            raise NotImplementedError(
                f"""No logical operator exists for the specified dataset construction.
                {input_schema}->{output_schema} {"with filter:'" + node._filter + "'" if node._filter is not None else ""}"""
            )

        # compute the input group ids and fields for this node
        input_group_ids, input_group_fields, input_group_properties = (
            self.construct_group_tree(dataset_nodes[:-1]) if len(dataset_nodes) > 1 else ([], {}, {})
        )

        # compute the fields added by this operation and all fields
        input_group_short_field_names = list(
            map(lambda full_field: full_field.split(".")[-1], input_group_fields.keys())
        )
        new_fields = {
            field_name: field
            for field_name, field in op.output_schema.field_map(unique=True, id=uid).items()
            if (field_name.split(".")[-1] not in input_group_short_field_names) or (node._udf is not None)
        }
        all_fields = {**input_group_fields, **new_fields}

        # compute the set of (short) field names this operation depends on
        depends_on_field_names = (
            {} if isinstance(node, DataReader) else {field_name.split(".")[-1] for field_name in node._depends_on}
        )

        # compute all properties including this operations'
        all_properties = deepcopy(input_group_properties)
        if isinstance(op, FilteredScan):
            # NOTE: we could use op.get_op_id() here, but storing filter strings makes
            #       debugging a bit easier as you can read which filters are in the Group
            op_filter_str = op.filter.get_filter_str()
            if "filters" in all_properties:
                all_properties["filters"].add(op_filter_str)
            else:
                all_properties["filters"] = set([op_filter_str])

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

        elif isinstance(op, MapScan):
            op_udf_str = op.udf.__name__
            if "udfs" in all_properties:
                all_properties["udfs"].add(op_udf_str)
            else:
                all_properties["udfs"] = set([op_udf_str])

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
        self.expressions[logical_expression.get_expr_id()] = logical_expression
        self.groups[group.group_id] = group
        logger.debug(f"Constructed group tree for dataset_nodes: {dataset_nodes}")
        logger.debug(f"Group: {group.group_id}, {all_fields}, {all_properties}")

        return [group.group_id], all_fields, all_properties

    def convert_query_plan_to_group_tree(self, query_plan: Dataset) -> str:
        logger.debug(f"Converting query plan to group tree for query_plan: {query_plan}")
        # Obtain ordered list of datasets
        dataset_nodes: list[Dataset | DataReader] = []
        node = query_plan.copy()

        # NOTE: the very first node will be a DataReader; the rest will be Dataset
        while isinstance(node, Dataset):
            dataset_nodes.append(node)
            node = node._source
        dataset_nodes.append(node)
        dataset_nodes = list(reversed(dataset_nodes))

        # compute depends_on field for every node
        short_to_full_field_name = {}
        for node_idx, node in enumerate(dataset_nodes):
            # update mapping from short to full field names
            short_field_names = node.schema.field_names()
            full_field_names = node.schema.field_names(unique=True, id=get_node_uid(node))
            for short_field_name, full_field_name in zip(short_field_names, full_field_names):
                # set mapping automatically if this is a new field
                if short_field_name not in short_to_full_field_name or (
                    node_idx > 0 and dataset_nodes[node_idx - 1].schema != node.schema and node._udf is not None
                ):
                    short_to_full_field_name[short_field_name] = full_field_name

            # if the node is a data source, then skip
            if isinstance(node, DataReader):
                continue

            # If the node already has depends_on specified, then resolve each field name to a full (unique) field name
            if len(node._depends_on) > 0:
                node._depends_on = list(map(lambda field: short_to_full_field_name[field], node._depends_on))
                continue

            # otherwise, make the node depend on all upstream nodes
            node._depends_on = set()
            for upstream_node in dataset_nodes[:node_idx]:
                node._depends_on.update(upstream_node.schema.field_names(unique=True, id=get_node_uid(upstream_node)))
            node._depends_on = list(node._depends_on)

        # construct tree of groups
        final_group_id, _, _ = self.construct_group_tree(dataset_nodes)

        # check that final_group_id is a singleton
        assert len(final_group_id) == 1
        final_group_id = final_group_id[0]
        logger.debug(f"Converted query plan to group tree for query_plan: {query_plan}")
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
            if isinstance(task, (OptimizeGroup, ExpandGroup)):
                new_tasks = task.perform(self.groups)
            elif isinstance(task, OptimizeLogicalExpression):
                new_tasks = task.perform(self.transformation_rules, self.implementation_rules)
            elif isinstance(task, ApplyRule):
                context = {"costed_phys_op_ids": self.costed_phys_op_ids}
                new_tasks = task.perform(
                    self.groups, self.expressions, context=context, **self.get_physical_op_params()
                )
            elif isinstance(task, OptimizePhysicalExpression):
                context = {"optimization_strategy_type": self.optimization_strategy_type}
                new_tasks = task.perform(self.cost_model, self.groups, self.policy, context=context)

            self.tasks_stack.extend(new_tasks)

        logger.debug(f"Done searching optimization space for group_id: {group_id}")

    def optimize(self, query_plan: Dataset, policy: Policy | None = None) -> list[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        """
        logger.info(f"Optimizing query plan: {query_plan}")
        # compute the initial group tree for the user plan
        final_group_id = self.convert_query_plan_to_group_tree(query_plan)

        # TODO
        # # do heuristic based pre-optimization
        # self.heuristic_optimization(final_group_id)

        # search the optimization space by applying logical and physical transformations to the initial group tree
        self.search_optimization_space(final_group_id)
        logger.info(f"Getting optimal plans for final group id: {final_group_id}")

        return self.strategy.get_optimal_plans(self.groups, final_group_id, policy, self.use_final_op_quality)
