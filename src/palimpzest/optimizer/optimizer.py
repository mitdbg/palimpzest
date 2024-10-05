from __future__ import annotations
from palimpzest.constants import OptimizationStrategy
from palimpzest.datamanager import DataDirectory
from palimpzest.datasources import DataSource
from palimpzest.operators import *
from palimpzest.optimizer import (
    LogicalExpression,
    Group,
    PhysicalPlan,
    SentinelPlan,
    IMPLEMENTATION_RULES,
    TRANSFORMATION_RULES,
)
from palimpzest.optimizer.rules import *
from palimpzest.optimizer.tasks import *
from palimpzest.optimizer.cost_model import *
from palimpzest.policy import Policy
from palimpzest.sets import Dataset, Set
from palimpzest.utils import getChampionModel, getCodeChampionModel, getConventionalFallbackModel

from copy import deepcopy
from typing import List

# DEFINITIONS
# NOTE: the name pz.Dataset has always been a bit awkward; from a user-facing perspective,
#       it makes sense for users to define a Dataset and then perform operations (e.g. convert,
#       filter, etc.) over that dataset. The awkwardness arises from the fact that the "Dataset"
#       doesn't actually contain data, but instead represents a declarative statement of a query plan
#       which we then manipulate internally. For now, the simplest thing for me to do is simply to
#       rename the class internally to make function signatures a bit clearer, but we may want to
#       revisit the naming of Dataset.
QueryPlan = Dataset

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
            no_cache: bool=False,
            verbose: bool=False,
            available_models: List[Model]=[],
            allow_bonded_query: bool=True,
            allow_conventional_query: bool=False,
            allow_code_synth: bool=True,
            allow_token_reduction: bool=True,
            optimization_strategy: OptimizationStrategy=OptimizationStrategy.OPTIMAL,
        ):
        # store the policy
        self.policy = policy

        # store the cost model
        self.cost_model = cost_model

        # mapping from each group id to its Group object
        self.groups = {}

        # mapping from each expression to its Expression object
        self.expressions = {}

        # the stack of tasks to perform during optimization
        self.tasks_stack = []

        # the lists of implementation and transformation rules that the optimizer can apply
        self.implementation_rules = IMPLEMENTATION_RULES
        self.transformation_rules = TRANSFORMATION_RULES

        # if we are doing SENTINEL / NONE optimization; remove transformation rules
        if optimization_strategy in [OptimizationStrategy.SENTINEL, OptimizationStrategy.NONE]:
            self.transformation_rules = []

        # if we are not performing optimization, set available models to be single model
        # and remove all optimizations (except for bonded queries)
        if optimization_strategy == OptimizationStrategy.NONE:
            self.allow_bonded_query = True
            self.allow_conventional_query = False
            self.allow_code_synth = False
            self.allow_token_reduction = False
            self.available_models = [available_models[0]]

        # store optimization hyperparameters
        self.no_cache = no_cache
        self.verbose = verbose
        self.available_models = available_models
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.optimization_strategy = optimization_strategy

        # prune implementation rules based on boolean flags
        if not self.allow_bonded_query:
            self.implementation_rules = [
                rule for rule in self.implementation_rules
                if rule not in [LLMConvertBondedRule, TokenReducedConvertBondedRule]
            ]

        if not self.allow_conventional_query:
            self.implementation_rules = [
                rule for rule in self.implementation_rules
                if rule not in [LLMConvertConventionalRule, TokenReducedConvertConventionalRule]
            ]

        if not self.allow_code_synth:
            self.implementation_rules = [
                rule for rule in self.implementation_rules
                if not issubclass(rule, CodeSynthesisConvertRule)
            ]

        if not self.allow_token_reduction:
            self.implementation_rules = [
                rule for rule in self.implementation_rules
                if not issubclass(rule, TokenReducedConvertRule)
            ]

    def update_cost_model(self, cost_model: CostModel):
        self.cost_model = cost_model

    def get_physical_op_params(self):
        return {
            "verbose": self.verbose,
            "available_models": self.available_models,
            "champion_model": getChampionModel(self.available_models),
            "code_champion_model": getCodeChampionModel(self.available_models),
            "conventional_fallback_model": getConventionalFallbackModel(self.available_models),
        }

    def construct_group_tree(self, dataset_nodes: List[Set]) -> Tuple[List[int], Set[str], Dict[str, Set[str]]]:
        # get node, outputSchema, and inputSchema(if applicable)
        node = dataset_nodes[-1]
        outputSchema = node.schema
        inputSchema = dataset_nodes[-2].schema if len(dataset_nodes) > 1 else None

        ### convert node --> Group ###
        uid = node.universalIdentifier()

        # create the op for the given node
        op: LogicalOperator = None
        if not self.no_cache and DataDirectory().hasCachedAnswer(uid):
            op = CacheScan(dataset_id=uid, outputSchema=outputSchema)

        elif isinstance(node, DataSource):
            op = BaseScan(dataset_id=uid, outputSchema=outputSchema)

        elif node._filter is not None:
            op = FilteredScan(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                filter=node._filter,
                depends_on=node._depends_on,
                targetCacheId=uid,
            )

        elif node._groupBy is not None:
            op = GroupByAggregate(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                gbySig=node._groupBy,
                targetCacheId=uid,
            )

        elif node._aggFunc is not None:
            op = Aggregate(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                aggFunc=node._aggFunc,
                targetCacheId=uid,
            )

        elif node._limit is not None:
            op = LimitScan(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                limit=node._limit,
                targetCacheId=uid,
            )

        elif not outputSchema == inputSchema:
            op = ConvertScan(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                cardinality=node._cardinality,
                udf=node._udf,
                image_conversion=node._image_conversion,
                depends_on=node._depends_on,
                targetCacheId=uid,
            )

        else:
            raise NotImplementedError("No logical operator exists for the specified dataset construction.")

        # compute the input group ids and fields for this node
        input_group_ids, input_group_fields, input_group_properties = (
            self.construct_group_tree(dataset_nodes[:-1])
            if len(dataset_nodes) > 1
            else ([], set(), {})
        )

        # compute the fields added by this operation and all fields
        input_group_short_fields = list(map(lambda full_field: full_field.split(".")[-1], input_group_fields))
        new_fields = set([
            field for field in op.outputSchema.fieldNames(unique=True, id=uid)
            if (field.split(".")[-1] not in input_group_short_fields) or (node._udf is not None)
        ])
        all_fields = new_fields.union(input_group_fields)

        # compute all properties including this operations'
        all_properties = deepcopy(input_group_properties)
        if isinstance(op, FilteredScan):
            # NOTE: we could use op.get_op_id() here, but storing filter strings makes
            #       debugging a bit easier as you can read which filters are in the Group
            op_filter_str = op.filter.getFilterStr()
            if "filters" in all_properties:
                all_properties["filters"].add(op_filter_str)
            else:
                all_properties["filters"] = set([op_filter_str])

        elif isinstance(op, LimitScan):
            op_limit_str = op.get_op_id()
            if "limits" in all_properties:
                all_properties["limits"].add(op_limit_str)
            else:
                all_properties["limits"] = set([op_limit_str])

        # construct the logical expression and group
        logical_expression = LogicalExpression(
            operator=op,
            input_group_ids=input_group_ids,
            input_fields=input_group_fields,
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

        return [group.group_id], all_fields, all_properties


    def convert_query_plan_to_group_tree(self, query_plan: QueryPlan) -> str:
        # Obtain ordered list of datasets
        dataset_nodes = []
        node = query_plan.copy()

        while isinstance(node, Dataset):
            dataset_nodes.append(node)
            node = node._source
        dataset_nodes.append(node)
        dataset_nodes = list(reversed(dataset_nodes))

        # remove unnecessary convert if output schema from data source scan matches
        # input schema for the next operator
        if len(dataset_nodes) > 1 and dataset_nodes[0].schema == dataset_nodes[1].schema:
            dataset_nodes = [dataset_nodes[0]] + dataset_nodes[2:]
            if len(dataset_nodes) > 1:
                dataset_nodes[1]._source = dataset_nodes[0]

        # compute depends_on field for every node
        short_to_full_field_name = {}
        for node_idx, node in enumerate(dataset_nodes):
            # update mapping from short to full field names
            short_field_names = node.schema.fieldNames()
            full_field_names = node.schema.fieldNames(unique=True, id=node.universalIdentifier())
            for short_field_name, full_field_name in zip(short_field_names, full_field_names):
                # set mapping automatically if this is a new field
                if short_field_name not in short_to_full_field_name:
                    short_to_full_field_name[short_field_name] = full_field_name

                # otherwise, update mapping if and only if this is a non-llm convert (which may overwrite an input field)
                elif node_idx > 0 and dataset_nodes[node_idx - 1].schema != node.schema and node._udf is not None:
                    short_to_full_field_name[short_field_name] = full_field_name

            # if the node is a data source, then skip
            if isinstance(node, DataSource):
                continue

            # If the node already has depends_on specified, then resolve each field name to a full (unique) field name
            if len(node._depends_on) > 0:
                node._depends_on = list(map(lambda field: short_to_full_field_name[field], node._depends_on))
                continue

            # otherwise, make the node depend on all upstream nodes
            node._depends_on = set()
            for upstream_node in dataset_nodes[:node_idx]:
                node._depends_on.update(upstream_node.schema.fieldNames(unique=True, id=node.universalIdentifier()))
            node._depends_on = list(node._depends_on)

        # construct tree of groups
        final_group_id, _, _ = self.construct_group_tree(dataset_nodes)

        # check that final_group_id is a singleton
        assert len(final_group_id) == 1
        final_group_id = final_group_id[0]

        return final_group_id

    def heuristic_optimization(self, group_id: int) -> None:
        """
        Apply universally desirable transformations (e.g. filter/projection push-down).
        """
        pass

    def search_optimization_space(self, group_id: int) -> None:
        # begin the search for an optimal plan with a task to optimize the final group
        initial_task = OptimizeGroup(group_id)
        self.tasks_stack.append(initial_task)

        # TODO: conditionally stop when X number of tasks have been executed to limit exhaustive search
        while len(self.tasks_stack) > 0:
            task = self.tasks_stack.pop(-1)
            new_tasks = []
            if isinstance(task, OptimizeGroup):
                new_tasks = task.perform(self.groups)
            elif isinstance(task, ExpandGroup):
                new_tasks = task.perform(self.groups)
            elif isinstance(task, OptimizeLogicalExpression):
                new_tasks = task.perform(self.transformation_rules, self.implementation_rules)
            elif isinstance(task, ApplyRule):
                new_tasks = task.perform(self.groups, self.expressions, **self.get_physical_op_params())
            elif isinstance(task, OptimizePhysicalExpression):
                context = {"optimization_strategy": self.optimization_strategy}
                new_tasks = task.perform(self.cost_model, self.groups, self.policy, context=context)

            self.tasks_stack.extend(new_tasks)


    def get_sentinel_plan(self, group_id: int) -> SentinelPlan:
        """
        Create and return a SentinelPlan object.
        """
        # get all the physical expressions for this group
        phys_exprs = self.groups[group_id].physical_expressions
        phys_op_set = [expr.operator for expr in phys_exprs]

        # if this expression has no inputs (i.e. it is a BaseScan or CacheScan),
        # create and return the physical plan
        best_phys_expr = self.groups[group_id].best_physical_expression
        if len(best_phys_expr.input_group_ids) == 0:
            return SentinelPlan(operator_sets=[phys_op_set])

        # TODO: need to handle joins
        # get the best physical plan(s) for this group's inputs
        best_phys_subplan = SentinelPlan(operator_sets=[])
        for input_group_id in best_phys_expr.input_group_ids:
            input_best_phys_plan = self.get_sentinel_plan(input_group_id)
            best_phys_subplan = SentinelPlan.fromOpsAndSubPlan(best_phys_subplan.operator_sets, input_best_phys_plan)

        # add this operator set to best physical plan and return
        return SentinelPlan.fromOpsAndSubPlan([phys_op_set], best_phys_subplan)


    def get_optimal_physical_plan(self, group_id: int) -> PhysicalPlan:
        """
        Return the best plan with respect to the user provided policy.
        """
        # get the best physical expression for this group
        best_phys_expr = self.groups[group_id].best_physical_expression

        # if this expression has no inputs (i.e. it is a BaseScan or CacheScan),
        # create and return the physical plan
        if len(best_phys_expr.input_group_ids) == 0:
            return PhysicalPlan(operators=[best_phys_expr.operator], plan_cost=best_phys_expr.plan_cost)

        # TODO: need to handle joins
        # get the best physical plan(s) for this group's inputs
        best_phys_subplan = PhysicalPlan(operators=[])
        for input_group_id in best_phys_expr.input_group_ids:
            input_best_phys_plan = self.get_optimal_physical_plan(input_group_id)
            best_phys_subplan = PhysicalPlan.fromOpsAndSubPlan(best_phys_subplan.operators, best_phys_subplan.plan_cost, input_best_phys_plan)

        # add this operator to best physical plan and return
        return PhysicalPlan.fromOpsAndSubPlan([best_phys_expr.operator], best_phys_expr.plan_cost, best_phys_subplan)


    def get_confidence_interval_optimal_plans(self, group_id: int) -> List[PhysicalPlan]:
        """
        Return all physical plans whose upper bound on the primary policy metric is greater than the
        best plan's lower bound on the primary policy metric (subject to satisfying the policy constraint).

        The OptimizePhysicalExpression task guarantees that each group's `ci_best_physical_expressions`
        maintains a list of expressions with overlapping CI's on the primary policy metric (while also
        satisfying the policy constraint).

        This function computes the cross-product of all such expressions across all groups.
        """
        # get all the physical expressions which could be the best for this group
        best_phys_exprs = self.groups[group_id].ci_best_physical_expressions

        best_plans = []
        for phys_expr in best_phys_exprs:
            # if this expression has no inputs (i.e. it is a BaseScan or CacheScan),
            # create the physical plan and append it to the best_plans for this group
            if len(phys_expr.input_group_ids) == 0:
                plan = PhysicalPlan(operators=[phys_expr.operator], plan_cost=phys_expr.plan_cost)
                best_plans.append(plan)

            # otherwise, get the best physical plan(s) for this group's inputs
            else:
                # TODO: need to handle joins
                best_phys_subplans = [PhysicalPlan(operators=[])]
                for input_group_id in phys_expr.input_group_ids:
                    input_best_phys_plans = self.get_confidence_interval_optimal_plans(input_group_id)
                    best_phys_subplans = [
                        PhysicalPlan.fromOpsAndSubPlan(subplan.operators, subplan.plan_cost, input_subplan)
                        for subplan in best_phys_subplans
                        for input_subplan in input_best_phys_plans
                    ]

                # add this operator to best physical plan and return
                for subplan in best_phys_subplans:
                    plan = PhysicalPlan.fromOpsAndSubPlan([phys_expr.operator], phys_expr.plan_cost, subplan)
                    best_plans.append(plan)

        return best_plans


    def optimize(self, query_plan: QueryPlan) -> List[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        """
        # compute the initial group tree for the user plan
        final_group_id = self.convert_query_plan_to_group_tree(query_plan)

        # TODO
        # # do heuristic based pre-optimization
        # self.heuristic_optimization(final_group_id)

        # search the optimization space by applying logical and physical transformations to the initial group tree
        self.search_optimization_space(final_group_id)

        # construct the optimal physical plan(s) by traversing the memo table
        plans = []
        if self.optimization_strategy == OptimizationStrategy.SENTINEL:
            plans = [self.get_sentinel_plan(final_group_id)]

        elif self.optimization_strategy == OptimizationStrategy.OPTIMAL:
            plans = [self.get_optimal_physical_plan(final_group_id)]

        elif self.optimization_strategy == OptimizationStrategy.CONFIDENCE_INTERVAL:
            plans = self.get_confidence_interval_optimal_plans(final_group_id)

        elif self.optimization_strategy == OptimizationStrategy.NONE:
            plans = [self.get_optimal_physical_plan(final_group_id)]

        return plans
