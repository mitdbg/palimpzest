from __future__ import annotations
from palimpzest.constants import OptimizationStrategy
from palimpzest.cost_model import CostModel
from palimpzest.datamanager import DataDirectory
from palimpzest.datasources import DataSource
from palimpzest.operators import *
from palimpzest.optimizer import (
    LogicalExpression,
    Group,
    PhysicalPlan,
    IMPLEMENTATION_RULES,
    TRANSFORMATION_RULES,
)
from palimpzest.optimizer.rules import *
from palimpzest.optimizer.tasks import *
from palimpzest.policy import Policy
from palimpzest.sets import Dataset, Set
from palimpzest.utils import getChampionModel, getCodeChampionModel, getConventionalFallbackModel

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

    NOTE: the optimizer currently makes the following assumptions:
    1. unique field names across schemas --> this must currently be enforced by the programmer,
      but we should quickly move to standardizing field names to be "[{source_name}.]{schema_name}.{field_name}"
      - this^ would relax our assumption to be that fields are unique for a given (source, schema), which
        I believe is very reasonable
    
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
            shouldProfile: bool=True,
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

        # store optimization hyperparameters
        self.no_cache = no_cache
        self.verbose = verbose
        self.available_models = available_models
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.optimization_strategy = optimization_strategy
        self.shouldProfile = shouldProfile

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
            "shouldProfile": self.shouldProfile,
            "available_models": self.available_models,
            "champion_model": getChampionModel(),
            "code_champion_model": getCodeChampionModel(),
            "conventional_fallback_model": getConventionalFallbackModel(),
        }

    def construct_group_tree(self, dataset_nodes: List[Set]) -> Tuple[List[int], Set[str], Set[str]]:
        # get node, outputSchema, and inputSchema(if applicable)
        node = dataset_nodes[-1]
        outputSchema = node.schema
        inputSchema = dataset_nodes[-2].schema if len(dataset_nodes) > 1 else None

        ### convert node --> Group ###
        uid = node.universalIdentifier()

        # create the op for the given node
        op: LogicalOperator = None
        if not self.no_cache and DataDirectory().hasCachedAnswer(uid):
            op = CacheScan(cachedDataIdentifier=uid, outputSchema=outputSchema)

        elif isinstance(node, DataSource):
            op = BaseScan(datasetIdentifier=uid, outputSchema=outputSchema)

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
        input_group_ids, input_group_fields, input_group_filter_strs = (
            self.construct_group_tree(dataset_nodes[:-1])
            if len(dataset_nodes) > 1
            else ([], set(), set())
        )

        # compute the fields added by this operation and all fields
        new_fields = set([
            field for field in op.outputSchema.fieldNames()
            if (field not in input_group_fields) or (node._udf is not None)
        ])
        all_fields = new_fields.union(input_group_fields)

        # compute all filters including this operation
        all_filter_strs = input_group_filter_strs.copy()
        if isinstance(op, FilteredScan):
            all_filter_strs.update(set([op.filter.getFilterStr()]))

        # construct the logical expression and group
        logical_expression = LogicalExpression(
            operator=op,
            input_group_ids=input_group_ids,
            input_fields=input_group_fields,
            generated_fields=new_fields,
            group_id=None
        )
        group = Group(
            logical_expressions=[logical_expression],
            fields=all_fields,
            filter_strs=all_filter_strs,
        )
        logical_expression.set_group_id(group.group_id)

        # add the expression and group to the optimizer's expressions and groups and return
        self.expressions[logical_expression.get_expr_id()] = logical_expression
        self.groups[group.group_id] = group

        return [group.group_id], all_fields, all_filter_strs


    def convert_query_plan_to_group_tree(self, query_plan: QueryPlan) -> str:
        # Obtain ordered list of datasets
        dataset_nodes = []
        node = query_plan # TODO: copy

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
        for node_idx, node in enumerate(dataset_nodes):
            # if the node is a data source or already has depends_on specified, then skip
            if isinstance(node, DataSource) or len(node._depends_on) > 0:
                continue

            # otherwise, make the node depend on all upstream nodes
            node._depends_on = set()
            for upstream_node in dataset_nodes[:node_idx]:
                node._depends_on.update(upstream_node.schema.fieldNames())
            node._depends_on = list(node._depends_on)

        # construct tree of groups
        final_group_id, _, _ = self.construct_group_tree(dataset_nodes)

        # check that final_group_id is a singleton
        assert len(final_group_id) == 1
        final_group_id = final_group_id[0]

        return final_group_id

    # def heuristic_optimization(self, group_id: int) -> None:
    #     """
    #     Compute the optimal ordering of filters based on the optimization metric and cost model.
    #     """
    #     # NOTE: groups are the wrong vessel for this; the last filter will have all
    #     #       the upstream filters + converts etc. in its group -- we only want to
    #     #       compute the cost of each filter's ancestors in its dependency graph.
    #     #
    #     # compute cost metric for every initial group with a filter operator
    #     filter_group_id_to_metric = []
    #     for group_id, group in self.groups.items():
    #         if isinstance(list(group.logical_expressions)[0].operator, FilteredScan):
    #             metric = self.compute_group_cost_metric(group)
    #             filter_group_id_to_metric.append((group_id, metric))

    #     # sort filter groups by min metric
    #     ordered_filter_group_ids = sorted(filter_group_id_to_metric, key=lambda tup: tup[1])

    #     # re-construct initial group tree
        

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


    def get_pareto_optimal_plans(self, group_id: int) -> List[PhysicalPlan]:
        """
        Return all physical plans who are on the pareto frontier, regardless of whether or not they
        satisfy the policy constraint.

        The OptimizePhysicalExpression task guarantees that each group's `pareto_optimal_physical_expressions`
        maintains a list of expressions with overlapping CI's on the primary policy metric (while also
        satisfying the policy constraint).

        This function computes the cross-product of all such expressions across all groups.
        """
        raise NotImplementedError("Future work")


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

        # TODO: OptimizationStrategy.SENTINEL

        # construct the optimal physical plan(s) by traversing the memo table
        plans = []
        if self.optimization_strategy == OptimizationStrategy.OPTIMAL:
            plans = [self.get_optimal_physical_plan(final_group_id)]

        elif self.optimization_strategy == OptimizationStrategy.CONFIDENCE_INTERVAL:
            plans = self.get_confidence_interval_optimal_plans(final_group_id)

        elif self.optimization_strategy == OptimizationStrategy.PARETO_OPTIMAL:
            raise NotImplementedError("Future work")

        return plans