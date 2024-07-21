from __future__ import annotations

from palimpzest.planner import LogicalPlan
from .plan import LogicalPlan
from .planner import Planner

import palimpzest as pz
import palimpzest.operators as pz_ops

from itertools import permutations
from typing import List


class DependencyGraphNode:
    def __init__(self, operator: pz_ops.LogicalOperator, op_idx: int):
        self.operator = operator
        self.op_idx = op_idx

        # list of nodes which depend on this logical node
        self.children = []

        # list of nodes which this logical node depends on
        self.parents = []

    def add_child(self, child: DependencyGraphNode):
        self.children.append(child)

    def add_parent(self, parent: DependencyGraphNode):
        self.parents.append(parent)

    def get_ancestors(self):
        all_ancestors = []
        for parent in self.parents:
            all_ancestors.append(parent.op_idx)
            parent_ancestors = parent.get_ancestors()
            all_ancestors.extend(parent_ancestors)

        return all_ancestors

    def prune_redundant_edges(self):
        # prune edges in your children
        for child in self.children:
            child.prune_redundant_edges()

        # prune your own edges
        prune_nodes = []
        for idx, prune_candidate in enumerate(self.parents):
            for other_idx, other in enumerate(self.parents):
                if idx == other_idx:
                    continue

                if prune_candidate.op_idx in other.get_ancestors():
                    prune_nodes.append(prune_candidate)

        prune_node_op_indices = list(map(lambda node: node.op_idx, prune_nodes))
        self.parents = [parent for parent in self.parents if parent.op_idx not in prune_node_op_indices]
        for node in prune_nodes:
            node.children = [child for child in node.children if child.op_idx != self.op_idx]

    def compute_upstream_subplan(self) -> List[DependencyGraphNode]:
        """
        We compute the upstream subplan for the given node.
        """
        upstream_subplan = [self]
        for node in self.parents:
            upstream_nodes = node.compute_upstream_subplan()
            upstream_subplan = upstream_nodes + upstream_subplan

        return upstream_subplan


class LogicalPlanner(Planner):
    def __init__(self, no_cache: bool=False, sentinel: bool=False, verbose: bool=False, *args, **kwargs):
        """A given planner should not have a dataset when it's being generated, since it could be used for multiple datasets.
        However, we currently cannot support this since the plans are stored within a single planner object.
        To support this, we can use a dictionary in the form [dataset -> [Plan, Plan, ...]].
        To discuss for future versions.
        """

        super().__init__(*args, **kwargs)
        self.no_cache = no_cache
        self.sentinel = sentinel
        self.verbose = verbose

    @staticmethod
    def _compute_logical_plan_reorderings(logical_plan: LogicalPlan) -> List[LogicalPlan]:
        """
        Given the naive logical plan, compute a set of equivalent plans with filter
        and convert operations re-ordered. This set is not exhaustive, but it considers
        every possible ordering of filters in order to provide plans with diverse tradeoffs.
        """
        # There are a few rules surrounding which permutation(s) of logical operators are legal:
        # 1. if a filter depends on a field in a convert's outputSchema, it must be executed after the convert
        # 2. if a convert depends on another operation's outputSchema, it must be executed after that operation
        # 3. if depends_on is not specified for a convert operator, it cannot be swapped with another convert
        # 4. if depends_on is not specified for a filter, it can not be swapped with a convert (but it can be swapped w/adjacent filters)

        # compute implicit depends_on relationships, keep in mind that operations closer to the start of the list are executed first;
        # if depends_on is not specified for a convert or filter, it implicitly depends_on all preceding converts
        dependency_graph_nodes = [DependencyGraphNode(op, idx) for idx, op in enumerate(logical_plan.operators)]
        for idx, node in enumerate(dependency_graph_nodes):
            for upstreamOpNode in dependency_graph_nodes[:idx]:
                # if upstream convert generates a field which this node depends on
                # add dependency in DependencyGraph; if we don't have a depends_on specification
                # then we implicitly assume this node depends on the convert
                if isinstance(upstreamOpNode.operator, pz_ops.ConvertScan):
                    if node.operator.depends_on is not None:
                        for field in upstreamOpNode.operator.generated_fields:
                            if field in node.operator.depends_on:
                                upstreamOpNode.add_child(node)
                                node.add_parent(upstreamOpNode)
                                break
                    else:
                        upstreamOpNode.add_child(node)
                        node.add_parent(upstreamOpNode)

                # if upstream node is anything other than a filter, add dependency
                elif not isinstance(upstreamOpNode.operator, pz_ops.FilteredScan):
                    upstreamOpNode.add_child(node)
                    node.add_parent(upstreamOpNode)

        # prune redundant edges from the dependency graph; for example, if we have the edges:
        # S --> C1, C1 --> F1, S --> F1
        # then we can prune S --> F1 because F1 already depends on S through its dependency on C1
        root_node = dependency_graph_nodes[0]
        root_node.prune_redundant_edges()

        # get filter nodes
        filter_nodes = [node for node in dependency_graph_nodes if isinstance(node.operator, pz_ops.FilteredScan)]

        # compute the upstream subplan for each filter
        filter_node_to_upstream_subplan = {node: node.compute_upstream_subplan() for node in filter_nodes}

        # compute the set of nodes that are downstream from all filters
        upstream_node_op_indices = set([
            node.op_idx
            for upstream_subplan in filter_node_to_upstream_subplan.values()
            for node in upstream_subplan
        ])
        downstream_nodes = [node for node in dependency_graph_nodes if node.op_idx not in upstream_node_op_indices]

        # permute the filters (sub-plans)
        filter_orders = permutations(filter_node_to_upstream_subplan.keys())

        # for each permutation of filters, create a logical plan by ordering the upstream plans for each filter
        logical_plans = []
        for filter_order in filter_orders:
            logical_plan_operators, operator_set = [], set()
            for filter_node in filter_order:
                # fetch the upstream subplan for this filter and add its non-redundant nodes to the plan
                upstream_subplan = filter_node_to_upstream_subplan[filter_node]
                for node in upstream_subplan:
                    if node.op_idx not in operator_set:
                        logical_plan_operators.append(node.operator)
                        operator_set.add(node.op_idx)

            # add nodes which are downstream from all filters
            for node in downstream_nodes:
                logical_plan_operators.append(node.operator)

            # otherwise, construct the logical plan and add the operator ordering to the logical_plan_set
            logical_plan = LogicalPlan(
                operators=logical_plan_operators,
                datasetIdentifier=logical_plan.datasetIdentifier,
            )
            logical_plans.append(logical_plan)

        return logical_plans

    def _construct_logical_plan(self, dataset_nodes: List[pz.Set]) -> LogicalPlan:
        operators = []
        datasetIdentifier = None
        for idx, node in enumerate(dataset_nodes):
            uid = node.universalIdentifier()

            # Use cache if allowed
            if not self.no_cache and pz.datamanager.DataDirectory().hasCachedAnswer(uid):
                op = pz_ops.CacheScan(node.schema, cachedDataIdentifier=uid)
                operators.append(op)
                #return LogicalPlan(operators=operators)
                continue

            # first node is DataSource
            if idx == 0:
                assert isinstance(node, pz.datasources.DataSource)
                datasetIdentifier = uid
                op = pz_ops.BaseScan(datasetIdentifier=uid, outputSchema=node.schema)

            # if the Set's source is another Set, apply the appropriate scan to the Set
            else:
                inputSchema = dataset_nodes[idx - 1].schema
                outputSchema = node.schema
                if node._filter is not None:
                    op = pz_ops.FilteredScan(
                        inputSchema=inputSchema,
                        outputSchema=outputSchema,
                        filter=node._filter,
                        depends_on=node._depends_on,
                        targetCacheId=uid,
                    )
                elif node._groupBy is not None:
                    op = pz_ops.GroupByAggregate(
                        inputSchema=inputSchema,
                        outputSchema=outputSchema,
                        gbySig=node._groupBy,
                        targetCacheId=uid,
                    )
                elif node._aggFunc is not None:
                    op = pz_ops.ApplyAggregateFunction(
                        inputSchema=inputSchema,
                        outputSchema=outputSchema,
                        aggregationFunction=node._aggFunc,
                        targetCacheId=uid,
                    )
                elif node._limit is not None:
                    op = pz_ops.LimitScan(
                        inputSchema=inputSchema,
                        outputSchema=outputSchema,
                        limit=node._limit,
                        targetCacheId=uid,
                    )
                elif not outputSchema == inputSchema:
                    op = pz_ops.ConvertScan(
                        inputSchema=inputSchema,
                        outputSchema=outputSchema,
                        cardinality=node._cardinality,
                        image_conversion=node._image_conversion,
                        depends_on=node._depends_on,
                        targetCacheId=uid,
                    )
                else:
                    raise NotImplementedError("No logical operator exists for the specified dataset construction.")

            operators.append(op)

        return LogicalPlan(operators=operators, datasetIdentifier=datasetIdentifier)

    def generate_plans(self, dataset: pz.Dataset) -> List[LogicalPlan]:
        """Return a set of possible logical trees of operators on Sets."""
        # Obtain ordered list of datasets
        dataset_nodes = []
        node = dataset

        while isinstance(node, pz.sets.Dataset):
            dataset_nodes.append(node)
            node = node._source
        dataset_nodes.append(node)
        dataset_nodes = list(reversed(dataset_nodes))

        # remove unnecessary convert if output schema from data source scan matches
        # input schema for the next operator
        if dataset_nodes[0].schema == dataset_nodes[1].schema:
            dataset_nodes = [dataset_nodes[0]] + dataset_nodes[2:]
            dataset_nodes[1]._source = dataset_nodes[0]

        # construct naive logical plan
        plan = self._construct_logical_plan(dataset_nodes)

        # at the moment, we only consider the naive logical plan for sentinel plans
        if self.sentinel:
            self.plans = [plan]
            return self.plans

        # compute all possible logical re-orderings of this plan
        self.plans = LogicalPlanner._compute_logical_plan_reorderings(plan)

        if self.verbose:
            print(f"LOGICAL PLANS: {len(self.plans)}")

        return self.plans
