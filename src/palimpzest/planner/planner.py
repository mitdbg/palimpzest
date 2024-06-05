import palimpzest as pz
from palimpzest import operators as ops
from .plan import LogicalPlan, PhysicalPlan
import pdb
from typing import List, Tuple


class Planner:
    """
    A Planner is responsible for generating a set of possible plans.
    The fundamental abstraction is, given an input of a graph (of datasets, or logical operators), it generates a set of possible graphs which correspond to the plan.
    These plans can be consumed from the planner using the __iter__ method.
    """

    def __init__(self):
        self.plans = []

    def generate_plans(self):
        return NotImplementedError

    def __iter__(self):
        return iter(self.plans)

    def __next__(self):
        return next(iter(self.plans))

    def __len__(self):
        return len(self.plans)


class LogicalPlanner(Planner):
    def __init__(self, no_cache=False, *args, **kwargs):
        """A given planner should not have a dataset when it's being generated, since it could be used for multiple datasets.
        However, we currently cannot support this since the plans are stored within a single planner object.
        To support this, we can use a dictionary in the form [dataset -> [Plan, Plan, ...]].
        To discuss for future versions.
        """

        super().__init__(*args, **kwargs)
        self.no_cache = no_cache
        # TODO planner should know num_samples, scan_start_idx ?

    def generate_plans(self, dataset) -> None:
        """Return a set of possible logical trees of operators on Sets."""
        # first, check to see if this set has previously been cached

        # Obtain ordered list of datasets
        dataset_nodes = []
        node = dataset

        while isinstance(node, pz.sets.Dataset):
            dataset_nodes.append(node)
            node = node._source
        dataset_nodes.append(node)
        dataset_nodes = list(reversed(dataset_nodes))

        print(dataset_nodes)
        operators = []
        for idx, node in enumerate(dataset_nodes):
            uid = node.universalIdentifier()

            # Use cache if allowed
            if not self.no_cache:
                if pz.datamanager.DataDirectory().hasCachedAnswer(uid):
                    op = ops.CacheScan(
                        node._schema, uid, node._num_samples, node._scan_start_idx
                    )

            # First node is DataSource
            if idx == 0:
                assert isinstance(node, pz.datasources.DataSource)
                dataset_id = node.universalIdentifier()
                sourceSchema = node.schema
                node._schema = node.schema

                if dataset_nodes[idx + 1]._schema == sourceSchema:
                    op = ops.BaseScan(
                        outputSchema=sourceSchema,
                        datasetIdentifier=dataset_id,
                    )
                else:
                    op = ops.ConvertScan(
                        outputSchema=dataset_nodes[idx + 1]._schema,
                        inputSchema=sourceSchema,
                        targetCacheId=uid,
                    )

            # if the Set's source is another Set, apply the appropriate scan to the Set
            else:
                inputSchema = dataset_nodes[idx - 1]._schema
                outputSchema = node._schema

                if node._filter is not None:
                    op = ops.FilteredScan(
                        outputSchema=outputSchema,
                        inputSchema=inputSchema,
                        filter=node._filter,
                        depends_on=node._depends_on,
                        targetCacheId=uid,
                    )
                elif node._groupBy is not None:
                    op = ops.GroupByAggregate(
                        outputSchema=outputSchema,
                        inputSchema=inputSchema,
                        gbySig=node._groupBy,
                        targetCacheId=uid,
                    )
                elif node._aggFunc is not None:
                    op = ops.ApplyAggregateFunction(
                        outputSchema=outputSchema,
                        inputSchema=inputSchema,
                        aggregationFunction=node._aggFunc,
                        targetCacheId=uid,
                    )
                elif node._limit is not None:
                    op = ops.LimitScan(
                        outputSchema=outputSchema,
                        inputSchema=inputSchema,
                        limit=node._limit,
                        targetCacheId=uid,
                    )
                elif node._fnid is not None:
                    op = ops.ApplyUserFunction(
                        outputSchema=outputSchema,
                        inputSchema=inputSchema,
                        fnid=node._fnid,
                        targetCacheId=uid,
                    )
                elif not outputSchema == inputSchema:
                    op = ops.ConvertScan(
                        outputSchema=outputSchema,
                        inputSchema=inputSchema,
                        cardinality=node._cardinality,
                        image_conversion=node._image_conversion,
                        depends_on=node._depends_on,
                        targetCacheId=uid,
                    )
                else:
                    assert NotImplementedError("TODO what happens in this case?")
                    # op = self._source.getLogicalTree()

            operators.append(op)

        plan = LogicalPlan(datasets=dataset_nodes, operators=operators)
        self.plans.append(plan)


class PhysicalPlanner(Planner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_plans(self, logical_plan: LogicalPlan) -> List[PhysicalPlan]:
        """Return a set of possible physical plans."""


# # Stub of physical planning code
# for plan in logical_plans:
#     for logical_op in plan:
#         applicable_ops = [phy for phy in physical_ops
#                           if phy.inputSchema == logical_op.inputSchema
#                           and phy.outputSchema == logical_op.outputSchema]


# # Stub of execution code
# for plan in physical_plan:
#     for phy_op in plan:
#             instantiated_op = phy_op()
#             for record in dataset:
#                 with Profiler: # or however the Stat collection works:
#                     result = instantiated_op(record)
