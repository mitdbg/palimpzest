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
                assert isinstance(node._source, pz.datasources.DataSource)
                dataset_id = node._source.universalIdentifier()
                sourceSchema = node._source.schema

                if node._schema == sourceSchema:
                    return ops.BaseScan(
                        node._schema,
                        dataset_id,
                    )
                else:
                    return ops.ConvertScan(
                        node._schema,
                        ops.BaseScan(
                            sourceSchema,
                            dataset_id,
                        ),
                        targetCacheId=uid,
                    )

            # if the Set's source is another Set, apply the appropriate scan to the Set
            elif self._filter is not None:
                op = ops.FilteredScan(
                    self._schema,
                    self._source.getLogicalTree(),
                    self._filter,
                    self._depends_on,
                    targetCacheId=uid,
                )
            elif self._groupBy is not None:
                op = ops.GroupByAggregate(
                    self._schema,
                    self._source.getLogicalTree(),
                    self._groupBy,
                    targetCacheId=uid,
                )
            elif self._aggFunc is not None:
                op = ops.ApplyAggregateFunction(
                    self._schema,
                    self._source.getLogicalTree(),
                    self._aggFunc,
                    targetCacheId=uid,
                )
            elif self._limit is not None:
                op = ops.LimitScan(
                    self._schema,
                    self._source.getLogicalTree(),
                    self._limit,
                    targetCacheId=uid,
                )
            elif self._fnid is not None:
                op = ops.ApplyUserFunction(
                    self._schema,
                    self._source.getLogicalTree(),
                    self._fnid,
                    targetCacheId=uid,
                )
            elif not self._schema == self._source._schema:
                op = ops.ConvertScan(
                    self._schema,
                    self._source.getLogicalTree(),
                    self._cardinality,
                    self._image_conversion,
                    self._depends_on,
                    targetCacheId=uid,
                )
            else:
                op = self._source.getLogicalTree()

            operators.append(op)

        plan = LogicalPlan(datasets=dataset_nodes, operators=operators)
        self.plans.append(plan)


class PhysicalPlanner(Planner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_plans(self, logical_plan: LogicalPlan) -> List[PhysicalPlan]:
        """Return a set of possible physical plans."""
