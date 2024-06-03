import palimpzest as pz
from palimpzest import operators as ops
from .plan import LogicalPlan, PhysicalPlan
import pdb; 

class Planner():
    def __init__(self):
        raise NotImplementedError

    def plan_logical(self):
        return NotImplementedError
    
    def plan_physical(self):
        return NotImplementedError
    
class SimplePlanner():
    def __init__(self, 
                 no_cache=False):
        self.no_cache = no_cache
        # TODO planner should know num_samples, scan_start_idx,

    def plan_logical(self, dataset):
        """Return the logical tree of operators on Sets."""
        # first, check to see if this set has previously been cached
        
        # Obtain ordered list of datasets
        datasets = []        
        node = dataset
        while isinstance(node, pz.datasources.DataSource):
            datasets.append(node)
            node = dataset._source
        datasets.apppend(node)
        datasets = reversed(datasets)

        operators = []

        for idx,node in enumerate(datasets):
            uid = node.universalIdentifier()

            # Use cache if allowed
            if not self.no_cache:
                if pz.datamanager.DataDirectory().hasCachedAnswer(uid):
                    op = ops.CacheScan(node._schema, uid, node._num_samples, node._scan_start_idx)
                    

            # First node is DataSource
            if idx == 0: 
                assert isinstance(node, pz.datasources.DataSource):
                dataset_id = node._source.universalIdentifier()
                sourceSchema = node._source.schema

                if node._schema == sourceSchema:
                    return ops.BaseScan(node._schema, dataset_id, node._num_samples, node._scan_start_idx)
                else:
                    return ops.ConvertScan(node._schema, ops.BaseScan(sourceSchema, dataset_id, node._num_samples, node._scan_start_idx), targetCacheId=uid)

            # if the Set's source is another Set, apply the appropriate scan to the Set
            elif self._filter is not None:
                op = ops.FilteredScan(self._schema, self._source.getLogicalTree(), self._filter, self._depends_on, targetCacheId=uid)
            elif self._groupBy is not None:
                op = ops.GroupByAggregate(self._schema, self._source.getLogicalTree(), self._groupBy, targetCacheId=uid)
            elif self._aggFunc is not None:
                op = ops.ApplyAggregateFunction(self._schema, self._source.getLogicalTree(), self._aggFunc, targetCacheId=uid)
            elif self._limit is not None:
                op = ops.LimitScan(self._schema, self._source.getLogicalTree(), self._limit, targetCacheId=uid)
            elif self._fnid is not None:
                op = ops.ApplyUserFunction(self._schema, self._source.getLogicalTree(), self._fnid, targetCacheId=uid)
            elif not self._schema == self._source._schema:
                op = ops.ConvertScan(self._schema, self._source.getLogicalTree(), self._cardinality, self._image_conversion, self._depends_on, targetCacheId=uid)
            else:
                op = self._source.getLogicalTree()
            
            operators.append(op)

        plan = LogicalPlan(datasets = datasets, operators = operators)
        return plan

    def plan_physical(self):
        return NotImplementedError
    