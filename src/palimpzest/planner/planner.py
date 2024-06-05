import palimpzest as pz
import palimpzest.operators as ops

# import palimpzest.operators.physical as physical
# import palimpzest.operators.logical as logical
# import palimpzest.operators.induce as induce
# import palimpzest.operators.filter as filter
# import palimpzest.operators.hardcoded_converts as hardcoded_converts

from .plan import LogicalPlan, PhysicalPlan
import pdb
import os
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
        
        # NOTE: ^yes, if this is where the logical operators are created, then we need num_samples and scan_start_idx
        #       when creating the BaseScan or CacheScan

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

        # NOTE: my main questions right now are the following:
        #       1. where do we envision performing logical optimization to get multiple logical plans? Should this be part of LogicalPlanner?
        #       2. where do we envision creating logical plans for sentinels?


class PhysicalPlanner(Planner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.physical_ops = ops.PHYSICAL_OPERATORS

    def _getModels(self, include_vision: bool = False):
        models = []
        if os.getenv("OPENAI_API_KEY") is not None:
            models.extend([pz.Model.GPT_3_5, pz.Model.GPT_4])

        if os.getenv("TOGETHER_API_KEY") is not None:
            models.extend([pz.Model.MIXTRAL])

        if os.getenv("GOOGLE_API_KEY") is not None:
            models.extend([pz.Model.GEMINI_1])

        if include_vision:
            models.append(pz.Model.GPT_4V)

        return models

    def _createSentinelPlan(self, model: pz.Model):
        """
        Create the sentinel plans, which -- at least for now --- are single model plans
        which follow the structure of the user-specified program.
        """
        # base case: this is a root op
        if self.inputOp is None:
            return self._getPhysicalTree(
                strategy=PhysicalOp.LOCAL_PLAN, shouldProfile=True
            )

        # recursive case: get list of possible input physical plans
        subTreePhysicalPlan = self.inputOp._createSentinelPlan(model)
        subTreePhysicalPlan = subTreePhysicalPlan.copy()

        physicalPlan = None
        if isinstance(self, ConvertScan):
            physicalPlan = self._getPhysicalTree(
                strategy=PhysicalOp.LOCAL_PLAN,
                source=subTreePhysicalPlan,
                model=model,
                query_strategy=QueryStrategy.BONDED_WITH_FALLBACK,
                token_budget=1.0,
                shouldProfile=True,
            )

        elif isinstance(self, FilteredScan):
            physicalPlan = self._getPhysicalTree(
                strategy=PhysicalOp.LOCAL_PLAN,
                source=subTreePhysicalPlan,
                model=model,
                shouldProfile=True,
            )

        else:
            physicalPlan = self._getPhysicalTree(
                strategy=PhysicalOp.LOCAL_PLAN,
                source=subTreePhysicalPlan,
                shouldProfile=True,
            )

        return physicalPlan

    def createPhysicalPlanCandidates(
        self,
        max: int = None,
        min: int = None,
        sentinels: bool = False,
        cost_estimate_sample_data: List[Dict[str, Any]] = None,
        allow_model_selection: bool = False,
        allow_codegen: bool = False,
        allow_token_reduction: bool = False,
        pareto_optimal: bool = True,
        include_baselines: bool = False,
        shouldProfile: bool = False,
    ) -> List[PhysicalPlan]:
        """Return a set of physical trees of operators."""
        # only fetch sentinel plans if specified
        if sentinels:
            models = self._getModels()
            assert (
                len(models) > 0
            ), "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"
            sentinel_plans = [self._createSentinelPlan(model) for model in models]
            return sentinel_plans

        # create set of logical plans (e.g. consider different filter/join orderings)
        logicalPlans = LogicalOperator._createLogicalPlans(self)
        print(f"LOGICAL PLANS: {len(logicalPlans)}")

        # iterate through logical plans and evaluate multiple physical plans
        physicalPlans = [
            physicalPlan
            for logicalPlan in logicalPlans
            for physicalPlan in logicalPlan._createPhysicalPlans(
                allow_model_selection=allow_model_selection,
                allow_codegen=allow_codegen,
                allow_token_reduction=allow_token_reduction,
                shouldProfile=shouldProfile,
            )
        ]
        print(f"INITIAL PLANS: {len(physicalPlans)}")

        # compute estimates for every operator
        op_filters_to_estimates = {}
        if cost_estimate_sample_data is not None and cost_estimate_sample_data != []:
            # construct full dataset of samples
            df = pd.DataFrame(cost_estimate_sample_data)

            # get unique set of operator filters:
            # - for base/cache scans this is very simple
            # - for filters, this is based on the unique filter string or function (per-model)
            # - for induce, this is based on the generated field(s) (per-model)
            op_filters_to_estimates = {}
            logical_op = logicalPlans[0]
            while logical_op is not None:
                op_filter, estimates = None, None
                if isinstance(logical_op, BaseScan):
                    op_filter = "op_name == 'base_scan'"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(
                                op_df
                            )
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, CacheScan):
                    op_filter = "op_name == 'cache_scan'"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(
                                op_df
                            )
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, ConvertScan):
                    generated_fields_str = "-".join(sorted(logical_op.generated_fields))
                    op_filter = f"(generated_fields == '{generated_fields_str}') & (op_name == 'induce' | op_name == 'p_induce')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        # compute estimates per-model, and add None which forces computation of avg. across all models
                        models = self._getModels(include_vision=True) + [None]
                        estimates = {model: None for model in models}
                        for model in models:
                            model_name = model.value if model is not None else None
                            est_tokens = StatsProcessor._est_num_input_output_tokens(
                                op_df, model_name=model_name
                            )
                            model_estimates = {
                                "time_per_record": StatsProcessor._est_time_per_record(
                                    op_df, model_name=model_name
                                ),
                                "cost_per_record": StatsProcessor._est_usd_per_record(
                                    op_df, model_name=model_name
                                ),
                                "est_num_input_tokens": est_tokens[0],
                                "est_num_output_tokens": est_tokens[1],
                                "selectivity": StatsProcessor._est_selectivity(
                                    df, op_df, model_name=model_name
                                ),
                                "quality": StatsProcessor._est_quality(
                                    op_df, model_name=model_name
                                ),
                            }
                            estimates[model_name] = model_estimates
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, FilteredScan):
                    filter_str = (
                        logical_op.filter.filterCondition
                        if logical_op.filter.filterCondition is not None
                        else str(logical_op.filter.filterFn)
                    )
                    op_filter = f"(filter == '{str(filter_str)}') & (op_name == 'filter' | op_name == 'p_filter')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        models = (
                            self._getModels()
                            if logical_op.filter.filterCondition is not None
                            else [None]
                        )
                        estimates = {model: None for model in models}
                        for model in models:
                            model_name = model.value if model is not None else None
                            est_tokens = StatsProcessor._est_num_input_output_tokens(
                                op_df, model_name=model_name
                            )
                            model_estimates = {
                                "time_per_record": StatsProcessor._est_time_per_record(
                                    op_df, model_name=model_name
                                ),
                                "cost_per_record": StatsProcessor._est_usd_per_record(
                                    op_df, model_name=model_name
                                ),
                                "est_num_input_tokens": est_tokens[0],
                                "est_num_output_tokens": est_tokens[1],
                                "selectivity": StatsProcessor._est_selectivity(
                                    df, op_df, model_name=model_name
                                ),
                                "quality": StatsProcessor._est_quality(
                                    op_df, model_name=model_name
                                ),
                            }
                            estimates[model_name] = model_estimates
                    op_filters_to_estimates[op_filter] = estimates

                elif isinstance(logical_op, LimitScan):
                    op_filter = "(op_name == 'limit')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(
                                op_df
                            )
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif (
                    isinstance(logical_op, ApplyAggregateFunction)
                    and logical_op.aggregationFunction.funcDesc == "COUNT"
                ):
                    op_filter = "(op_name == 'count')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(
                                op_df
                            )
                        }
                    op_filters_to_estimates[op_filter] = estimates

                elif (
                    isinstance(logical_op, ApplyAggregateFunction)
                    and logical_op.aggregationFunction.funcDesc == "AVERAGE"
                ):
                    op_filter = "(op_name == 'average')"
                    op_df = df.query(op_filter)
                    if not op_df.empty:
                        estimates = {
                            "time_per_record": StatsProcessor._est_time_per_record(
                                op_df
                            )
                        }
                    op_filters_to_estimates[op_filter] = estimates

                logical_op = logical_op.inputOp

        # estimate the cost (in terms of USD, latency, throughput, etc.) for each plan
        plans = []
        cost_est_data = (
            None if op_filters_to_estimates == {} else op_filters_to_estimates
        )
        for physicalPlan in physicalPlans:
            planCost, fullPlanCostEst = physicalPlan.estimateCost(
                cost_est_data=cost_est_data
            )

            totalTime = planCost["totalTime"]
            totalCost = planCost["totalUSD"]  # for now, cost == USD
            quality = planCost["quality"]

            plans.append((totalTime, totalCost, quality, physicalPlan, fullPlanCostEst))

        # drop duplicate plans in terms of time, cost, and quality, as these can cause
        # plans on the pareto frontier to be dropped if they are "dominated" by a duplicate
        dedup_plans, dedup_desc_set = [], set()
        for plan in plans:
            planDesc = (plan[0], plan[1], plan[2])
            if planDesc not in dedup_desc_set:
                dedup_desc_set.add(planDesc)
                dedup_plans.append(plan)

        print(f"DEDUP PLANS: {len(dedup_plans)}")

        # return de-duplicated set of plans if we don't want to compute the pareto frontier
        if not pareto_optimal:
            if max is not None:
                dedup_plans = dedup_plans[:max]
                print(f"LIMIT DEDUP PLANS: {len(dedup_plans)}")

            return dedup_plans

        # compute the pareto frontier of candidate physical plans and return the list of such plans
        # - brute force: O(d*n^2);
        #   - for every tuple, check if it is dominated by any other tuple;
        #   - if it is, throw it out; otherwise, add it to pareto frontier
        #
        # more efficient algo.'s exist, but they are non-trivial to implement, so for now I'm using
        # brute force; it may ultimately be best to compute a cheap approx. of the pareto front:
        # - e.g.: https://link.springer.com/chapter/10.1007/978-3-642-12002-2_6
        paretoFrontierPlans, baselinePlans = [], []
        for i, (
            totalTime_i,
            totalCost_i,
            quality_i,
            plan,
            fullPlanCostEst,
        ) in enumerate(dedup_plans):
            paretoFrontier = True

            # ensure that all baseline plans are included if specified
            if include_baselines:
                for baselinePlan in [
                    self._createBaselinePlan(model) for model in self._getModels()
                ]:
                    if baselinePlan == plan:
                        baselinePlans.append(
                            (totalTime_i, totalCost_i, quality_i, plan, fullPlanCostEst)
                        )
                        continue

            # check if any other plan dominates plan i
            for j, (totalTime_j, totalCost_j, quality_j, _, _) in enumerate(
                dedup_plans
            ):
                if i == j:
                    continue

                # if plan i is dominated by plan j, set paretoFrontier = False and break
                if (
                    totalTime_j <= totalTime_i
                    and totalCost_j <= totalCost_i
                    and quality_j >= quality_i
                ):
                    paretoFrontier = False
                    break

            # add plan i to pareto frontier if it's not dominated
            if paretoFrontier:
                paretoFrontierPlans.append(
                    (totalTime_i, totalCost_i, quality_i, plan, fullPlanCostEst)
                )

        print(f"PARETO PLANS: {len(paretoFrontierPlans)}")
        print(f"BASELINE PLANS: {len(baselinePlans)}")

        # if specified, grab up to `min` total plans, and choose the remaining plans
        # based on their smallest agg. distance to the pareto frontier; distance is computed
        # by summing the pct. difference to the pareto frontier across each dimension
        def is_in_final_plans(plan, finalPlans):
            # determine if this plan is already in the final set of plans
            for _, _, _, finalPlan, _ in finalPlans:
                if plan == finalPlan:
                    return True
            return False

        finalPlans = paretoFrontierPlans
        for planInfo in baselinePlans:
            if is_in_final_plans(planInfo[3], finalPlans):
                continue
            else:
                finalPlans.append(planInfo)

        if min is not None and len(finalPlans) < min:
            min_distances = []
            for i, (totalTime, totalCost, quality, plan, fullPlanCostEst) in enumerate(
                dedup_plans
            ):
                # determine if this plan is already in the final set of plans
                if is_in_final_plans(plan, finalPlans):
                    continue

                # otherwise compute min distance to plans on pareto frontier
                min_dist, min_dist_idx = np.inf, -1
                for paretoTime, paretoCost, paretoQuality, _, _ in paretoFrontierPlans:
                    time_dist = (totalTime - paretoTime) / paretoTime
                    cost_dist = (totalCost - paretoCost) / paretoCost
                    quality_dist = (
                        (paretoQuality - quality) / quality if quality > 0 else 10.0
                    )
                    dist = time_dist + cost_dist + quality_dist
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_idx = i

                min_distances.append((min_dist, min_dist_idx))

            # sort based on distance
            min_distances = sorted(min_distances, key=lambda tup: tup[0])

            # add closest plans to finalPlans
            k = min - len(finalPlans)
            k_indices = list(map(lambda tup: tup[1], min_distances[:k]))
            for idx in k_indices:
                finalPlans.append(dedup_plans[idx])

        return finalPlans

    def generate_plans(self, logical_plan: LogicalPlan, sentinels: bool=False) -> List[PhysicalPlan]:
        """Return a set of possible physical plans."""
        # only fetch sentinel plans if specified
        if sentinels:
            models = self._getModels()
            assert (
                len(models) > 0
            ), "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"
            sentinel_plans = [self._createSentinelPlan(logical_plan, model) for model in models]
            return sentinel_plans

        # NOTE: more questions:
        #       1. Should PhysicalPlanner accept all logical_plans created by the LogicalPlanner?
        #          Or does it only compute physical plans for a single LogicalPlan?
        #       2. I assume we will pass in sample cost est. data here (if not running sentinels?)

    def generate_plans(self, logical_plan: LogicalPlan) -> List[PhysicalPlan]:
        """Return a set of possible physical plans."""

        # Stub of physical planning code
        for logical_op in logical_plan:
            applicable_ops = [
                phy
                for phy in physical_ops
                if phy.inputSchema == logical_op.inputSchema
                and phy.outputSchema == logical_op.outputSchema
            ]  # Here this should be double checked


# # Stub of execution code
# for plan in physical_plan:
#     for phy_op in plan:
#             instantiated_op = phy_op()
#             for record in dataset:
#                 with Profiler: # or however the Stat collection works:
#                     result = instantiated_op(record)

