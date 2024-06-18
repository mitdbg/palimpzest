from palimpzest.constants import Model, PromptStrategy, QueryStrategy
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.operators import PhysicalOperator
from palimpzest.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.planner import LogicalPlan, PhysicalPlan
from palimpzest.planner.planner import Planner
from .plan import LogicalPlan, PhysicalPlan

import palimpzest as pz
import palimpzest.operators as pz_ops
import palimpzest.corelib.schemas as schemas

from typing import List, Optional

import numpy as np

import multiprocessing
import palimpzest.strategies as physical_strategies



class PhysicalPlanner(Planner):
    def __init__(
        self,
            # I feel that num_samples is an attribute who does not belong in the planner, but in the exeuction module: because all we use it for  here is to pass it to the physical operators (which again should have no business knowing the num samples they process)
            num_samples: Optional[int]=10,
            scan_start_idx: Optional[int]=0,
            available_models: Optional[List[Model]]=[],
            allow_model_selection: Optional[bool]=True,
            allow_code_synth: Optional[bool]=True,
            allow_token_reduction: Optional[bool]=True,
            shouldProfile: Optional[bool]=True,
            useParallelOps: Optional[bool]=False,
            useStrategies: Optional[bool]=False, # only for debug purposes 
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx
        self.available_models = available_models
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.shouldProfile = shouldProfile
        self.useParallelOps = useParallelOps
        self.useStrategies = useStrategies
        # Ideally this gets customized by model selection, code synth, etc. etc. using strategies
        self.physical_ops = pz_ops.PHYSICAL_OPERATORS

        # This is a dictionary where the physical planner will keep track for all the logical plan operators defined,
        # which physical operators are available to implement them.
        self.logical_physical_map = {}
        for logical_op in pz_ops.LOGICAL_OPERATORS:
            self.logical_physical_map[logical_op] = []
            for physical_op in self.physical_ops:
                if physical_op.implements(logical_op):
                    self.logical_physical_map[logical_op].append(physical_op)

            if self.useStrategies:
                for strategy in physical_strategies.REGISTERED_STRATEGIES:
                    if strategy.logical_op_class == logical_op:
                        ops = strategy(self.available_models)
                        self.logical_physical_map.get(logical_op, []).extend(ops)

    def _resolveLogicalConvertOp(
        self,
        logical_convert_op: pz_ops.ConvertScan,
        model: Optional[Model] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        query_strategy: Optional[QueryStrategy] = None,
        token_budget: Optional[float] = 1.0,
        shouldProfile: bool = False,
        sentinel: bool = False,
    ) -> PhysicalOperator:
        """
        Given the logical operator for a convert, determine which (set of) physical operation(s)
        the PhysicalPlanner can use to implement that logical operation.
        """
        if sentinel:
            shouldProfile = True

        # get input and output schemas for convert
        inputSchema = logical_convert_op.inputSchema
        outputSchema = logical_convert_op.outputSchema

        # TODO: test schema equality
        # use simple convert if the input and output schemas are the same
        if inputSchema == outputSchema:
            op = pz_ops.SimpleTypeConvert(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                targetCacheId=logical_convert_op.targetCacheId,
                shouldProfile=shouldProfile,
            )

        # TODO: replace all these elif's with iteration over self.physical_ops
        #       - Q: what happens if we ever have two hard-coded conversions w/same input and output schema?
        #            - e.g. imagine if we had ConvertDownloadToFileTypeA and ConvertDownloadToFileTypeB
        # if input and output schema are covered by a hard-coded convert; use that
        elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.TextFile):
            op = pz_ops.ConvertFileToText(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                targetCacheId=logical_convert_op.targetCacheId,
                shouldProfile=shouldProfile,
            )

        elif isinstance(inputSchema, schemas.ImageFile) and isinstance(outputSchema, schemas.EquationImage):
            op = pz_ops.ConvertImageToEquation(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                targetCacheId=logical_convert_op.targetCacheId,
                shouldProfile=shouldProfile,
            )

        elif isinstance(inputSchema, schemas.Download) and isinstance(outputSchema, schemas.File):
            op = pz_ops.ConvertDownloadToFile(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                targetCacheId=logical_convert_op.targetCacheId,
                shouldProfile=shouldProfile,
            )

        elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.XLSFile):
            op = pz_ops.ConvertFileToXLS(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                targetCacheId=logical_convert_op.targetCacheId,
                shouldProfile=shouldProfile,
            )

        elif isinstance(inputSchema, schemas.XLSFile) and isinstance(outputSchema, schemas.Table):
            op = pz_ops.ConvertXLSToTable(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                cardinality=logical_convert_op.cardinality,
                targetCacheId=logical_convert_op.targetCacheId,
                shouldProfile=shouldProfile,
            )

        elif isinstance(inputSchema, schemas.File) and isinstance(outputSchema, schemas.PDFFile):
            op = pz_ops.ConvertFileToPDF(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                pdfprocessor=pz.DataDirectory().current_config.get("pdfprocessing"),
                targetCacheId=logical_convert_op.targetCacheId,
                shouldProfile=shouldProfile,
            )

        # otherwise, create convert op for the given set of hyper-parameters
        else:
            assert prompt_strategy is not None, "Prompt strategy must be specified for LLMConvert"
            op = pz_ops.LLMConvert(
                inputSchema=inputSchema,
                outputSchema=outputSchema,
                model=model,
                prompt_strategy=prompt_strategy,
                query_strategy=query_strategy,
                token_budget=token_budget,
                cardinality=logical_convert_op.cardinality,
                image_conversion=logical_convert_op.image_conversion,
                desc=logical_convert_op.desc,
                targetCacheId=logical_convert_op.targetCacheId,
            )

        return op

    def _resolveLogicalFilterOp(
        self,
        logical_filter_op: pz_ops.FilteredScan,
        model: Optional[Model] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        shouldProfile: bool = False,
        sentinel: bool = False,
    ) -> PhysicalOperator:
        """
        Given the logical operator for a filter, determine which (set of) physical operation(s)
        the PhysicalPlanner can use to implement that logical operation.
        """
        if sentinel:
            shouldProfile = True

        use_llm_filter = logical_filter_op.filter.filterFn is None
        op = (
            pz_ops.LLMFilter(
                inputSchema=logical_filter_op.inputSchema,
                outputSchema=logical_filter_op.outputSchema,
                filter=logical_filter_op.filter,
                model=model,
                prompt_strategy=prompt_strategy,
                targetCacheId=logical_filter_op.targetCacheId,
                shouldProfile=shouldProfile,
                max_workers=multiprocessing.cpu_count() if self.useParallelOps else 1,
            )
            if use_llm_filter
            else pz_ops.NonLLMFilter(
                inputSchema=logical_filter_op.inputSchema,
                outputSchema=logical_filter_op.outputSchema,
                filter=logical_filter_op.filter,
                targetCacheId=logical_filter_op.targetCacheId,
                shouldProfile=shouldProfile,
                max_workers=multiprocessing.cpu_count() if self.useParallelOps else 1,
            )
        )

        return op

    def _resolveLogicalApplyAggFuncOp(
        self,
        logical_apply_agg_fn_op: pz_ops.ApplyAggregateFunction,
        shouldProfile: bool = False,
        sentinel: bool = False,
    ) -> PhysicalOperator:
        """
        Given the logical operator for a group by, determine which (set of) physical operation(s)
        the PhysicalPlanner can use to implement that logical operation.
        """
        if sentinel:
            shouldProfile = True

        # TODO: use an Enum to list possible funcDesc(s)
        physicalOp = None
        if logical_apply_agg_fn_op.aggregationFunction.funcDesc == "COUNT":
            physicalOp = pz_ops.ApplyCountAggregateOp
        elif logical_apply_agg_fn_op.aggregationFunction.funcDesc == "AVERAGE":
            physicalOp = pz_ops.ApplyAverageAggregateOp

        op = physicalOp(
            inputSchema=logical_apply_agg_fn_op.inputSchema,
            aggFunction=logical_apply_agg_fn_op.aggregationFunction,
            targetCacheId=logical_apply_agg_fn_op.targetCacheId,
            shouldProfile=shouldProfile,
        )

        return op

    def _createBaselinePlan(self, logical_plan: LogicalPlan, model: Model) -> PhysicalPlan:
        """A simple wrapper around _createSentinelPlan as right now these are one and the same."""
        return self._createSentinelPlan(logical_plan, model)

    def _createSentinelPlan(self, logical_plan: LogicalPlan, model: Model) -> PhysicalPlan:
        """
        Create the sentinel plan for the given model. At least for now --- each
        sentinel plan is a plan with a single model which follows the naive logical
        plan implied by the user's program.
        """
        datasetIdentifier = logical_plan.datasetIdentifier
        physical_operators = []
        for logical_op in logical_plan.operators:
            op = None
            shouldProfile = True

            if isinstance(logical_op, pz_ops.BaseScan):
                op_class = self.logical_physical_map[type(logical_op)][0] # only one physical operator
                dataset_type = DataDirectory().getRegisteredDatasetType(datasetIdentifier)

                op = op_class(outputSchema=logical_op.outputSchema,
                              dataset_type=dataset_type,
                              num_samples=self.num_samples,
                              scan_start_idx=self.scan_start_idx,
                              shouldProfile=shouldProfile,)

            elif isinstance(logical_op, pz_ops.CacheScan):
                op_class = self.logical_physical_map[type(logical_op)][0]
                op = op_class(outputSchema=logical_op.outputSchema,
                                cachedDataIdentifier=logical_op.cachedDataIdentifier,
                                num_samples=self.num_samples,
                                scan_start_idx=self.scan_start_idx,
                                shouldProfile=shouldProfile,
                )

                              
            elif isinstance(logical_op, pz_ops.ConvertScan):
                op = self._resolveLogicalConvertOp(
                    logical_op,
                    model=model,
                    prompt_strategy=PromptStrategy.DSPY_COT_QA,
                    query_strategy=QueryStrategy.BONDED_WITH_FALLBACK,
                    token_budget=1.0,
                    sentinel=True,
                )

            elif isinstance(logical_op, pz_ops.FilteredScan):
                if self.useStrategies:
                    op_class: PhysicalOperator = None
                    for op in self.logical_physical_map[type(logical_op)]:
                        if op in [LLMFilter, NonLLMFilter]:
                            continue
                        if op.model == model:
                            op_class = op
                            break
                    op = op_class(
                            inputSchema=logical_op.inputSchema,
                            outputSchema=logical_op.outputSchema,
                            filter=logical_op.filter,
                            shouldProfile=self.shouldProfile,
                        )
                else:
                    op = self._resolveLogicalFilterOp(
                        logical_op,
                        model=model,
                        prompt_strategy=PromptStrategy.DSPY_COT_BOOL,
                        sentinel=True,
                    )

            elif isinstance(logical_op, pz_ops.LimitScan):
                op_class = self.logical_physical_map[type(logical_op)][0]
                op = op_class( # **logical_op.getParameters()
                        inputSchema=logical_op.inputSchema,
                        outputSchema=logical_op.outputSchema,
                        limit=logical_op.limit,
                        targetCacheId=logical_op.targetCacheId,
                        shouldProfile=shouldProfile,
                    )

            elif isinstance(logical_op, pz_ops.GroupByAggregate):
                op_class = self.logical_physical_map[type(logical_op)][0]
                op = op_class(
                    inputSchema=logical_op.inputSchema,
                    gbySig=logical_op.gbySig,
                    targetCacheId=logical_op.targetCacheId,
                    shouldProfile=shouldProfile,
                    )

            elif isinstance(logical_op, pz_ops.ApplyAggregateFunction):
                op = self._resolveLogicalApplyAggFuncOp(logical_op, sentinel=True)
                # op_class = self.logical_physical_map[logical_op][0]
                # op = op_class(
                #     inputSchema=logical_op.inputSchema,
                #     gbySig=logical_op.gbySig,
                #     targetCacheId=logical_op.targetCacheId,
                #     shouldProfile=shouldProfile,
                #     )

            physical_operators.append(op)

        return PhysicalPlan(operators=physical_operators, datasetIdentifier=datasetIdentifier)

    def _createPhysicalPlans(self, logical_plan: LogicalPlan) -> List[PhysicalPlan]:
        """
        Given the logical plan implied by this LogicalOperator, enumerate up to `max`
        possible physical plans and return them as a list.
        """
        # check that at least one llm service has been provided
        assert (
            len(self.available_models) > 0
        ), "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"
        datasetIdentifier = logical_plan.datasetIdentifier
        # determine which query strategies may be used
        query_strategies = [QueryStrategy.BONDED_WITH_FALLBACK]
        if self.allow_code_synth:
            query_strategies.append(QueryStrategy.CODE_GEN_WITH_FALLBACK)

        token_budgets = [1.0]
        if self.allow_token_reduction:
            token_budgets.extend([0.1, 0.5, 0.9])

        # get logical operators
        operators = logical_plan.operators

        all_plans = []
        for logical_op in operators:
            # base case, if this operator is a BaseScan set all_plans to be the physical plan with just this operator
            if isinstance(logical_op, pz_ops.BaseScan):
                op_class = self.logical_physical_map[type(logical_op)][0] # only one physical operator
                dataset_type = DataDirectory().getRegisteredDatasetType(datasetIdentifier)
                physical_op = op_class(outputSchema = logical_op.outputSchema,
                              dataset_type=dataset_type,
                              num_samples=self.num_samples,
                              scan_start_idx=self.scan_start_idx,
                              shouldProfile=self.shouldProfile,)
                all_plans = [PhysicalPlan(operators=[physical_op], datasetIdentifier=datasetIdentifier)]

            # base case (possibly), if this operator is a CacheScan and all_plans is empty, set all_plans to be
            # the physical plan with just this operator; if all_plans is NOT empty, then merge w/all_plans
            elif isinstance(logical_op, pz_ops.CacheScan):
                op_class = self.logical_physical_map[type(logical_op)][0] # only one physical operator
                physical_op = op_class(outputSchema=logical_op.outputSchema,
                                cachedDataIdentifier=logical_op.cachedDataIdentifier,
                                num_samples=self.num_samples,
                                scan_start_idx=self.scan_start_idx,
                                shouldProfile=self.shouldProfile,
                )


                if all_plans == []:
                    all_plans = [PhysicalPlan(operators=[physical_op], datasetIdentifier=datasetIdentifier)]
                else:
                    plans = []
                    for subplan in all_plans:
                        new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                        plans.append(new_physical_plan)

                    # update all_plans
                    all_plans = plans

            elif isinstance(logical_op, pz_ops.ConvertScan):
                plans = []
                for subplan in all_plans:
                    # TODO: if hard-coded conversion, don't iterate over all plan possibilities
                    for qs in query_strategies:
                        # for code generation: we do not need to iterate over models and token budgets
                        if qs in [QueryStrategy.CODE_GEN_WITH_FALLBACK, QueryStrategy.CODE_GEN]:
                            physical_op = self._resolveLogicalConvertOp(
                                logical_op,
                                query_strategy=qs,
                                prompt_strategy=PromptStrategy.DSPY_COT_QA,
                                shouldProfile=self.shouldProfile,
                            )
                            new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                            plans.append(new_physical_plan)
                            continue

                        # for non-code generation query strategies, consider models and token budgets
                        models = self.available_models
                        for model in models:
                            for token_budget in token_budgets:
                                physical_op = self._resolveLogicalConvertOp(
                                    logical_op,
                                    model=model,
                                    query_strategy=qs,
                                    prompt_strategy=PromptStrategy.DSPY_COT_QA,
                                    token_budget=token_budget,
                                    shouldProfile=self.shouldProfile,
                                )
                                new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                                plans.append(new_physical_plan)

                # update all_plans
                all_plans = plans

            elif isinstance(logical_op, pz_ops.FilteredScan):
                plans = []
                for subplan in all_plans:
                    # TODO: if non-llm filter, don't iterate over all plan possibilities
                    if self.useStrategies:
                        for op in self.logical_physical_map[type(logical_op)]:
                            if logical_op in [LLMFilter, NonLLMFilter]:
                                continue
                            physical_op = op(
                                inputSchema=logical_op.inputSchema,
                                outputSchema=logical_op.outputSchema,
                                filter=logical_op.filter,
                                shouldProfile=self.shouldProfile,
                            )
                    else:
                        models = self.available_models
                        for m in models:
                            physical_op = self._resolveLogicalFilterOp(
                                logical_op,
                                model=m,
                                prompt_strategy=PromptStrategy.DSPY_COT_BOOL,
                                shouldProfile=self.shouldProfile,
                            )


                    new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                    plans.append(new_physical_plan)

                # update all_plans
                all_plans = plans

            elif isinstance(logical_op, pz_ops.LimitScan):
                op_class = self.logical_physical_map[type(logical_op)][0] # only one physical operator
                physical_op = op_class(
                        inputSchema=logical_op.inputSchema,
                        outputSchema=logical_op.outputSchema,
                        limit=logical_op.limit,
                        targetCacheId=logical_op.targetCacheId,
                        shouldProfile=self.shouldProfile,
                    )

                plans = []
                for subplan in all_plans:
                    new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                    plans.append(new_physical_plan)

                # update all_plans
                all_plans = plans

            elif isinstance(logical_op, pz_ops.GroupByAggregate):
                op_class = self.logical_physical_map[type(logical_op)][0]
                physical_op = op_class(
                    inputSchema=logical_op.inputSchema,
                    gbySig=logical_op.gbySig,
                    targetCacheId=logical_op.targetCacheId,
                    shouldProfile=self.shouldProfile,
                    )

                plans = []
                for subplan in all_plans:
                    new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                    plans.append(new_physical_plan)

                # update all_plans
                all_plans = plans

            elif isinstance(logical_op, pz_ops.ApplyAggregateFunction):
                physical_op = self._resolveLogicalApplyAggFuncOp(logical_op, shouldProfile=self.shouldProfile)

                plans = []
                for subplan in all_plans:
                    new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                    plans.append(new_physical_plan)

                # update all_plans
                all_plans = plans

        return all_plans

    def deduplicate_plans(self, physical_plans: List[PhysicalPlan]) -> List[PhysicalPlan]:
        """De-duplicate plans with identical estimates for runtime, cost, and quality."""
        # drop duplicate plans in terms of time, cost, and quality, as these can cause
        # plans on the pareto frontier to be dropped if they are "dominated" by a duplicate
        dedup_plans, dedup_tuple_set = [], set()
        for plan in physical_plans:
            plan_tuple = (plan.total_time, plan.total_cost, plan.quality)
            if plan_tuple not in dedup_tuple_set:
                dedup_tuple_set.add(plan_tuple)
                dedup_plans.append(plan)

        print(f"DEDUP PLANS: {len(dedup_plans)}")
        return dedup_plans

    def select_pareto_optimal_plans(self, physical_plans: List[PhysicalPlan]) -> List[PhysicalPlan]:
        """Select the subset of physical plans which lie on the pareto frontier of our runtime, cost, and quality estimates."""
        # compute the pareto frontier of candidate physical plans and return the list of such plans
        # - brute force: O(d*n^2);
        #   - for every tuple, check if it is dominated by any other tuple;
        #   - if it is, throw it out; otherwise, add it to pareto frontier
        #
        # more efficient algo.'s exist, but they are non-trivial to implement, so for now I'm using
        # brute force; it may ultimately be best to compute a cheap approx. of the pareto front:
        # - e.g.: https://link.springer.com/chapter/10.1007/978-3-642-12002-2_6
        pareto_frontier_plans = []
        for i, plan_i in enumerate(physical_plans):
            paretoFrontier = True

            # check if any other plan dominates plan i
            for j, plan_j in enumerate(physical_plans):
                if i == j:
                    continue

                # if plan i is dominated by plan j, set paretoFrontier = False and break
                if (
                    plan_j.total_time <= plan_i.total_time
                    and plan_j.total_cost <= plan_i.total_cost
                    and plan_j.quality >= plan_i.quality
                ):
                    paretoFrontier = False
                    break

            # add plan i to pareto frontier if it's not dominated
            if paretoFrontier:
                pareto_frontier_plans.append(plan_i)

        print(f"PARETO PLANS: {len(pareto_frontier_plans)}")

        # return the set of plans on the estimated pareto frontier
        return pareto_frontier_plans

    def add_baseline_plans(self, final_plans: List[PhysicalPlan]) -> List[PhysicalPlan]:
        # if specified, include all baseline plans in the final set of plans
        for plan in [self._createBaselinePlan(model) for model in self.available_models]:
            for final_plan in final_plans:
                if plan == final_plan:
                    continue

            final_plans.append(plan)

        return final_plans

    def add_plans_closest_to_frontier(
        self,
        final_plans: List[PhysicalPlan],
        physical_plans: List[PhysicalPlan],
        min_plans: int,
    ) -> List[PhysicalPlan]:

        # if specified, grab up to `min` total plans, and choose the remaining plans
        # based on their smallest agg. distance to the pareto frontier; distance is computed
        # by summing the pct. difference to the pareto frontier across each dimension
        min_distances = []
        for idx, plan in enumerate(physical_plans):
            # determine if this plan is already in the final set of plans
            for final_plan in final_plans:
                if plan == final_plan:
                    continue

            # otherwise compute min distance to plans on pareto frontier
            min_dist, min_dist_idx = np.inf, -1
            for pareto_plan in final_plans:
                time_dist = (plan.total_time - pareto_plan.total_time) / pareto_plan.total_time
                cost_dist = (plan.total_cost - pareto_plan.total_cost) / pareto_plan.total_cost
                quality_dist = (
                    (pareto_plan.quality - plan.quality) / plan.quality if plan.quality > 0 else 10.0
                )
                dist = time_dist + cost_dist + quality_dist
                if dist < min_dist:
                    min_dist = dist
                    min_dist_idx = idx

            min_distances.append((min_dist, min_dist_idx))

        # sort based on distance
        min_distances = sorted(min_distances, key=lambda tup: tup[0])

        # add k closest plans to final_plans
        k = min_plans - len(final_plans)
        k_indices = list(map(lambda tup: tup[1], min_distances[:k]))
        for idx in k_indices:
            final_plans.append(physical_plans[idx])

        return final_plans

    def generate_plans(self, logical_plan: LogicalPlan, sentinels: bool=False) -> List[PhysicalPlan]:
        """Return a set of possible physical plans."""
        # only fetch sentinel plans if specified
        if sentinels:
            models = self.available_models
            assert (
                len(models) > 0
            ), "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"
            sentinel_plans = [self._createSentinelPlan(logical_plan, model) for model in models]
            return sentinel_plans

        # compute all physical plans for this logical plan
        physicalPlans = [
            physicalPlan
            for physicalPlan in self._createPhysicalPlans(logical_plan)
        ]
        print(f"INITIAL PLANS: {len(physicalPlans)}")
        return physicalPlans
