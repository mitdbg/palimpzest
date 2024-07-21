from palimpzest.constants import Model, PromptStrategy, QueryStrategy
from palimpzest.planner import LogicalPlan, PhysicalPlan, Planner
from palimpzest.pruning import PruningStrategy
from palimpzest.utils import getModels

import palimpzest as pz
import palimpzest.operators as pz_ops

from typing import List, Optional

import numpy as np

import palimpzest.strategies as physical_strategies


class PhysicalPlanner(Planner):
    def __init__(
        self,
        pruning_strategy: Optional[PruningStrategy]=None,
        available_models: Optional[List[Model]]=[],
        allow_bonded_query: Optional[bool]=True,
        allow_conventional_query: Optional[bool]=False,
        allow_model_selection: Optional[bool]=True,
        allow_code_synth: Optional[bool]=True,
        allow_token_reduction: Optional[bool]=True,
        shouldProfile: Optional[bool]=True,
        no_cache: Optional[bool]=False,
        verbose: Optional[bool]=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pruning_strategy = pruning_strategy
        self.available_models = available_models
        self.allow_model_selection = allow_model_selection
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.shouldProfile = shouldProfile
        self.no_cache = no_cache
        self.verbose = verbose
        self.physical_ops = [op for op in pz_ops.PHYSICAL_OPERATORS if op.final]

        # This is a dictionary where the physical planner will keep track for all the logical plan operators defined,
        # which physical operators are available to implement them.
        self.logical_physical_map = {}
        for logical_op in pz_ops.LOGICAL_OPERATORS:
            self.logical_physical_map[logical_op] = []
            for physical_op in self.physical_ops:
                if physical_op.implements(logical_op):
                    self.logical_physical_map[logical_op].append(physical_op)

            # NOTE: If model selection is turned off you can not have LLMFilters b/c the only place where "final"
            #       LLMFilters are produced is in the ModelSelectionFilterStrategy
            for strategy in physical_strategies.REGISTERED_STRATEGIES:
                if not self.allow_bonded_query and issubclass(strategy, physical_strategies.LLMBondedConvertStrategy):
                    continue
                if not self.allow_bonded_query and issubclass(strategy, physical_strategies.TokenReducedBondedConvertStrategy):
                    continue
                if not self.allow_conventional_query and issubclass(strategy, physical_strategies.LLMConventionalConvertStrategy):
                    continue
                if not self.allow_conventional_query and issubclass(strategy, physical_strategies.TokenReducedConventionalConvertStrategy):
                    continue
                if not self.allow_token_reduction and issubclass(strategy, physical_strategies.TokenReductionStrategy):
                    continue                    
                if not self.allow_code_synth and issubclass(strategy, physical_strategies.CodeSynthesisConvertStrategy):
                    continue

                if strategy.logical_op_class == logical_op:
                    ops = strategy(available_models=self.available_models,
                                    prompt_strategy=PromptStrategy.DSPY_COT_QA if logical_op != pz_ops.FilteredScan else PromptStrategy.DSPY_COT_BOOL,
                                    token_budgets=[0.1, 0.5, 0.9],)
                    self.logical_physical_map.get(logical_op, []).extend(ops)

        self.hardcoded_converts = [x for x in self.logical_physical_map[pz_ops.ConvertScan] if issubclass(x, pz.HardcodedConvert)]
        # print("Available strategies")
        # print(physical_strategies.REGISTERED_STRATEGIES)
        # print("Maps")
        # print(self.logical_physical_map[pz_ops.ConvertScan])
        # print(self.logical_physical_map[pz_ops.FilteredScan])

    def _getSubPlanModel(self, subplan: PhysicalPlan) -> Optional[Model]:
        """
        Returns the model used by the subplan if it is using one, otherwise returns None.
        """
        for op in subplan.operators:
            if getattr(op, "model", None) not in [None, Model.GPT_4V]:
                return op.model

        return None

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

    def add_baseline_plans(self, final_plans: List[PhysicalPlan], initial_logical_plan: LogicalPlan) -> List[PhysicalPlan]:
        """
        Generate all baseline plans and add them to the final set of plans if they do not already exist.
        """
        # store original parameters
        allow_bonded_query = self.allow_bonded_query
        allow_conventional_query = self.allow_conventional_query
        allow_model_selection = self.allow_model_selection
        allow_code_synth = self.allow_code_synth
        allow_token_reduction = self.allow_token_reduction

        # update parameters to construct baseline plans
        self.allow_bonded_query = True
        self.allow_conventional_query = False
        self.allow_model_selection = False
        self.allow_code_synth = False
        self.allow_token_reduction = False

        # generate baseline plans and add them to final plans
        for plan in self.generate_plans(initial_logical_plan):
            for final_plan in final_plans:
                if plan == final_plan:
                    continue

            final_plans.append(plan)

        # set parameters back to original
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction

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

    def generate_plans(self, logical_plan: LogicalPlan) -> List[PhysicalPlan]:
        """
        Return the set of possible physical plans for the given logical plan and planner instance.
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

        # get logical operators and execution parameters
        operators = logical_plan.operators
        execution_parameters = {"shouldProfile": self.shouldProfile}

        # construct the set of physical plans
        all_plans = []
        for logical_op in operators:

            if isinstance(logical_op, pz_ops.ConvertScan):
                plans = []
                for subplan in all_plans:
                    hardcoded_fns = [x for x in self.hardcoded_converts if x.materializes(logical_op)]
                    if len(hardcoded_fns) > 0:
                        for op_class in hardcoded_fns:
                            physical_op = op_class(
                                    inputSchema=logical_op.inputSchema,
                                    outputSchema=logical_op.outputSchema,
                                    shouldProfile=self.shouldProfile,
                                    verbose=self.verbose,
                                )
                            new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                            plans.append(new_physical_plan)
                            break
                    else:
                        for op_class in self.logical_physical_map[type(logical_op)]:
                            if not op_class.materializes(logical_op):
                                continue
                            if not self.allow_model_selection and getattr(op_class, "model", None) in getModels(include_vision=False):
                                # skip this physical op if the subplan already uses a model that is different
                                # from the one used by this op_class; (we allow mixing vision models for now)
                                subplan_model = self._getSubPlanModel(subplan)
                                if subplan_model is not None and subplan_model != op_class.model:
                                    continue
                            physical_op = op_class(
                                inputSchema=logical_op.inputSchema,
                                outputSchema=logical_op.outputSchema,
                                image_conversion=logical_op.image_conversion,
                                query_strategy=QueryStrategy.BONDED_WITH_FALLBACK,
                                shouldProfile=self.shouldProfile,
                                verbose=self.verbose,
                            )
                            new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                            plans.append(new_physical_plan)

                # update all_plans
                all_plans = plans

            elif isinstance(logical_op, pz_ops.FilteredScan):
                plans = []
                for subplan in all_plans:
                    # TODO: if non-llm filter, don't iterate over all plan possibilities
                    for op_class in self.logical_physical_map[type(logical_op)]:
                        if not op_class.materializes(logical_op):
                            continue
                        if not self.allow_model_selection and getattr(op_class, "model", None) in getModels(include_vision=False):
                            # skip this physical op if the subplan already uses a model that is different
                            # from the one used by this op_class; (we allow mixing vision models for now)
                            subplan_model = self._getSubPlanModel(subplan)
                            if subplan_model is not None and subplan_model != op_class.model:
                                continue
                        physical_op = op_class(
                            inputSchema=logical_op.inputSchema,
                            outputSchema=logical_op.outputSchema,
                            filter=logical_op.filter,
                            shouldProfile=self.shouldProfile,
                            verbose=self.verbose,
                        )
                        new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                        plans.append(new_physical_plan)

                # update all_plans
                all_plans = plans

            else:
                op_class = self.logical_physical_map[type(logical_op)][0]
                kw_parameters = logical_op.getParameters()
                kw_parameters.update(execution_parameters)
                physical_op = op_class(**kw_parameters)

                # base case, if this operator is a BaseScan set all_plans to be the physical plan with just this operator
                # This also happens if the operator is a CacheScan and all_plans is empty
                if (
                    isinstance(logical_op, pz_ops.BaseScan) or
                    (isinstance(logical_op, pz_ops.CacheScan) and all_plans == [])
                ):
                    all_plans = [PhysicalPlan(operators=[physical_op], datasetIdentifier=datasetIdentifier)]
                else:
                    plans = []
                    for subplan in all_plans:
                        new_physical_plan = PhysicalPlan.fromOpsAndSubPlan([physical_op], subplan)
                    all_plans = plans

            # prune plans
            if self.pruning_strategy is not None:
                all_plans = [plan for plan in all_plans if not self.pruning_strategy.prune_plan(plan)]
                self.pruning_strategy.clear_frontier()

        if self.verbose:
            print(f"PHYSICAL PLANS: {len(all_plans)}")

        return all_plans
