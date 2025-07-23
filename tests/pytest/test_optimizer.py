import time

import pytest

from palimpzest.constants import Cardinality, Model
from palimpzest.core.elements.filters import Filter
from palimpzest.core.lib.schemas import TextFile
from palimpzest.core.models import OperatorCostEstimates, PlanCost
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.query.operators.code_synthesis_convert import CodeSynthesisConvert
from palimpzest.query.operators.convert import LLMConvert, LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.logical import ConvertScan, FilteredScan
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import MarshalAndScanDataOp, ScanPhysicalOp
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.primitives import Group, LogicalExpression


class TestPrimitives:
    def test_group_id_equality(self, email_schema):
        filter1_op = FilteredScan(
            input_schema=TextFile,
            output_schema=TextFile,
            filter=Filter("filter1"),
            depends_on=[],
        )
        LogicalExpression(
            operator=filter1_op,
            input_group_ids=[0],
            input_fields={"contents": TextFile.model_fields["contents"]},
            depends_on_field_names=set(["contents"]),
            generated_fields={},
            group_id=None,
        )
        filter2_op = FilteredScan(
            input_schema=TextFile,
            output_schema=TextFile,
            filter=Filter("filter2"),
            depends_on=[],
        )
        filter2_expr = LogicalExpression(
            operator=filter2_op,
            input_group_ids=[1],
            input_fields={"contents": TextFile.model_fields["contents"]},
            depends_on_field_names=set(["contents"]),
            generated_fields={},
            group_id=None,
        )
        convert_op = ConvertScan(
            input_schema=TextFile,
            output_schema=email_schema,
            cardinality=Cardinality.ONE_TO_ONE,
            depends_on=[],
        )
        convert_expr = LogicalExpression(
            operator=convert_op,
            input_group_ids=[2],
            input_fields={"contents": TextFile.model_fields["contents"]},
            depends_on_field_names=set(["contents"]),
            generated_fields={
                "sender": email_schema.model_fields["sender"],
                "subject": email_schema.model_fields["subject"],
            },
            group_id=None,
        )
        g1_properties = {
            "filter_strs": set([filter1_op.filter.get_filter_str(), filter2_op.filter.get_filter_str()]),
        }
        g1 = Group(
            logical_expressions=[convert_expr],
            fields={
                "sender": email_schema.model_fields["sender"],
                "subject": email_schema.model_fields["subject"],
                "contents": TextFile.model_fields["contents"],
                "filename": TextFile.model_fields["filename"],
            },
            properties=g1_properties,
        )
        g2_properties = {
            "filter_strs": set([filter2_op.filter.get_filter_str(), filter1_op.filter.get_filter_str()]),
        }
        g2 = Group(
            logical_expressions=[filter2_expr],
            fields={
                "sender": email_schema.model_fields["sender"],
                "subject": email_schema.model_fields["subject"],
                "contents": TextFile.model_fields["contents"],
                "filename": TextFile.model_fields["filename"],
            },
            properties=g2_properties,
        )
        assert g1.group_id == g2.group_id


@pytest.mark.parametrize(
    argnames=("opt_strategy",),
    argvalues=[
        pytest.param(OptimizationStrategyType.GREEDY, id="greedy"),
        pytest.param(OptimizationStrategyType.PARETO, id="pareto"),
    ],
)
class TestOptimizer:
    def test_basic_functionality(self, enron_eval_tiny, opt_strategy):
        plan = enron_eval_tiny
        policy = MaxQuality()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            optimizer_strategy=opt_strategy,
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 1
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)

    def test_simple_max_quality_convert(self, enron_eval_tiny, email_schema, opt_strategy):
        plan = enron_eval_tiny
        plan = plan.sem_add_columns(email_schema)
        policy = MaxQuality()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            optimizer_strategy=opt_strategy,
            allow_code_synth=False,
            allow_rag_reduction=False,
            allow_mixtures=False,
            allow_critic=False,
            allow_split_merge=False,
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 2
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], LLMConvertBonded)
        assert physical_plan[1].model == Model.GPT_4o

    def test_simple_min_cost_convert(self, enron_eval_tiny, email_schema, opt_strategy):
        plan = enron_eval_tiny
        plan = plan.sem_add_columns(email_schema)
        policy = MinCost()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            optimizer_strategy=opt_strategy,
            allow_code_synth=True,
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 2
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], CodeSynthesisConvert)

    def test_simple_min_time_convert(self, enron_eval_tiny, email_schema, opt_strategy):
        plan = enron_eval_tiny
        plan = plan.sem_add_columns(email_schema)
        policy = MinTime()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            optimizer_strategy=opt_strategy,
            allow_code_synth=True,
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 2
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], CodeSynthesisConvert)

    def test_push_down_filter(self, enron_eval_tiny, email_schema, opt_strategy):
        plan = enron_eval_tiny
        plan = plan.sem_add_columns(email_schema)
        plan = plan.sem_filter("some text filter", depends_on=["contents"])
        policy = MinCost()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            optimizer_strategy=opt_strategy,
            allow_code_synth=True,
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 3
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], LLMFilter)
        assert isinstance(physical_plan[2], CodeSynthesisConvert)

    def test_push_down_two_filters(self, enron_eval_tiny, email_schema, opt_strategy):
        plan = enron_eval_tiny
        plan = plan.sem_add_columns(email_schema)
        plan = plan.sem_filter("some text filter", depends_on=["contents"])
        plan = plan.sem_filter("another text filter", depends_on=["contents"])
        policy = MinCost()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            optimizer_strategy=opt_strategy,
            allow_code_synth=True,
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 4
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], LLMFilter)
        assert isinstance(physical_plan[2], LLMFilter)
        assert isinstance(physical_plan[3], CodeSynthesisConvert)

    def test_real_estate_logical_reorder(self, real_estate_workload, opt_strategy):
        policy = MinCost()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            allow_code_synth=False,
            allow_rag_reduction=False,
            allow_mixtures=False,
            allow_critic=False,
            allow_split_merge=False,
            optimizer_strategy=opt_strategy,
        )
        physical_plans = optimizer.optimize(real_estate_workload)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 6
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)  # RealEstateListingFiles
        assert isinstance(physical_plan[1], LLMConvert)  # TextRealEstateListing
        assert isinstance(physical_plan[2], NonLLMFilter)  # TextRealEstateListing(price/addr)
        assert isinstance(physical_plan[3], NonLLMFilter)  # TextRealEstateListing(price/addr)
        assert isinstance(physical_plan[4], LLMConvert)  # ImageRealEstateListing
        assert isinstance(physical_plan[5], LLMFilter)  # ImageRealEstateListing(attractive)

    def test_seven_filters(self, enron_eval_tiny, email_schema, opt_strategy):
        start_time = time.time()

        plan = enron_eval_tiny
        plan = plan.sem_add_columns(email_schema)
        plan = plan.sem_filter("filter1", depends_on=["contents"])
        plan = plan.sem_filter("filter2", depends_on=["contents"])
        plan = plan.sem_filter("filter3", depends_on=["contents"])
        plan = plan.sem_filter("filter4", depends_on=["contents"])
        plan = plan.sem_filter("filter5", depends_on=["contents"])
        plan = plan.sem_filter("filter6", depends_on=["contents"])
        plan = plan.sem_filter("filter7", depends_on=["contents"])
        policy = MinCost()
        cost_model = CostModel(sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL],
            optimizer_strategy=opt_strategy,
            allow_code_synth=True,
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 9
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], LLMFilter)
        assert isinstance(physical_plan[2], LLMFilter)
        assert isinstance(physical_plan[3], LLMFilter)
        assert isinstance(physical_plan[4], LLMFilter)
        assert isinstance(physical_plan[5], LLMFilter)
        assert isinstance(physical_plan[6], LLMFilter)
        assert isinstance(physical_plan[7], LLMFilter)
        assert isinstance(physical_plan[8], CodeSynthesisConvert)

        assert time.time() - start_time < 5, (
            "Optimizer should complete this test within 2 to 5 seconds; if it's failed, something has caused a regression, and you should ping Matthew Russo (mdrusso@mit.edu)"
        )


class MockSampleBasedCostModel:
    """ """

    def __init__(self, operator_to_stats):
        # construct cost, time, quality, and selectivity matrices for each operator set;
        self.operator_to_stats = operator_to_stats

        # compute set of costed full op ids from operator_to_stats
        self.costed_full_op_ids = set(
            [
                full_op_id
                for _, full_op_id_to_stats in self.operator_to_stats.items()
                for full_op_id, _ in full_op_id_to_stats.items()
            ]
        )

    def get_costed_full_op_ids(self):
        return self.costed_full_op_ids

    def __call__(
        self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None
    ) -> PlanCost:
        # NOTE: some physical operators may not have any sample execution data in this cost model;
        #       these physical operators are filtered out of the Optimizer, thus we can assume that
        #       we will have execution data for each operator passed into __call__; nevertheless, we
        #       still perform a sanity check
        # look up physical and logical op ids associated with this physical operator
        full_op_id = operator.get_full_op_id()
        logical_op_id = operator.logical_op_id
        assert self.operator_to_stats.get(logical_op_id).get(full_op_id) is not None, (
            f"No execution data for {str(operator)}"
        )

        # look up stats for this operation
        est_cost_per_record = self.operator_to_stats[logical_op_id][full_op_id]["cost"]
        est_time_per_record = self.operator_to_stats[logical_op_id][full_op_id]["time"]
        est_quality = self.operator_to_stats[logical_op_id][full_op_id]["quality"]
        est_selectivity = self.operator_to_stats[logical_op_id][full_op_id]["selectivity"]

        # create source_op_estimates for scan operators if they are not provided
        if isinstance(operator, ScanPhysicalOp):
            # get handle to scan operator and pre-compute its size (number of records)
            datasource_len = len(operator.datasource)

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

        # generate new set of OperatorCostEstimates
        op_estimates = OperatorCostEstimates(
            cardinality=est_selectivity * source_op_estimates.cardinality,
            time_per_record=est_time_per_record,
            cost_per_record=est_cost_per_record,
            quality=est_quality,
        )

        # compute estimates for this operator
        op_time = op_estimates.time_per_record * source_op_estimates.cardinality
        op_cost = op_estimates.cost_per_record * source_op_estimates.cardinality
        op_quality = op_estimates.quality

        # construct and return op estimates
        return PlanCost(cost=op_cost, time=op_time, quality=op_quality, op_estimates=op_estimates)


@pytest.mark.parametrize(
    argnames=("workload", "policy", "operator_to_stats", "expected_plan"),
    argvalues=[
        pytest.param("three-converts", "mincost", "3c-mincost", "3c-mincost", id="3c-mincost"),
        pytest.param("three-converts", "maxquality", "3c-maxquality", "3c-maxquality", id="3c-maxquality"),
        pytest.param(
            "three-converts",
            "mincost@quality=0.8",
            "3c-mincost@quality=0.8",
            "3c-mincost@quality=0.8",
            id="3c-mincostfixedquality",
        ),
        pytest.param(
            "three-converts",
            "maxquality@cost=1.0",
            "3c-maxquality@cost=1.0",
            "3c-maxquality@cost=1.0",
            id="3c-maxqualityfixedcost",
        ),
        pytest.param("one-filter-one-convert", "mincost", "1f-1c-mincost", "1f-1c-mincost", id="1f-1c-mincost"),
        pytest.param("two-converts-two-filters", "mincost", "2c-2f-mincost", "2c-2f-mincost", id="2c-2f-mincost"),
        pytest.param(
            "two-converts-two-filters", "maxquality", "2c-2f-maxquality", "2c-2f-maxquality", id="2c-2f-maxquality"
        ),
        pytest.param(
            "two-converts-two-filters",
            "mincost@quality=0.8",
            "2c-2f-mincost@quality=0.8",
            "2c-2f-mincost@quality=0.8",
            id="2c-2f-mincostfixedquality",
        ),
        pytest.param(
            "two-converts-two-filters",
            "maxquality@cost=1.0",
            "2c-2f-maxquality@cost=1.0",
            "2c-2f-maxquality@cost=1.0",
            id="2c-2f-maxqualityfixedcost",
        ),
    ],
    indirect=True,
)
class TestParetoOptimizer:
    def test_pareto_optimization_strategy(self, workload, policy, operator_to_stats, expected_plan):
        # initialize cost model with sample execution data
        cost_model = MockSampleBasedCostModel(operator_to_stats)

        # run optimizer using the cost model and the given policy
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            verbose=True,
            available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.LLAMA3_3_70B],
            optimizer_strategy=OptimizationStrategyType.PARETO,
            allow_code_synth=False,
            allow_rag_reduction=False,
            allow_mixtures=False,
            allow_critic=False,
            allow_split_merge=False,
        )
        # run optimizer to get physical plan
        physical_plans = optimizer.optimize(workload)
        physical_plan = physical_plans[0]

        # assert that physical plan matches expected plan
        assert physical_plan.plan_cost.quality == pytest.approx(expected_plan.plan_cost.quality)
        assert physical_plan.plan_cost.cost == pytest.approx(expected_plan.plan_cost.cost)
        assert physical_plan.plan_cost.time == pytest.approx(expected_plan.plan_cost.time)
        assert physical_plan.plan_id == expected_plan.plan_id
