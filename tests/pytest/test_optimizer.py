from palimpzest.constants import Model
from palimpzest.cost_model import CostModel
from palimpzest.optimizer import LogicalExpression, Group, Optimizer
from palimpzest.operators import *
from palimpzest.policy import *
import palimpzest as pz
import pytest

class TestPrimitives:
    def test_group_id_equality(self, email_schema):
        filter1_op = FilteredScan(
            inputSchema=TextFile,
            outputSchema=TextFile,
            filter=Filter("filter1"),
            depends_on=[],
            targetCacheId="filter1",
        )
        filter1_expr = LogicalExpression(
            operator=filter1_op,
            input_group_ids=[0],
            input_fields=set(['contents']),
            generated_fields=set([]),
            group_id=None,
        )
        filter2_op = FilteredScan(
            inputSchema=TextFile,
            outputSchema=TextFile,
            filter=Filter("filter2"),
            depends_on=[],
            targetCacheId="filter2",
        )
        filter2_expr = LogicalExpression(
            operator=filter2_op,
            input_group_ids=[1],
            input_fields=set(['contents']),
            generated_fields=set([]),
            group_id=None,
        )
        convert_op = ConvertScan(
            inputSchema=TextFile,
            outputSchema=email_schema,
            cardinality=Cardinality.ONE_TO_ONE,
            image_conversion=False,
            depends_on=[],
            targetCacheId="convert1",
        )
        convert_expr = LogicalExpression(
            operator=convert_op,
            input_group_ids=[2],
            input_fields=set(['contents']),
            generated_fields=set([]),
            group_id=None,
        )
        g1 = Group(
            logical_expressions=[convert_expr],
            fields=['sender', 'subject', 'contents', 'filename'],
            filter_strs=[filter1_op.filter.getFilterStr(), filter2_op.filter.getFilterStr()],
        )
        g2 = Group(
            logical_expressions=[filter2_expr],
            fields=['sender', 'subject', 'contents', 'filename'],
            filter_strs=[filter2_op.filter.getFilterStr(), filter1_op.filter.getFilterStr()],
        )
        assert g1.group_id == g2.group_id


class TestOptimizer:

    def test_basic_functionality(self, enron_eval_tiny):
        plan = pz.Dataset(enron_eval_tiny, schema=pz.TextFile)
        policy = MaxQuality()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL],
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 1
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)

    def test_simple_max_quality_convert(self, enron_eval_tiny, email_schema):
        plan = pz.Dataset(enron_eval_tiny, schema=email_schema)
        policy = MaxQuality()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL],
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 2
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], LLMConvertBonded)
        assert physical_plan[1].model == Model.GPT_4

    def test_simple_min_cost_convert(self, enron_eval_tiny, email_schema):
        plan = pz.Dataset(enron_eval_tiny, schema=email_schema)
        policy = MinCost()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL],
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 2
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], CodeSynthesisConvert)

    def test_simple_min_time_convert(self, enron_eval_tiny, email_schema):
        plan = pz.Dataset(enron_eval_tiny, schema=email_schema)
        policy = MinTime()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL],
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 2
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], CodeSynthesisConvert)

    def test_push_down_filter(self, enron_eval_tiny, email_schema):
        plan = pz.Dataset(enron_eval_tiny, schema=email_schema)
        plan = plan.filter("some text filter", depends_on=["contents"])
        policy = MinCost()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL],
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 3
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], LLMFilter)
        assert isinstance(physical_plan[2], CodeSynthesisConvert)

    def test_push_down_two_filters(self, enron_eval_tiny, email_schema):
        plan = pz.Dataset(enron_eval_tiny, schema=email_schema)
        plan = plan.filter("some text filter", depends_on=["contents"])
        plan = plan.filter("another text filter", depends_on=["contents"])
        policy = MinCost()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL],
        )
        physical_plans = optimizer.optimize(plan)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 4
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)
        assert isinstance(physical_plan[1], LLMFilter)
        assert isinstance(physical_plan[2], LLMFilter)
        assert isinstance(physical_plan[3], CodeSynthesisConvert)

    def test_real_estate_logical_reorder(self, real_estate_eval_tiny, real_estate_workload):
        policy = MinCost()
        cost_model = CostModel(real_estate_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL, Model.GPT_4V],
            allow_token_reduction=False,
            allow_code_synth=False,
        )
        physical_plans = optimizer.optimize(real_estate_workload)
        physical_plan = physical_plans[0]

        assert len(physical_plan) == 6
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)  # RealEstateListingFiles
        assert isinstance(physical_plan[1], LLMConvert)            # TextRealEstateListing
        assert isinstance(physical_plan[2], NonLLMFilter)          # TextRealEstateListing(price/addr)
        assert isinstance(physical_plan[3], NonLLMFilter)          # TextRealEstateListing(price/addr)
        assert isinstance(physical_plan[4], LLMConvert)            # ImageRealEstateListing
        assert isinstance(physical_plan[5], LLMFilter)             # ImageRealEstateListing(attractive)

    def test_seven_filters(self, enron_eval_tiny, email_schema):
        plan = pz.Dataset(enron_eval_tiny, schema=email_schema)
        plan = plan.filter("filter1", depends_on=["contents"])
        plan = plan.filter("filter2", depends_on=["contents"])
        plan = plan.filter("filter3", depends_on=["contents"])
        plan = plan.filter("filter4", depends_on=["contents"])
        plan = plan.filter("filter5", depends_on=["contents"])
        plan = plan.filter("filter6", depends_on=["contents"])
        plan = plan.filter("filter7", depends_on=["contents"])
        policy = MinCost()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5, Model.MIXTRAL, Model.GPT_4V],
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