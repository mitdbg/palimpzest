from palimpzest.constants import Cardinality, Model
from palimpzest.corelib.schemas import TextFile
from palimpzest.cost_model import CostModel
from palimpzest.elements.filters import Filter
from palimpzest.operators.code_synthesis_convert import CodeSynthesisConvert
from palimpzest.operators.convert import LLMConvert, LLMConvertBonded
from palimpzest.operators.datasource import MarshalAndScanDataOp
from palimpzest.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.operators.logical import (
    ConvertScan,
    FilteredScan,
)
from palimpzest.optimizer.optimizer import Group, LogicalExpression, Optimizer
from palimpzest.policy import (
    MaxQuality,
    MinCost,
    MinTime,
)
from palimpzest.sets import Dataset


class TestPrimitives:
    def test_group_id_equality(self, email_schema):
        filter1_op = FilteredScan(
            input_schema=TextFile,
            output_schema=TextFile,
            filter=Filter("filter1"),
            depends_on=[],
            target_cache_id="filter1",
        )
        LogicalExpression(
            operator=filter1_op,
            input_group_ids=[0],
            input_fields=set(["contents"]),
            generated_fields=set([]),
            group_id=None,
        )
        filter2_op = FilteredScan(
            input_schema=TextFile,
            output_schema=TextFile,
            filter=Filter("filter2"),
            depends_on=[],
            target_cache_id="filter2",
        )
        filter2_expr = LogicalExpression(
            operator=filter2_op,
            input_group_ids=[1],
            input_fields=set(["contents"]),
            generated_fields=set([]),
            group_id=None,
        )
        convert_op = ConvertScan(
            input_schema=TextFile,
            output_schema=email_schema,
            cardinality=Cardinality.ONE_TO_ONE,
            image_conversion=False,
            depends_on=[],
            target_cache_id="convert1",
        )
        convert_expr = LogicalExpression(
            operator=convert_op,
            input_group_ids=[2],
            input_fields=set(["contents"]),
            generated_fields=set([]),
            group_id=None,
        )
        g1_properties = {
            "filter_strs": set([filter1_op.filter.get_filter_str(), filter2_op.filter.get_filter_str()]),
        }
        g1 = Group(
            logical_expressions=[convert_expr],
            fields=set(["sender", "subject", "contents", "filename"]),
            properties=g1_properties,
        )
        g2_properties = {
            "filter_strs": set([filter2_op.filter.get_filter_str(), filter1_op.filter.get_filter_str()]),
        }
        g2 = Group(
            logical_expressions=[filter2_expr],
            fields=set(["sender", "subject", "contents", "filename"]),
            properties=g2_properties,
        )
        assert g1.group_id == g2.group_id


class TestOptimizer:
    def test_basic_functionality(self, enron_eval_tiny):
        plan = Dataset(enron_eval_tiny, schema=TextFile)
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
        plan = Dataset(enron_eval_tiny, schema=email_schema)
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
        plan = Dataset(enron_eval_tiny, schema=email_schema)
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
        plan = Dataset(enron_eval_tiny, schema=email_schema)
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
        plan = Dataset(enron_eval_tiny, schema=email_schema)
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
        plan = Dataset(enron_eval_tiny, schema=email_schema)
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
        assert isinstance(physical_plan[1], LLMConvert)  # TextRealEstateListing
        assert isinstance(physical_plan[2], NonLLMFilter)  # TextRealEstateListing(price/addr)
        assert isinstance(physical_plan[3], NonLLMFilter)  # TextRealEstateListing(price/addr)
        assert isinstance(physical_plan[4], LLMConvert)  # ImageRealEstateListing
        assert isinstance(physical_plan[5], LLMFilter)  # ImageRealEstateListing(attractive)

    def test_seven_filters(self, enron_eval_tiny, email_schema):
        plan = Dataset(enron_eval_tiny, schema=email_schema)
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
