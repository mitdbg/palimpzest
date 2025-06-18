import pytest

from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.utils.model_helpers import get_models


class TestCostModel:
    def test_compute_operator_estimates(
        self, simple_plan_sample_execution_data, simple_plan_expected_operator_estimates
    ):
        # construct estimator
        estimator = CostModel(
            sample_execution_data=simple_plan_sample_execution_data,
            available_models=get_models(),
        )

        # get computed operator estimates
        operator_estimates = estimator.operator_estimates

        # check that all operators are present
        assert sorted(operator_estimates.keys()) == sorted(simple_plan_expected_operator_estimates.keys())

        # validate operator estimates
        for full_op_id, expected_op_estimates in simple_plan_expected_operator_estimates.items():
            # validate scan operator estimates
            if "scan" in full_op_id:
                for metric, expected_value in expected_op_estimates.items():
                    assert operator_estimates[full_op_id][metric] == expected_value

            # validate convert and filter operator estimates
            if "convert" in full_op_id or "filter" in full_op_id:
                for model_name, expected_model_estimates in expected_op_estimates.items():
                    for metric, expected_value in expected_model_estimates.items():
                        assert operator_estimates[full_op_id][model_name][metric] == expected_value

    # TODO: rewrite this test to be agnostic to the simple plan
    @pytest.mark.parametrize(
        argnames=("physical_plan", "expected_cost_est_results"),
        argvalues=[
            pytest.param("cost-est-simple-plan-gpt4-gpt4", "cost-est-simple-plan-gpt4-gpt4", id="gpt4-gpt4"),
            pytest.param("cost-est-simple-plan-gpt4-gpt4m", "cost-est-simple-plan-gpt4-gpt4m", id="gpt4-gpt4m"),
            pytest.param("cost-est-simple-plan-gpt4-mixtral", "cost-est-simple-plan-gpt4-mixtral", id="gpt4-mixtral"),
            pytest.param("cost-est-simple-plan-gpt4m-gpt4", "cost-est-simple-plan-gpt4m-gpt4", id="gpt4m-gpt4"),
            pytest.param("cost-est-simple-plan-gpt4m-gpt4m", "cost-est-simple-plan-gpt4m-gpt4m", id="gpt4m-gpt4m"),
            pytest.param("cost-est-simple-plan-gpt4m-mixtral", "cost-est-simple-plan-gpt4m-mixtral", id="gpt4m-mixtral"),
            pytest.param("cost-est-simple-plan-mixtral-gpt4", "cost-est-simple-plan-mixtral-gpt4", id="mixtral-gpt4"),
            pytest.param("cost-est-simple-plan-mixtral-gpt4m", "cost-est-simple-plan-mixtral-gpt4m", id="mixtral-gpt4m"),
            pytest.param("cost-est-simple-plan-mixtral-mixtral", "cost-est-simple-plan-mixtral-mixtral", id="mixtral-mixtral"),
        ],
        indirect=True,
    )
    def test_estimate_plan_cost(self, simple_plan_sample_execution_data, physical_plan, expected_cost_est_results):
        # TODO: refactor this test to work even if we make changes to simple plans in test/pytest/fixtures/physical_plans.py
        input_cardinality = 6

        # TODO: if we test with a plan other than the simple test plan; this will break
        # get the scan, convert, and filter op ids from the physical_plan and update the simple_plan_sample_execution_data
        scan_full_op_id = physical_plan.operators[0].get_full_op_id()
        convert_full_op_id = physical_plan.operators[1].get_full_op_id()
        filter_full_op_id = physical_plan.operators[2].get_full_op_id()
        test_full_op_id_to_new_full_op_id = {"scan123": scan_full_op_id, "convert123": convert_full_op_id, "filter123": filter_full_op_id}
        for record_op_stats in simple_plan_sample_execution_data:
            record_op_stats.full_op_id = test_full_op_id_to_new_full_op_id[record_op_stats.full_op_id]
            if record_op_stats.source_full_op_id is not None:
                record_op_stats.source_full_op_id = test_full_op_id_to_new_full_op_id[record_op_stats.source_full_op_id]

        # construct cost model
        cost_model = CostModel(
            sample_execution_data=simple_plan_sample_execution_data,
            available_models=get_models(),
        )

        # estimate cost of plan operators
        source_op_estimates = None
        for op in physical_plan:
            op_plan_cost = cost_model(op, source_op_estimates)
            source_op_estimates = op_plan_cost.op_estimates

            # check that estimated time, cost, and quality are as expected
            op_cost, op_time, op_quality, output_cardinality = expected_cost_est_results(op, input_cardinality)
            assert op_plan_cost.cost == op_cost
            assert op_plan_cost.time == op_time
            assert op_plan_cost.quality == op_quality

            # update input_cardinality for next operator
            input_cardinality = output_cardinality
