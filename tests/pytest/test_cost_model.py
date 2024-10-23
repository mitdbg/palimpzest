import pytest

import palimpzest as pz
from palimpzest.execution import CostModel


class TestCostModel:
    def test_compute_operator_estimates(
        self,
        simple_plan_sample_execution_data,
        simple_plan_expected_operator_estimates,
    ):
        # construct estimator
        estimator = CostModel(
            source_dataset_id=None,
            sample_execution_data=simple_plan_sample_execution_data,
        )

        # get computed operator estimates
        operator_estimates = estimator.operator_estimates

        # check that all operators are present
        assert sorted(operator_estimates.keys()) == sorted(
            simple_plan_expected_operator_estimates.keys()
        )

        # validate operator estimates
        for (
            op_id,
            expected_op_estimates,
        ) in simple_plan_expected_operator_estimates.items():
            # validate scan operator estimates
            if "scan" in op_id:
                for metric, expected_value in expected_op_estimates.items():
                    assert operator_estimates[op_id][metric] == expected_value

            # validate convert and filter operator estimates
            if "convert" in op_id or "filter" in op_id:
                for (
                    model_name,
                    expected_model_estimates,
                ) in expected_op_estimates.items():
                    for (
                        metric,
                        expected_value,
                    ) in expected_model_estimates.items():
                        assert (
                            operator_estimates[op_id][model_name][metric]
                            == expected_value
                        )

    # TODO: rewrite this test to be agnostic to the simple plan
    @pytest.mark.parametrize(
        argnames=("physical_plan", "expected_cost_est_results"),
        argvalues=[
            pytest.param(
                "cost-est-simple-plan-gpt4-gpt4",
                "cost-est-simple-plan-gpt4-gpt4",
                id="gpt4-gpt4",
            ),
            pytest.param(
                "cost-est-simple-plan-gpt4-gpt35",
                "cost-est-simple-plan-gpt4-gpt35",
                id="gpt4-gpt35",
            ),
            pytest.param(
                "cost-est-simple-plan-gpt4-mixtral",
                "cost-est-simple-plan-gpt4-mixtral",
                id="gpt4-mixtral",
            ),
            pytest.param(
                "cost-est-simple-plan-gpt35-gpt4",
                "cost-est-simple-plan-gpt35-gpt4",
                id="gpt35-gpt4",
            ),
            pytest.param(
                "cost-est-simple-plan-gpt35-gpt35",
                "cost-est-simple-plan-gpt35-gpt35",
                id="gpt35-gpt35",
            ),
            pytest.param(
                "cost-est-simple-plan-gpt35-mixtral",
                "cost-est-simple-plan-gpt35-mixtral",
                id="gpt35-mixtral",
            ),
            pytest.param(
                "cost-est-simple-plan-mixtral-gpt4",
                "cost-est-simple-plan-mixtral-gpt4",
                id="mixtral-gpt4",
            ),
            pytest.param(
                "cost-est-simple-plan-mixtral-gpt35",
                "cost-est-simple-plan-mixtral-gpt35",
                id="mixtral-gpt35",
            ),
            pytest.param(
                "cost-est-simple-plan-mixtral-mixtral",
                "cost-est-simple-plan-mixtral-mixtral",
                id="mixtral-mixtral",
            ),
        ],
        indirect=True,
    )
    def test_estimate_plan_cost(
        self,
        simple_plan_sample_execution_data,
        physical_plan,
        expected_cost_est_results,
    ):
        # register a fake dataset
        dataset_id = "foobar"
        vals = [1, 2, 3, 4, 5, 6]
        pz.DataDirectory().registerDataset(
            vals=vals,
            dataset_id=dataset_id,
        )
        input_cardinality = len(vals)

        # TODO: if we test with a plan other than the simple test plan; this will break
        # get the scan, convert, and filter op ids from the physical_plan and update the simple_plan_sample_execution_data
        scan_op_id = physical_plan.operators[0].get_op_id()
        convert_op_id = physical_plan.operators[1].get_op_id()
        filter_op_id = physical_plan.operators[2].get_op_id()
        test_op_id_to_new_op_id = {
            "scan123": scan_op_id,
            "convert123": convert_op_id,
            "filter123": filter_op_id,
        }
        for record_op_stats in simple_plan_sample_execution_data:
            record_op_stats.op_id = test_op_id_to_new_op_id[
                record_op_stats.op_id
            ]
            if record_op_stats.source_op_id is not None:
                record_op_stats.source_op_id = test_op_id_to_new_op_id[
                    record_op_stats.source_op_id
                ]

        # construct cost model
        cost_model = CostModel(
            source_dataset_id=dataset_id,
            sample_execution_data=simple_plan_sample_execution_data,
        )

        # estimate cost of plan operators
        source_op_estimates = None
        for op in physical_plan:
            op_plan_cost = cost_model(op, source_op_estimates)
            source_op_estimates = op_plan_cost.op_estimates

            # check that estimated time, cost, and quality are as expected
            op_cost, op_time, op_quality, output_cardinality = (
                expected_cost_est_results(op, input_cardinality)
            )
            assert op_plan_cost.cost == op_cost
            assert op_plan_cost.time == op_time
            assert op_plan_cost.quality == op_quality

            # update input_cardinality for next operator
            input_cardinality = output_cardinality
