import pytest

from palimpzest.datamanager import DataDirectory
from palimpzest.optimizer.cost_model import CostModel
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
        for op_id, expected_op_estimates in simple_plan_expected_operator_estimates.items():
            # validate scan operator estimates
            if "scan" in op_id:
                for metric, expected_value in expected_op_estimates.items():
                    assert operator_estimates[op_id][metric] == expected_value

            # validate convert and filter operator estimates
            if "convert" in op_id or "filter" in op_id:
                for model_name, expected_model_estimates in expected_op_estimates.items():
                    for metric, expected_value in expected_model_estimates.items():
                        assert operator_estimates[op_id][model_name][metric] == expected_value

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
        # register a fake dataset
        dataset_id = "foobar"
        vals = [1, 2, 3, 4, 5, 6]
        DataDirectory().register_dataset(
            vals=vals,
            dataset_id=dataset_id,
        )
        input_cardinality = len(vals)

        # TODO: if we test with a plan other than the simple test plan; this will break
        # get the scan, convert, and filter op ids from the physical_plan and update the simple_plan_sample_execution_data
        scan_op_id = physical_plan.operators[0].get_op_id()
        convert_op_id = physical_plan.operators[1].get_op_id()
        filter_op_id = physical_plan.operators[2].get_op_id()
        test_op_id_to_new_op_id = {"scan123": scan_op_id, "convert123": convert_op_id, "filter123": filter_op_id}
        for record_op_stats in simple_plan_sample_execution_data:
            record_op_stats.op_id = test_op_id_to_new_op_id[record_op_stats.op_id]
            if record_op_stats.source_op_id is not None:
                record_op_stats.source_op_id = test_op_id_to_new_op_id[record_op_stats.source_op_id]

        # construct cost model
        cost_model = CostModel(
            sample_execution_data=simple_plan_sample_execution_data,
            available_models=getModels(),
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


# class TestMatrixCompletionCostModel:

#     @pytest.mark.parametrize(
#         argnames=("sentinel_plan", "execution_data", "champion_outputs", "expected_records", "expected_qualities"),
#         argvalues=[
#             # execution data, champion model, and expected outputs all agree
#             pytest.param("scf", "scf", "scf", "scf", "scf", id="scan-convert-filter"),
#             # execution data and champion model agree, expected outputs is empty so we cannot use it to assign quality to filters
#             pytest.param("scf", "scf", "scf", "empty", "scf", id="scan-convert-filter-empty"),
#             # champion model agrees w/expected output (empty) so quality 0 is assigned for records which are not filtered out
#             pytest.param("scf", "scf", "empty", "empty", "empty", id="scan-convert-filter-empty-champion"),
#             # champion model disagrees w/some outputs in other models, so those outputs receive 0 quality
#             pytest.param("scf", "scf-varied", "scf-varied", "empty", "scf-varied", id="scan-convert-filter-varied"),
#             # champion model disagrees w/some outputs in other models, but expected output overrides some champion decisions
#             pytest.param("scf", "scf-varied", "scf-varied", "scf-varied", "scf-varied-override", id="scan-convert-filter-override"),
#             # multi-convert, multi-filter plan; champion model is used to score filters for outputs not present in expected output
#             pytest.param("scffc", "scffc", "scffc", "scffc", "scffc", id="scffc"),
#         ],
#         indirect=True,
#     )
#     def test_score_quality(self, sentinel_plan, execution_data, champion_outputs, expected_records, expected_qualities):
#         # create fake instance of MatrixCompletionCostModel which has a no-op __init__ method;
#         # this will make it possible to test the score_quality() function (which is normally called
#         # in the __init__() method) in isolation
#         class FakeMatrixCompletionCostModel(MatrixCompletionCostModel):
#             def __init__(self, *args, **kwargs):
#                 pass

#         # initialize cost model
#         cost_model = FakeMatrixCompletionCostModel()

#         # call the score_quality() function
#         new_execution_data = cost_model.score_quality(
#             sentinel_plan.operator_sets,
#             execution_data,
#             champion_outputs,
#             expected_records,
#         )

#         # check the quality of all outputs
#         for op_set_id, source_id_to_data_record_sets in new_execution_data.items():
#             for source_id, data_record_sets in source_id_to_data_record_sets.items():
#                 record_set_expected_qualities = expected_qualities[op_set_id][source_id]
#                 for data_record_set, expected_qualities_lst in zip(data_record_sets, record_set_expected_qualities):
#                     for record_op_stats, expected_quality in zip(data_record_set.record_op_stats, expected_qualities_lst):
#                         assert record_op_stats.quality == expected_quality

#     # TODO: need to update test inputs to:
#     # - include sample_matrices in sentinel_plan
#     # - adjust downstream checks to account for lack of physical_op_id_to_matrix_col
#     #
#     # @pytest.mark.parametrize(
#     #     argnames=("sentinel_plan", "execution_data", "expected_matrices"),
#     #     argvalues=[
#     #         pytest.param("scf", "scf", "scf", id="scan-convert-filter"),
#     #     ],
#     #     indirect=True,
#     # )
#     # def test_construct_matrix(self, sentinel_plan, execution_data, expected_matrices):
#     #     # update execution_data to have quality = 1.0 (it will be None by default)
#     #     for op_set in sentinel_plan.operator_sets:
#     #         op_set_id = SentinelPlan.compute_op_set_id(op_set)
#     #         for _, record_sets in execution_data[op_set_id].items():
#     #             for record_set in record_sets:
#     #                 for record_op_stats in record_set.record_op_stats:
#     #                     # set all record_op_stats.quality = 1.0
#     #                     record_op_stats.quality = 1.0

#     #     # create fake instance of MatrixCompletionCostModel which has a no-op __init__ method;
#     #     # this will make it possible to test the construct_matrix() function (which is normally called
#     #     # in the __init__() method) in isolation
#     #     class FakeMatrixCompletionCostModel(MatrixCompletionCostModel):
#     #         def __init__(self, *args, **kwargs):
#     #             pass

#     #     # initialize cost model
#     #     cost_model = FakeMatrixCompletionCostModel()

#     #     # compute mapping from logical_op_id --> sample mask
#     #     op_set_id_to_logical_id = {
#     #         SentinelPlan.compute_op_set_id(op_set): op_set[0].logical_op_id
#     #         for op_set in sentinel_plan.operator_sets
#     #     }
#     #     logical_op_id_to_sample_masks = {
#     #         op_set_id_to_logical_id[op_set_id]: (sample_matrix, record_to_row_map, phys_op_to_col_map)
#     #         for op_set_id, (sample_matrix, record_to_row_map, phys_op_to_col_map) in sentinel_plan.sample_matrices.items()
#     #     }

#     #     # call the score_quality() function
#     #     logical_op_id_to_raw_matrices = cost_model.construct_matrices(execution_data, sentinel_plan.operator_sets, logical_op_id_to_sample_masks)

#     #     # check the mapping from logical_op_ids to matrices
#     #     for op_set in sentinel_plan.operator_sets:
#     #         logical_op_id = op_set[0].logical_op_id
#     #         assert logical_op_id in logical_op_id_to_raw_matrices

#     #         matrix_dict = logical_op_id_to_raw_matrices[logical_op_id]
#     #         expected_matrix_dict = expected_matrices[logical_op_id]
#     #         for key in ["time", "cost", "quality"]:
#     #             assert (matrix_dict[key] == expected_matrix_dict[key]).all()

#     #         # for selectivity, we check that half of the rows are all 0 and half of the rows are all 1
#     #         assert (np.sort(matrix_dict["selectivity"], axis=0) == expected_matrix_dict["selectivity"]).all()

#     #     # check that every physical_op_id maps to a matrix
#     #     for op_set in sentinel_plan.operator_sets:
#     #         for physical_op in op_set:
#     #             phys_op_id = physical_op.get_op_id()
#     #             assert phys_op_id in physical_op_id_to_matrix_col


#     @pytest.mark.parametrize(
#         argnames=("sentinel_plan", "execution_data"),
#         argvalues=[
#             pytest.param("scf", "scf", id="scan-convert-filter"),
#         ],
#         indirect=True,
#     )
#     def test_complete_matrix(self, sentinel_plan, execution_data):
#         logical_op_ids = {op_set[0].logical_op_id for op_set in sentinel_plan.operator_sets}
#         logical_op_id_to_matrices = {
#             logical_op_id: {
#                 "cost": np.array(
#                     [[0.1, 2.0, 0.0],
#                      [0.1, 0.0, 3.0],
#                      [0.0, 2.0, 3.0]],
#                 ),
#                 "time": np.array(
#                     [[0.1, 2.0, 0.0],
#                      [0.1, 0.0, 3.0],
#                      [0.0, 2.0, 3.0]],
#                 ),
#                 "quality": np.array(
#                     [[0.1, 0.2, 0.0],
#                      [0.1, 0.0, 0.3],
#                      [0.0, 0.2, 0.3]],
#                 ),
#                 "selectivity": np.array(
#                     [[0.1, 2.0, 0.0],
#                      [0.1, 0.0, 3.0],
#                      [0.0, 2.0, 3.0]],
#                 ),
#             }
#             for logical_op_id in logical_op_ids
#         } #: Dict[str, Dict[str, np.array]],

#         logical_op_id_to_sample_masks = {
#             logical_op_id: (
#                 np.array(
#                     [[1, 1, 0],
#                     [1, 0, 1],
#                     [0, 1, 1]]
#                 ),
#                 {},
#                 {}
#             )
#             for logical_op_id in logical_op_ids
#         }  #: Dict[str, np.array],
#         logical_op_id_to_expected_matrices = {
#             logical_op_id: {
#                 "cost": np.array(
#                     [[0.1, 2.0, 3.0],
#                      [0.1, 2.0, 3.0],
#                      [0.1, 2.0, 3.0]],
#                 ),
#                 "time": np.array(
#                     [[0.1, 2.0, 3.0],
#                      [0.1, 2.0, 3.0],
#                      [0.1, 2.0, 3.0]],
#                 ),
#                 "quality": np.array(
#                     [[0.1, 0.2, 0.3],
#                      [0.1, 0.2, 0.3],
#                      [0.1, 0.2, 0.3]],
#                 ),
#                 "selectivity": np.array(
#                     [[0.1, 2.0, 3.0],
#                      [0.1, 2.0, 3.0],
#                      [0.1, 2.0, 3.0]],
#                 ),
#             }
#             for logical_op_id in logical_op_ids
#         } #: Dict[str, Dict[str, np.array]],

#         # create fake instance of MatrixCompletionCostModel which has a no-op __init__ method;
#         # this will make it possible to test the construct_matrix() function (which is normally called
#         # in the __init__() method) in isolation
#         class FakeMatrixCompletionCostModel(MatrixCompletionCostModel):
#             def __init__(self, *args, **kwargs):
#                 self.rank = 1
#                 self.num_epochs = 5000

#         # initialize cost model with rank=1
#         cost_model = FakeMatrixCompletionCostModel()

#         # complete matrices
#         logical_op_id_to_completed_matrices = cost_model.complete_matrices(logical_op_id_to_matrices, logical_op_id_to_sample_masks)

#         # verify output
#         for logical_op_id, metric_to_expected_matrix in logical_op_id_to_expected_matrices.items():
#             for metric, expected_matrix in metric_to_expected_matrix.items():
#                 completed_matrix = logical_op_id_to_completed_matrices[logical_op_id][metric]
#                 matrix_diff = np.abs(expected_matrix - completed_matrix)
#                 assert (matrix_diff < 0.1).all()

#     def test_call_method(self, scan_convert_filter_sentinel_plan, cost_model_test_dataset):
#         # set the scan operator for the sentinel plan to read from the cost_model_test_dataset
#         scan_convert_filter_sentinel_plan.operator_sets[0][0].dataset_id = cost_model_test_dataset

#         # create fake instance of MatrixCompletionCostModel which has a no-op __init__ method;
#         # this will make it possible to test the construct_matrix() function (which is normally called
#         # in the __init__() method) in isolation
#         class FakeMatrixCompletionCostModel(MatrixCompletionCostModel):
#             def __init__(self, *args, **kwargs):
#                 self.datadir = DataDirectory()

#                 # construct mapping from each logical operator id to its previous logical operator id
#                 self.logical_op_id_to_prev_logical_op_id = {}
#                 for idx, op_set in enumerate(scan_convert_filter_sentinel_plan.operator_sets):
#                     logical_op_id = op_set[0].logical_op_id
#                     if idx == 0:
#                         self.logical_op_id_to_prev_logical_op_id[logical_op_id] = None
#                     else:
#                         prev_op_set = scan_convert_filter_sentinel_plan.operator_sets[idx - 1]
#                         prev_logical_op_id = prev_op_set[0].logical_op_id
#                         self.logical_op_id_to_prev_logical_op_id[logical_op_id] = prev_logical_op_id

#         # initialize cost model with rank=1
#         cost_model = FakeMatrixCompletionCostModel()

#         # set physical_op_id_to_matrix_col and logical_op_id_to_expected_matrices
#         cost_model.physical_op_id_to_matrix_col = {}
#         cost_model.logical_op_id_to_matrices = {}
#         for op_set in scan_convert_filter_sentinel_plan.operator_sets:
#             logical_op_id = op_set[0].logical_op_id
#             for idx, op in enumerate(op_set):
#                 physical_op_id = op.get_op_id()
#                 cost_model.physical_op_id_to_matrix_col[physical_op_id] = (logical_op_id, idx)

#             cost_model.logical_op_id_to_matrices[logical_op_id] = {
#                 metric: {} for metric in ["cost", "time", "quality", "selectivity"]
#             }
#             arr = np.array([[idx / 10 for idx in range(len(op_set))] for _ in range(3)])
#             for metric in ["cost", "time", "quality"]:
#                 cost_model.logical_op_id_to_matrices[logical_op_id][metric] = arr

#             cost_model.logical_op_id_to_matrices[logical_op_id]["selectivity"] = np.ones((3, len(op_set)))

#         # validate outputs
#         cardinality = len(cost_model.datadir.get_registered_dataset(cost_model_test_dataset))
#         source_op_estimates = None
#         for op_set in scan_convert_filter_sentinel_plan.operator_sets:
#             for idx, physical_op in enumerate(op_set):
#                 plan_cost = cost_model(physical_op, source_op_estimates)

#                 # check that calculations are correct
#                 assert plan_cost.cost - (idx / 10) * cardinality < 1e-6
#                 assert plan_cost.time - (idx / 10) * cardinality < 1e-6
#                 assert plan_cost.quality - (idx / 10) < 1e-6
#                 assert plan_cost.op_estimates.cardinality == cardinality

#             # update source_op_estimates
#             source_op_estimates = plan_cost.op_estimates
