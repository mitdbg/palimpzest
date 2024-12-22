import numpy as np
import pytest


@pytest.fixture
def scan_convert_filter_matrices(scan_convert_filter_sentinel_plan):
    expected_matrices = {}
    for op_idx, op_set in enumerate(scan_convert_filter_sentinel_plan.operator_sets):
        logical_op_id = op_set[0].logical_op_id

        if op_idx == 0:
            time_arr = np.ones((10, 1))
            cost_arr = np.zeros((10, 1))
            selectivity_arr = np.ones((10, 1))
            quality_arr = np.ones((10, 1))

        elif op_idx == 1:
            time_arr = np.ones((10, 3))
            cost_arr = np.ones((10, 3))
            selectivity_arr = np.ones((10, 3))
            quality_arr = np.ones((10, 3))

        elif op_idx == 2:
            time_arr = np.ones((10, 3))
            cost_arr = np.ones((10, 3))
            selectivity_arr = np.concatenate([np.zeros((5,3)), np.ones((5, 3))])
            quality_arr = np.ones((10, 3))

        expected_matrices[logical_op_id] = {
            "time": time_arr,
            "cost": cost_arr,
            "selectivity": selectivity_arr,
            "quality": quality_arr,
        }

    return expected_matrices
