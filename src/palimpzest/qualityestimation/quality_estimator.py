
import logging

from palimpzest.dataclasses import PlanStats
from palimpzest.elements.records import DataRecord

import math

class QualityEstimator:

    # TODO(chjun): WIP, different ways to calculate the quality score.
    @staticmethod
    def _update_plan_stats(planStats: PlanStats, f1: float):
        if planStats is None:
            raise Exception("PlanStats is required.")
        
        num_op = len(planStats.operator_stats)
        quality_score = math.sqrt(f1, num_op)
        # for op in planStats.operator_stats:
        #     op.


    @staticmethod
    def update_quality_score(planStats: PlanStats=None, real: list[DataRecord]=None, expected: list[DataRecord]=None):
        """
        WIP: Update the quality score of the plan based on the real and expected data.
        """

        if planStats is None:
            raise Exception("PlanStats is required.")
        if real is None and expected is None:
            logging.DEBUG("Real and expected data are None, so they're equal to each other.")
        elif real is None:
            
            return 
        elif expected is None:
            return
        
        # real and expected are not None
        m = len(real)
        n = len(expected)
        i, j = 0, 0
        same_count = 0
        while i < m and j < n:
            if real[i] == expected[j]:
                same_count += 1
            i += 1
            j += 1

        precision, recall = same_count / m, same_count / n
        f1 = 2 * precision * recall / (precision + recall)
        return f1


       