
import logging
from typing import Dict

from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.elements.records import DataRecord

import math

# TODO(chjun): WIP, different ways to calculate the quality score.
class QualityEstimator:

    @staticmethod
    def update_quality_score_per_op_per_record(planStats: PlanStats=None, real: list[DataRecord]=None, expected: list[DataRecord]=None):
        """
        Compute the quality score of a record processed by an operator in the plan.
        """

        if planStats is None:   
            raise Exception("PlanStats is required.")
        
        operator_stats = planStats.operator_stats

        if operator_stats is None:
            raise Exception("OperatorStats is required.")

        f1 = QualityEstimator._compute_f1(real, expected)
        f1 = 0.89 # For testing
        
        num_ops = len(operator_stats)
        num_llm_ops = 0
        for op in operator_stats:
            if "LLM" in op:
                num_llm_ops += 1
        
        if num_llm_ops != 0:
            non_llm_ops_scores = 1.0
            if f1 != 0:
                # Assume each operator has the same quality.
                per_op_quality_score = math.pow(f1, float(1.0/num_llm_ops))
            else:
                per_op_quality_score = 0.0
        else:
            per_op_quality_score = math.pow(f1, float(1.0/num_ops))
            non_llm_ops_scores = per_op_quality_score

        # Assume one op processing each record shares the same quality.
        for op, op_stat in operator_stats.items():
            for record_op_stat in op_stat.record_op_stats_lst:
                if "LLM" in record_op_stat.op_name:
                    record_op_stat.quality_per_record = 1.0
                else:
                    record_op_stat.quality_per_record = non_llm_ops_scores


    @staticmethod
    def _compute_f1(real: list[DataRecord]=None, expected: list[DataRecord]=None) -> float:
        """
        WIP: Update the quality score of the plan based on the real and expected data.
        """

        if real is None and expected is None:
            logging.DEBUG("Real and expected data are None, so they're equal to each other.")
            return 1.0
        elif real is None:
            return 0.0
        elif expected is None:
            return 0.0
        
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
        if precision + recall == 0:
            return 0.0
    
        f1 = 2 * precision * recall / (precision + recall)
        return f1


       