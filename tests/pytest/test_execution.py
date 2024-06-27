from conftest import ENRON_EVAL_TINY_DATASET_ID

from palimpzest.corelib import File, TextFile
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.elements import Filter
from palimpzest.execution import Execute, SimpleExecution
from palimpzest.operators import *
from palimpzest.planner import PhysicalPlan
from palimpzest.policy import MaxQuality
from palimpzest.strategies import ModelSelectionFilterStrategy

import os
import pytest

# TODO: mock out all model calls

class TestExecutionNoCache:

    def test_set_source_dataset_id(self, enron_eval):
        simple_execution = SimpleExecution()
        simple_execution.set_source_dataset_id(enron_eval)
        assert simple_execution.source_dataset_id == ENRON_EVAL_TINY_DATASET_ID

    # def test_legal_discovery(self, enron_eval):
    #     output = Execute(enron_eval, policy=MaxQuality(), num_samples=2, nocache=True)

    # TODO: register dataset in fixture
    def test_execute_sequential_simple_scan(self):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        plan_stats = PlanStats(plan.plan_id())
        op_id = scanOp.get_op_id()
        plan_stats.operator_stats[op_id] = OperatorStats(op_idx=0, op_id=op_id, op_name=scanOp.op_name())

        simple_execution = SimpleExecution(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling a single record
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=1)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename == "testdata/enron-eval-tiny/buy-r-inbox-628.txt"
        assert hasattr(dr, 'contents') and dr.contents != None

        operator_stats = plan_stats.operator_stats[op_id]
        assert operator_stats.total_op_time > 0.0
        assert operator_stats.total_op_cost == 0.0

        record_stats = operator_stats.record_op_stats_lst[0]
        assert record_stats.record_uuid == dr._uuid
        assert record_stats.record_parent_uuid is None
        assert record_stats.op_id == op_id
        assert record_stats.op_name == "MarshalAndScanDataOp"
        assert record_stats.time_per_record > 0.0
        assert record_stats.cost_per_record == 0.0
        assert record_stats.record_state == dr._asDict(include_bytes=False)

        # test full scan
        simple_execution = SimpleExecution(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, float("inf"))

        assert len(output_records) == 6

        expected_filenames = sorted(os.listdir("testdata/enron-eval-tiny"))
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename == os.path.join("testdata/enron-eval-tiny/", expected_filename)
            assert hasattr(dr, 'contents') and dr.contents != None


    def test_execute_sequential_with_non_llm_filter(self):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        def filter_buy_emails(record):
            time.sleep(0.001)
            return "buy" not in record.filename
        filter = Filter(filterFn=filter_buy_emails)
        filterOp = NonLLMFilter(inputSchema=File, outputSchema=File, filter=filter, targetCacheId="abc123", shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp, filterOp],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        plan_stats = PlanStats(plan.plan_id())
        for op in plan.operators:
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=0, op_id=op_id, op_name=op.op_name())

        simple_execution = SimpleExecution(num_samples=3, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling three records, with one making it past the filter
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=3)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename == "testdata/enron-eval-tiny/kaminski-v-deleted-items-1902.txt"
        assert hasattr(dr, 'contents') and dr.contents != None

        for op in plan.operators:
            op_id = op.get_op_id()
            operator_stats = plan_stats.operator_stats[op_id]
            assert operator_stats.total_op_time > 0.0
            assert operator_stats.total_op_cost == 0.0

            if isinstance(op, FilterOp):
                record_stats = operator_stats.record_op_stats_lst[-1]
                assert record_stats.record_uuid == dr._uuid
                assert record_stats.record_parent_uuid == dr._parent_uuid
                assert record_stats.op_id == op_id
                assert record_stats.op_name == op.op_name()
                assert record_stats.time_per_record > 0.0
                assert record_stats.cost_per_record == 0.0
                assert record_stats.record_state == dr._asDict(include_bytes=False)
                assert record_stats.fn_call_duration_secs > 0
                assert record_stats.filter_str == filter.getFilterStr()

        # test full scan
        simple_execution = SimpleExecution(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=float("inf"))

        assert len(output_records) == 4

        expected_filenames = [fn for fn in sorted(os.listdir("testdata/enron-eval-tiny")) if "buy" not in fn]
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename == os.path.join("testdata/enron-eval-tiny/", expected_filename)
            assert hasattr(dr, 'contents') and dr.contents != None

    # TODO: mock response from GPT_3_5 if it's wrong
    def test_execute_sequential_with_llm_filter(self):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        filter = Filter("The filename does not contain the string 'buy'")
        filterOpClass = ModelSelectionFilterStrategy(available_models=[Model.GPT_3_5], prompt_strategy=PromptStrategy.DSPY_COT_BOOL)
        filterOp = filterOpClass[0](inputSchema=File, outputSchema=File, filter=filter, targetCacheId="abc123", shouldProfile=True,)
        plan = PhysicalPlan(
            operators=[scanOp, filterOp],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        plan_stats = PlanStats(plan.plan_id())
        for op in plan.operators:
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=0, op_id=op_id, op_name=op.op_name())

        simple_execution = SimpleExecution(num_samples=3, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling three records, with one making it past the filter
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=3)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename == "testdata/enron-eval-tiny/kaminski-v-deleted-items-1902.txt"
        assert hasattr(dr, 'contents') and dr.contents != None

        for op in plan.operators:
            op_id = op.get_op_id()
            operator_stats = plan_stats.operator_stats[op_id]
            assert operator_stats.total_op_time > 0.0
            if isinstance(op, MarshalAndScanDataOp):
                assert operator_stats.total_op_cost == 0.0

            if isinstance(op, FilterOp):
                assert operator_stats.total_op_cost > 0.0
                record_stats = operator_stats.record_op_stats_lst[-1]
                assert record_stats.record_uuid == dr._uuid
                assert record_stats.record_parent_uuid == dr._parent_uuid
                assert record_stats.op_id == op_id
                assert record_stats.op_name == op.op_name()
                assert record_stats.time_per_record > 0.0
                assert record_stats.cost_per_record > 0.0
                assert record_stats.record_state == dr._asDict(include_bytes=False)
                assert record_stats.llm_call_duration_secs > 0
                assert record_stats.filter_str == filter.getFilterStr()

        # test full scan
        simple_execution = SimpleExecution(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=float("inf"))

        assert len(output_records) == 4

        expected_filenames = [fn for fn in sorted(os.listdir("testdata/enron-eval-tiny")) if "buy" not in fn]
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename == os.path.join("testdata/enron-eval-tiny/", expected_filename)
            assert hasattr(dr, 'contents') and dr.contents != None


    def test_execute_dag_with_hardcoded_convert(self):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        convertOp = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp, convertOp],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        plan_stats = PlanStats(plan.plan_id())
        for op in plan.operators:
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=0, op_id=op_id, op_name=op.op_name())

        simple_execution = SimpleExecution(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling three records, with one making it past the filter
        output_records, plan_stats = simple_execution.execute_dag(plan, plan_stats, num_samples=1)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename == "testdata/enron-eval-tiny/buy-r-inbox-628.txt"
        assert hasattr(dr, 'contents') and dr.contents != None

        for op in plan.operators:
            op_id = op.get_op_id()
            operator_stats = plan_stats.operator_stats[op_id]
            assert operator_stats.total_op_time > 0.0
            assert operator_stats.total_op_cost == 0.0

            if isinstance(op, HardcodedConvert):
                record_stats = operator_stats.record_op_stats_lst[-1]
                assert record_stats.record_uuid == dr._uuid
                assert record_stats.record_parent_uuid == dr._parent_uuid
                assert record_stats.op_id == op_id
                assert record_stats.op_name == op.op_name()
                assert record_stats.time_per_record > 0.0
                assert record_stats.cost_per_record == 0.0
                assert record_stats.record_state == dr._asDict(include_bytes=False)

        # test full scan
        simple_execution = SimpleExecution(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_dag(plan, plan_stats)

        assert len(output_records) == 6

        expected_filenames = sorted(os.listdir("testdata/enron-eval-tiny"))
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename == os.path.join("testdata/enron-eval-tiny/", expected_filename)
            assert hasattr(dr, 'contents') and dr.contents != None

    # # TODO
    # def test_execute_dag_with_aggregate(self):
    #     raise Exception("TODO")

    # # TODO
    # def test_execute_dag_with_limit(self):
    #     raise Exception("TODO")
