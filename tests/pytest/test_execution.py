from conftest import ENRON_EVAL_TINY_DATASET_ID

from palimpzest.corelib import File, TextFile
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.elements import Filter
from palimpzest.execution import Execute, SimpleExecution
from palimpzest.operators import *
from palimpzest.planner import PhysicalPlan
from palimpzest.policy import MaxQuality
<<<<<<< HEAD
from palimpzest.strategies import ModelSelectionFilterStrategy
=======
from palimpzest.strategies import ModelSelectionFilterStrategy, BondedQueryConvertStrategy
>>>>>>> c7142cd (debugging llm convert)

import os
import time
import pytest

# TODO: mock out all model calls
# custom class to be the mock return value
# will override the requests.Response returned from requests.get
class MockLLMFilterCall:

    # mock __call__() method always returns the desired output
    def __call__(self, candidate):
        start_time = time.time()
        text_content = candidate._asJSONStr(include_bytes=False)
        response, _, gen_stats = self.generator.generate(
            context=text_content,
            question=self.filter.filterCondition,
        )
        response = str("buy" not in candidate.filename)

        # compute whether the record passed the filter or not
        passed_filter = (
            "true" in response.lower()
            if response is not None
            else False
        )

        # NOTE: this will treat the cost of failed LLM invocations as having 0.0 tokens and dollars,
        #       when in reality this is only true if the error in generator.generate() happens before
        #       the invocation of the LLM -- not if it happens after. (If it happens *during* the
        #       invocation, then it's difficult to say what the true cost really should be). I think
        #       the best solution is to place a try-except inside of the DSPyGenerator to still capture
        #       and return the gen_stats if/when there is an error after invocation.
        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=candidate._uuid,
            record_parent_uuid=candidate._parent_uuid,
            record_state=candidate._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=gen_stats.get('cost_per_record', 0.0),
            model_name=self.model.value,
            filter_str=self.filter.getFilterStr(),
            total_input_tokens=gen_stats.get('input_tokens', 0.0),
            total_output_tokens=gen_stats.get('output_tokens', 0.0),
            total_input_cost=gen_stats.get('input_cost', 0.0),
            total_output_cost=gen_stats.get('output_cost', 0.0),
            llm_call_duration_secs=gen_stats.get('llm_call_duration_secs', 0.0),
            answer=response,
            passed_filter=passed_filter,
        )

        # set _passed_filter attribute and return
        setattr(candidate, "_passed_filter", passed_filter)

        return [candidate], [record_op_stats]

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
        assert dr.filename.endswith("buy-r-inbox-628.txt")
        assert getattr(dr, 'contents', None) != None

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
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None


    def test_execute_sequential_with_non_llm_filter(self):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        def filter_buy_emails(record):
            time.sleep(0.001)
            return "buy" not in record.filename
        filter = Filter(filterFn=filter_buy_emails)
        filterOpClass = ModelSelectionFilterStrategy(available_models=[Model.GPT_3_5], prompt_strategy=PromptStrategy.DSPY_COT_BOOL)
        filterOp = filterOpClass[0](inputSchema=File, outputSchema=File, filter=filter, targetCacheId="abc123", shouldProfile=True)
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
        assert dr.filename.endswith("kaminski-v-deleted-items-1902.txt")
        assert getattr(dr, 'contents', None) != None

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
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None

    # TODO: mock response from GPT_3_5 if it's wrong
    def test_execute_sequential_with_llm_filter(self, monkeypatch):
        # define Mock to ensure correct response is returned by the LLM call
        def mock_call(llm_filter, candidate):
            start_time = time.time()
            text_content = candidate._asJSONStr(include_bytes=False)
            response, _, gen_stats = llm_filter.generator.generate(
                context=text_content,
                question=llm_filter.filter.filterCondition,
            )
            response = str("buy" not in candidate.filename)

            # compute whether the record passed the filter or not
            passed_filter = (
                "true" in response.lower()
                if response is not None
                else False
            )

            # NOTE: this will treat the cost of failed LLM invocations as having 0.0 tokens and dollars,
            #       when in reality this is only true if the error in generator.generate() happens before
            #       the invocation of the LLM -- not if it happens after. (If it happens *during* the
            #       invocation, then it's difficult to say what the true cost really should be). I think
            #       the best solution is to place a try-except inside of the DSPyGenerator to still capture
            #       and return the gen_stats if/when there is an error after invocation.
            # create RecordOpStats object
            record_op_stats = RecordOpStats(
                record_uuid=candidate._uuid,
                record_parent_uuid=candidate._parent_uuid,
                record_state=candidate._asDict(include_bytes=False),
                op_id=llm_filter.get_op_id(),
                op_name=llm_filter.op_name(),
                time_per_record=time.time() - start_time,
                cost_per_record=gen_stats.get('cost_per_record', 0.0),
                model_name=llm_filter.model.value,
                filter_str=llm_filter.filter.getFilterStr(),
                total_input_tokens=gen_stats.get('input_tokens', 0.0),
                total_output_tokens=gen_stats.get('output_tokens', 0.0),
                total_input_cost=gen_stats.get('input_cost', 0.0),
                total_output_cost=gen_stats.get('output_cost', 0.0),
                llm_call_duration_secs=gen_stats.get('llm_call_duration_secs', 0.0),
                answer=response,
                passed_filter=passed_filter,
            )

            # set _passed_filter attribute and return
            setattr(candidate, "_passed_filter", passed_filter)

            return [candidate], [record_op_stats]

        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        filter = Filter("The filename does not contain the string 'buy'")
        filterOpClass = ModelSelectionFilterStrategy(available_models=[Model.GPT_3_5], prompt_strategy=PromptStrategy.DSPY_COT_BOOL)[0]

        # apply the monkeypatch for requests.get to mock_get
        monkeypatch.setattr(filterOpClass, "__call__", mock_call)

        filterOp = filterOpClass(inputSchema=File, outputSchema=File, filter=filter, targetCacheId="abc123", shouldProfile=True)
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
        assert dr.filename.endswith("kaminski-v-deleted-items-1902.txt")
        assert getattr(dr, 'contents', None) != None

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
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None


    def test_execute_sequential_with_hardcoded_convert(self):
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
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=1)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename.endswith("buy-r-inbox-628.txt")
        assert getattr(dr, 'contents', None) != None

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
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=float("inf"))

        assert len(output_records) == 6

        expected_filenames = sorted(os.listdir("testdata/enron-eval-tiny"))
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None

    def test_execute_sequential_with_llm_convert(self, email_schema):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        convertOpHardcoded = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
        convertOpClass = BondedQueryConvertStrategy(available_models=[Model.GPT_3_5])[0]

        convertOpLLM = convertOpClass(inputSchema=TextFile, outputSchema=email_schema, targetCacheId="abc123", shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp, convertOpHardcoded, convertOpLLM],
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
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=1)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename.endswith("buy-r-inbox-628.txt")
        assert getattr(dr, 'sender', None) == "sherron.watkins@enron.com"
        assert getattr(dr, 'subject', None) == "RE: portrac"

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
        output_records, plan_stats = simple_execution.execute_sequential(plan, plan_stats, num_samples=float("inf"))

        assert len(output_records) == 6

        expected_filenames = sorted(os.listdir("testdata/enron-eval-tiny"))
        expected_senders = ["david.port@enron.com", "vkaminski@aol.com", "sarah.palmer@enron.com", "gary@cioclub.com", "travis.mccullough@enron.com"]
        expected_subjects = ["RE: NewPower", "Fwd: FYI", "Enron Mentions -- 01/18/02", "Information Security Executive", "Redraft of the Exclusivity Agreement"]
        for dr, expected_filename, expected_sender, expected_subject in zip(output_records, expected_filenames, expected_senders, expected_subjects):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'sender', None) == expected_sender
            assert getattr(dr, 'subject', None) == expected_subject

    # # TODO
    # def test_execute_dag_with_agg(self):
    #     raise Exception("TODO")
    # def test_execute_dag_with_limit(self):
    #     raise Exception("TODO")
