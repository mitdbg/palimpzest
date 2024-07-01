from conftest import *
from palimpzest.corelib import File, TextFile
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.elements import Filter
from palimpzest.execution import *
from palimpzest.operators import *
from palimpzest.planner import PhysicalPlan
from palimpzest.policy import MaxQuality
from palimpzest.strategies import (
    BondedQueryConvertStrategy,
    CodeSynthesisConvertStrategy,
    ModelSelectionFilterStrategy,
)

import os
import time
import pytest

@pytest.mark.parametrize("execution_engine", [SequentialSingleThreadExecution, PipelinedSingleThreadExecution])
class TestSingleThreadExecutionNoCache:

    def test_set_source_dataset_id(self, execution_engine, enron_eval):
        simple_execution = execution_engine()
        simple_execution.set_source_dataset_id(enron_eval)
        assert simple_execution.source_dataset_id == ENRON_EVAL_TINY_DATASET_ID

    # TODO: register dataset in fixture
    def test_execute_plan_simple_scan(self, execution_engine):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        simple_execution = execution_engine(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling a single record
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename.endswith("buy-r-inbox-628.txt")
        assert getattr(dr, 'contents', None) != None

        op_id = scanOp.get_op_id()
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
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) == 6

        expected_filenames = sorted(os.listdir(ENRON_EVAL_TINY_TEST_DATA))
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None


    def test_execute_plan_with_non_llm_filter(self, execution_engine):
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
        simple_execution = execution_engine(num_samples=3, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling three records, with one making it past the filter
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

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
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) == 4

        expected_filenames = [fn for fn in sorted(os.listdir(ENRON_EVAL_TINY_TEST_DATA)) if "buy" not in fn]
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None

    def test_execute_plan_with_llm_filter(self, execution_engine, monkeypatch):
        # define Mock to ensure correct response is returned by the LLM call
        def mock_call(llm_filter, candidate):
            start_time = time.time()
            text_content = candidate._asJSONStr(include_bytes=False)
            response, gen_stats = llm_filter.generator.generate(
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

            # create RecordOpStats object
            record_op_stats = RecordOpStats(
                record_uuid=candidate._uuid,
                record_parent_uuid=candidate._parent_uuid,
                record_state=candidate._asDict(include_bytes=False),
                op_id=llm_filter.get_op_id(),
                op_name=llm_filter.op_name(),
                time_per_record=time.time() - start_time,
                cost_per_record=gen_stats.total_cost,
                model_name=llm_filter.model.value,
                filter_str=llm_filter.filter.getFilterStr(),
                total_input_tokens=gen_stats.total_input_tokens,
                total_output_tokens=gen_stats.total_output_tokens,
                total_input_cost=gen_stats.total_input_cost,
                total_output_cost=gen_stats.total_output_cost,
                llm_call_duration_secs=gen_stats.llm_call_duration_secs,
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
        simple_execution = execution_engine(num_samples=3, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling three records, with one making it past the filter
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

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
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) == 4

        expected_filenames = [fn for fn in sorted(os.listdir(ENRON_EVAL_TINY_TEST_DATA)) if "buy" not in fn]
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None


    def test_execute_plan_with_hardcoded_convert(self, execution_engine):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        convertOp = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp, convertOp],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        simple_execution = execution_engine(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling one record
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

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
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) == 6

        expected_filenames = sorted(os.listdir(ENRON_EVAL_TINY_TEST_DATA))
        for dr, expected_filename in zip(output_records, expected_filenames):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'contents', None) != None

    def test_execute_plan_with_llm_convert(self, execution_engine, email_schema):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        convertOpHardcoded = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
        convertOpClass = BondedQueryConvertStrategy(available_models=[Model.GPT_3_5])[0]

        convertOpLLM = convertOpClass(inputSchema=TextFile, outputSchema=email_schema, targetCacheId="abc123", shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp, convertOpHardcoded, convertOpLLM],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        simple_execution = execution_engine(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling one record
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename.endswith("buy-r-inbox-628.txt")
        assert getattr(dr, 'sender', None) == "sherron.watkins@enron.com"
        assert getattr(dr, 'subject', None) == "RE: portrac"

        for op in plan.operators:
            op_id = op.get_op_id()
            operator_stats = plan_stats.operator_stats[op_id]
            assert operator_stats.total_op_time > 0.0

            if isinstance(op, LLMConvert):
                record_stats = operator_stats.record_op_stats_lst[-1]
                assert record_stats.record_uuid == dr._uuid
                assert record_stats.record_parent_uuid == dr._parent_uuid
                assert record_stats.op_id == op_id
                assert record_stats.op_name == op.op_name()
                assert record_stats.time_per_record > 0.0
                assert record_stats.cost_per_record > 0.0
                assert record_stats.record_state == dr._asDict(include_bytes=False)

        # test full scan
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) == 6

        # TODO: mock out call(s) to LLM
        expected_filenames = sorted(os.listdir(ENRON_EVAL_TINY_TEST_DATA))
        expected_senders = ["sherron.watkins@enron.com", "david.port@enron.com", "vkaminski@aol.com", "sarah.palmer@enron.com", "gary@cioclub.com", "travis.mccullough@enron.com"]
        expected_subjects = ["RE: portrac", "RE: NewPower", "Fwd: FYI", "Enron Mentions -- 01/18/02", "Information Security Executive -092501", "Redraft of the Exclusivity Agreement"]
        for dr, expected_filename, expected_sender, expected_subject in zip(output_records, expected_filenames, expected_senders, expected_subjects):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'sender', None) == expected_sender
            assert getattr(dr, 'subject', None) == expected_subject

    def test_execute_plan_with_code_synth_convert(self, execution_engine, email_schema):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        convertOpHardcoded = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
        convertOpClass = CodeSynthesisConvertStrategy(code_synth_strategy=CodeSynthStrategy.SINGLE)[0]
        convertOpLLM = convertOpClass(inputSchema=TextFile, outputSchema=email_schema, targetCacheId="abc123", shouldProfile=True, cache_across_plans=False)
        plan = PhysicalPlan(
            operators=[scanOp, convertOpHardcoded, convertOpLLM],
            datasetIdentifier=ENRON_EVAL_TINY_DATASET_ID,
        )
        simple_execution = execution_engine(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID

        # test sampling three records, with one making it past the filter
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

        assert len(output_records) == 1

        dr = output_records[0]
        assert dr.filename.endswith("buy-r-inbox-628.txt")
        assert getattr(dr, 'sender', None) == "sherron.watkins@enron.com"
        assert getattr(dr, 'subject', None) == "RE: portrac"

        for op in plan.operators:
            op_id = op.get_op_id()
            operator_stats = plan_stats.operator_stats[op_id]
            assert operator_stats.total_op_time > 0.0

            if isinstance(op, LLMConvert):
                record_stats = operator_stats.record_op_stats_lst[-1]
                assert record_stats.record_uuid == dr._uuid
                assert record_stats.record_parent_uuid == dr._parent_uuid
                assert record_stats.op_id == op_id
                assert record_stats.op_name == op.op_name()
                assert record_stats.time_per_record > 0.0
                assert record_stats.cost_per_record > 0.0
                assert record_stats.record_state == dr._asDict(include_bytes=False)

        # test full scan
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = ENRON_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) == 6

        # TODO: mock out call(s) to synthesized code
        expected_filenames = sorted(os.listdir(ENRON_EVAL_TINY_TEST_DATA))
        expected_senders = ["sherron.watkins@enron.com", "david.port@enron.com", "vkaminski@aol.com", "sarah.palmer@enron.com", "gary@cioclub.com", "travis.mccullough@enron.com"]
        expected_subjects = ["RE: portrac", "RE: NewPower", "Fwd: FYI", "Enron Mentions -- 01/18/02", "Information Security Executive", "Redraft of the Exclusivity Agreement"]
        for dr, expected_filename, expected_sender, expected_subject in zip(output_records, expected_filenames, expected_senders, expected_subjects):
            assert dr.filename.endswith(expected_filename)
            assert getattr(dr, 'sender', None) == expected_sender
            assert getattr(dr, 'subject', None) == expected_subject

    def test_execute_plan_with_image_convert(self, execution_engine, real_estate_listing_datasource, real_estate_listing_files_schema, image_real_estate_listing_schema):
        # register user data source
        pz.DataDirectory().registerUserSource(
            real_estate_listing_datasource(REAL_ESTATE_EVAL_TINY_DATASET_ID, REAL_ESTATE_EVAL_TINY_TEST_DATA), REAL_ESTATE_EVAL_TINY_DATASET_ID
        )

        scanOp = MarshalAndScanDataOp(outputSchema=real_estate_listing_files_schema, dataset_type="dir", shouldProfile=True)
        convertOpClass = BondedQueryConvertStrategy(available_models=[Model.GPT_3_5])[0]
        convertOpLLM = convertOpClass(inputSchema=real_estate_listing_files_schema, outputSchema=image_real_estate_listing_schema, targetCacheId="abc123", shouldProfile=True, image_conversion=True)
        plan = PhysicalPlan(
            operators=[scanOp, convertOpLLM],
            datasetIdentifier=REAL_ESTATE_EVAL_TINY_DATASET_ID,
        )
        simple_execution = execution_engine(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = REAL_ESTATE_EVAL_TINY_DATASET_ID

        # test sampling three records, with one making it past the filter
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

        assert len(output_records) == 1
        dr = output_records[0]
        assert dr.listing == "listing1"
        assert getattr(dr, 'is_modern_and_attractive', None) == True
        assert getattr(dr, 'has_natural_sunlight', None) == True

        for op in plan.operators:
            op_id = op.get_op_id()
            operator_stats = plan_stats.operator_stats[op_id]
            assert operator_stats.total_op_time > 0.0

            if isinstance(op, LLMConvert):
                record_stats = operator_stats.record_op_stats_lst[-1]
                assert record_stats.record_uuid == dr._uuid
                assert record_stats.record_parent_uuid == dr._parent_uuid
                assert record_stats.op_id == op_id
                assert record_stats.op_name == op.op_name()
                assert record_stats.time_per_record > 0.0
                assert record_stats.cost_per_record > 0.0
                assert record_stats.record_state == dr._asDict(include_bytes=False)

        # test full scan
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = REAL_ESTATE_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) == 3

        # TODO: mock out call(s) to synthesized code
        expected_listings = sorted(os.listdir(REAL_ESTATE_EVAL_TINY_TEST_DATA))
        expected_moderns = [True, False, False]
        expected_sunlights = [True, True, False]
        for dr, expected_listing, expected_modern, expected_sunlight in zip(output_records, expected_listings, expected_moderns, expected_sunlights):
            assert dr.listing == expected_listing
            assert getattr(dr, 'is_modern_and_attractive', None) == expected_modern
            assert getattr(dr, 'has_natural_sunlight', None) == expected_sunlight

    def test_execute_plan_with_one_to_many(self, execution_engine, case_data_schema):
        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
        convertToXLSFileOp = ConvertFileToXLS(inputSchema=File, outputSchema=XLSFile, shouldProfile=True)
        convertToTableOp = ConvertXLSToTable(inputSchema=XLSFile, outputSchema=Table, cardinality=Cardinality.ONE_TO_MANY, shouldProfile=True)
        filter = Filter("The rows of the table contain the patient age")
        filterOpClass = ModelSelectionFilterStrategy(available_models=[Model.GPT_3_5], prompt_strategy=PromptStrategy.DSPY_COT_BOOL)[0]
        filterOpLLM = filterOpClass(inputSchema=Table, outputSchema=Table, filter=filter, targetCacheId="abc123", shouldProfile=True)
        convertOpClass = BondedQueryConvertStrategy(available_models=[Model.GPT_3_5])[0]
        convertOpLLM = convertOpClass(inputSchema=Table, outputSchema=case_data_schema, cardinality=Cardinality.ONE_TO_MANY, desc="The patient data in the table", targetCacheId="abc123", shouldProfile=True)
        plan = PhysicalPlan(
            operators=[scanOp, convertToXLSFileOp, convertToTableOp, filterOpLLM, convertOpLLM],
            datasetIdentifier=BIOFABRIC_EVAL_TINY_DATASET_ID,
        )
        simple_execution = execution_engine(num_samples=1, nocache=True)

        # set state which is computed in execute(); should try to remove this side-effect from the code
        simple_execution.source_dataset_id = BIOFABRIC_EVAL_TINY_DATASET_ID

        # test sampling one record
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.SENTINEL)

        assert len(output_records) > 1

        dr = output_records[0]
        assert dr.filename.endswith("dou_mmc1.xlsx")
        for field in case_data_schema.fieldNames():
            assert hasattr(dr, field)

        # test full scan
        simple_execution = execution_engine(nocache=True)
        simple_execution.source_dataset_id = BIOFABRIC_EVAL_TINY_DATASET_ID
        output_records, plan_stats = simple_execution.execute_plan(plan, plan_type=PlanType.FINAL)

        assert len(output_records) > 3

        # TODO: mock out call(s) to LLM
        expected_filenames = sorted(os.listdir(BIOFABRIC_EVAL_TINY_TEST_DATA))
        for dr in output_records:
            assert any([dr.filename.endswith(filename) for filename in expected_filenames])
            for field in case_data_schema.fieldNames():
                assert hasattr(dr, field)

    # # TODO
    # def test_execute_plan_with_agg(self):
    #     raise Exception("TODO")
    # def test_execute_plan_with_limit(self):
    #     raise Exception("TODO")
