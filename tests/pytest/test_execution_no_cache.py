import time

import pytest
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.query.execution.nosentinel_execution import (
    NoSentinelPipelinedParallelExecution,
    NoSentinelSequentialSingleThreadExecution,
)
from palimpzest.query.operators.code_synthesis_convert import CodeSynthesisConvert
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter


@pytest.mark.parametrize(
    argnames=("execution_engine",),
    argvalues=[
        pytest.param(NoSentinelSequentialSingleThreadExecution, id="seq-single-thread"),
        pytest.param(NoSentinelPipelinedParallelExecution, id="parallel"),
    ]
)
class TestParallelExecutionNoCache:
    # the number of sentinel samples to be drawn for each execution of test_execute_sentinel_plan
    TEST_SENTINEL_NUM_SAMPLES: int = 3

    # TODO: needs to be updated to reflect changes to SentinelPlan
    # @pytest.mark.parametrize(
    #     argnames=("dataset", "physical_plan"),
    #     argvalues=[
    #         pytest.param("enron-eval-tiny", "scan-only", id="scan-only"),
    #         pytest.param("enron-eval-tiny", "non-llm-filter", id="non-llm-filter"),
    #     ],
    #     indirect=True,
    # )
    # def test_execute_sentinel_plan(self, execution_engine, dataset, physical_plan):
    #     # fetch datasource
    #     datasource = DataDirectory().get_registered_dataset(dataset)

    #     # create execution instance
    #     execution = execution_engine(datasource=datasource, num_samples=self.TEST_SENTINEL_NUM_SAMPLES, nocache=True)

    #     # execute the plan
    #     _, plan_stats = execution.execute_plan(physical_plan, num_samples=self.TEST_SENTINEL_NUM_SAMPLES)

    #     # NOTE: when we enable multi-source plans; this will need to be updated
    #     # get the stats from the source operator
    #     source_op_stats = list(plan_stats.operator_stats.values())[0]

    #     # test that we only executed plan on num_samples records
    #     assert len(source_op_stats.record_op_stats_lst) == self.TEST_SENTINEL_NUM_SAMPLES

    @pytest.mark.parametrize(
        argnames=("dataset", "physical_plan", "expected_records", "side_effect"),
        argvalues=[
            pytest.param("enron-eval-tiny", "scan-only", "enron-all-records", None, id="scan-only"),
            pytest.param("enron-eval-tiny", "non-llm-filter", "enron-filtered-records", None, id="non-llm-filter"),
            pytest.param("enron-eval-tiny", "llm-filter", "enron-filtered-records", "enron-filter", id="llm-filter"),
            pytest.param(
                "enron-eval-tiny", "bonded-llm-convert", "enron-all-records", "enron-convert", id="bonded-llm-convert"
            ),
            pytest.param(
                "enron-eval-tiny", "code-synth-convert", "enron-all-records", "enron-convert", id="code-synth-convert"
            ),
            pytest.param(
                "enron-eval-tiny",
                "rag-convert",
                "enron-all-records",
                "enron-convert",
                id="rag-convert",
            ),
            # pytest.param(
            #     "enron-eval-tiny",
            #     "token-reduction-convert",
            #     "enron-all-records",
            #     "enron-convert",
            #     id="token-reduction-convert",
            # ),
            pytest.param(
                "real-estate-eval-tiny",
                "image-convert",
                "real-estate-all-records",
                "real-estate-convert",
                id="image-convert",
            ),
            pytest.param(
                "real-estate-eval-tiny",
                "one-to-many-convert",
                "real-estate-one-to-many-records",
                "real-estate-one-to-many-convert",
                id="one-to-many-convert",
            ),
        ],
        indirect=True,
    )
    def test_execute_full_plan(self, mocker, execution_engine, dataset, physical_plan, expected_records, side_effect):
        """
        This test executes the given
        """
        start_time = time.time()

        # fetch datasource
        datasource = DataDirectory().get_registered_dataset(dataset)

        # create execution instance
        execution = execution_engine(datasource=datasource, nocache=True)

        # manually set source_dataset_id
        execution.source_dataset_id = dataset

        # mock out calls to generators used by the plans which parameterize this test
        mocker.patch.object(LLMFilter, "__call__", side_effect=side_effect)
        mocker.patch.object(LLMConvert, "__call__", side_effect=side_effect)
        mocker.patch.object(CodeSynthesisConvert, "__call__", side_effect=side_effect)

        # execute the plan
        output_records, plan_stats = execution.execute_plan(physical_plan)
        plan_stats.finalize(time.time() - start_time)

        # check that we get the expected set of output records
        def get_id(record):
            return record.listing if "real-estate" in dataset else record.filename

        assert len(output_records) == len(expected_records)
        assert sorted(map(get_id, output_records)) == sorted(map(get_id, expected_records))

        # sanity check plan stats
        assert plan_stats.total_plan_time > 0.0

        # if the plan used (mocked) calls to an LLM, assert that the plan cost money
        if side_effect is not None:
            assert plan_stats.total_plan_cost > 0.0
        else:
            assert plan_stats.total_plan_cost == 0.0
