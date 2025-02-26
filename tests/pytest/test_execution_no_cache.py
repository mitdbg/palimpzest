import pytest

from palimpzest.policy import MaxQuality
from palimpzest.query.operators.code_synthesis_convert import CodeSynthesisConvert
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.rag_convert import RAGConvert
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.nosentinel_processor import (
    NoSentinelParallelProcessor,
    NoSentinelSequentialSingleThreadProcessor,
)


@pytest.mark.parametrize(
    argnames=("query_processor",),
    argvalues=[
        pytest.param(NoSentinelSequentialSingleThreadProcessor, id="seq-single-thread"),
        pytest.param(NoSentinelParallelProcessor, id="parallel"),
    ]
)
class TestParallelExecutionNoCache:

    @pytest.mark.parametrize(
        argnames=("datareader", "physical_plan", "expected_records", "side_effect"),
        argvalues=[
            pytest.param("enron-eval-tiny", "scan-only", "enron-all-records", None, id="scan-only"),
            pytest.param("enron-eval-tiny", "non-llm-filter", "enron-filtered-records", None, id="non-llm-filter"),
            pytest.param("enron-eval-tiny", "llm-filter", "enron-filtered-records", "enron-filter", id="llm-filter"),
            pytest.param(
                "enron-eval-tiny", "bonded-llm-convert", "enron-all-records", "enron-convert", id="bonded-llm-convert"
            ),
            pytest.param(
                "enron-eval-tiny", 
                "code-synth-convert",
                "enron-all-records",
                "enron-convert",
                id="code-synth-convert"
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
    def test_execute_full_plan(self, mocker, query_processor, datareader, physical_plan, expected_records, side_effect):
        """
        This test executes the given
        """
        # NOTE: supplying datareader in place of dataset is a bit of a band-aid but it works
        # create processor
        processor = query_processor(
            dataset=datareader,
            config=QueryProcessorConfig(),
            optimizer=Optimizer(policy=MaxQuality(), cost_model=CostModel()),
        )

        # mock out calls to generators used by the plans which parameterize this test
        mocker.patch.object(LLMFilter, "filter", side_effect=side_effect)
        mocker.patch.object(LLMConvertBonded, "convert", side_effect=side_effect)
        mocker.patch.object(CodeSynthesisConvert, "convert", side_effect=side_effect)
        mocker.patch.object(RAGConvert, "convert", side_effect=side_effect)

        # execute the plan
        output_records, plan_stats = processor.execute_plan(physical_plan)     

        # check that we get the expected set of output records
        def get_id(record):
            return record.listing if "RealEstate" in datareader.__class__.__name__ else record.filename

        assert len(output_records) == len(expected_records)
        assert sorted(map(get_id, output_records)) == sorted(map(get_id, expected_records))

        # sanity check plan stats
        assert plan_stats.total_plan_time > 0.0

        # if the plan used (mocked) calls to an LLM, assert that the plan cost money
        if side_effect is not None:
            assert plan_stats.total_plan_cost > 0.0
        else:
            assert plan_stats.total_plan_cost == 0.0
