from __future__ import annotations

import os
import time
from typing import Callable

from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from openai import OpenAI
from ragatouille.RAGPretrainedModel import RAGPretrainedModel

from palimpzest.constants import MODEL_CARDS, Model
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.operators.physical import PhysicalOperator


class RetrieveOp(PhysicalOperator):
    def __init__(
        self,
        index: Collection | RAGPretrainedModel,
        search_attr: str,
        output_attr: str,
        search_func: Callable | None,
        k: int,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the RetrieveOp object.
        
        Args:
            index (Collection | RAGPretrainedModel): The PZ index to use for retrieval.
            search_attr (str): The attribute to search on.
            output_attr (str): The attribute to output the search results to.
            search_func (Callable | None): The function to use for searching the index. If None, the default search function will be used.
            k (int): The number of top results to retrieve.
        """
        super().__init__(*args, **kwargs)
        self.index = index
        self.search_attr = search_attr
        self.output_attr = output_attr
        self.search_func = search_func if search_func is not None else self.default_search_func
        self.k = k

    def __str__(self):
        op = super().__str__()
        op += f"    Retrieve: {self.index.__class__.__name__} with top {self.k}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "index": self.index.__class__.__name__,
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "index": self.index,
            "search_func": self.search_func,
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
            **op_params,
        }

        return op_params

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Compute naive cost estimates for the Retrieve operation. These estimates assume
        that the Retrieve (1) has no cost and (2) has perfect quality.
        """
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.01 * self.k,   # estimate 10 ms execution lookup per output
            cost_per_record=0.001 * self.k,  # estimate small marginal cost of lookups 
            quality=1.0,
        )

    def default_search_func(self, index: Collection | RAGPretrainedModel, query: list[str] | list[list[float]], k: int) -> list[str] | list[list[str]]:
        """
        Default search function for the Retrieve operation. This function uses the index to
        retrieve the top-k results for the given query. The query will be a (possibly singleton)
        list of strings or a list of lists of floats (i.e., embeddings). The function will return
        the top-k results per-query in (descending) sorted order. If the input is a singleton list,
        then the output will be a list of strings. If the input is a list of lists, then the output
        will be a list of lists of strings.

        Args:
            index (PZIndex): The index to use for retrieval.
            query (list[str] | list[list[float]]): The query (or queries) to search for.
            k (int): The maximum number of results the retrieve operator will return.

        Returns:
            list[str] | list[list[str]]: The top results in (descending) sorted order per query.
        """
        # check if the input is a singleton list or a list of lists
        is_singleton_list = len(query) == 1

        if isinstance(index, Collection):
            # if the index is a chromadb collection, use the query method
            results = index.query(query, n_results=k)

            # the results["documents"] will be a list[list[str]]; if the input is a singleton list,
            # then we output the list of strings (i.e., the first element of the list), otherwise
            # we output the list of lists
            return results["documents"][0] if is_singleton_list else results["documents"]

        elif isinstance(index, RAGPretrainedModel):
            # if the index is a rag model, use the rag model to get the top k results
            results = index.search(query, k=k)

            # the results will be a list[dict]; if the input is a singleton list, however
            # it will be a list[list[dict]]; if the input is a list of lists
            final_results = []
            if is_singleton_list:
                final_results = [result["content"] for result in results]
            else:
                for query_results in results:
                    final_results.append([result["content"] for result in query_results])

            return final_results

        else:
            raise ValueError("Unsupported index type. Must be either a Collection or RAGPretrainedModel.")

    def _create_record_set(
        self,
        candidate: DataRecord,
        top_k_results: list[str] | list[list[str]] | None,
        generation_stats: GenerationStats,
        total_time: float,
    ) -> DataRecordSet:
        """
        Given an input DataRecord and the top_k_results, construct the resulting RecordSet.
        """
        # create output DataRecord an set the output attribute
        output_dr = DataRecord.from_parent(self.output_schema, parent_record=candidate)
        setattr(output_dr, self.output_attr, top_k_results)

        # get the answer, record_state, and generated_fields
        answer = {self.output_attr: top_k_results}
        record_state = output_dr.to_dict(include_bytes=False)

        # NOTE: right now this should be equivalent to [self.output_attr], but in the future we may
        #       want to support the RetrieveOp generating multiple fields. (Also, the function will
        #       return the full field name (as opposed to the short field name))
        generated_fields = self.get_fields_to_generate(candidate)

        # construct the RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=output_dr.id,
            record_parent_id=output_dr.parent_id,
            record_source_idx=output_dr.source_idx,
            record_state=record_state,
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=total_time,
            cost_per_record=generation_stats.cost_per_record,
            answer=answer,
            input_fields=self.input_schema.field_names(),
            generated_fields=generated_fields,
            fn_call_duration_secs=total_time - generation_stats.llm_call_duration_secs,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        drs = [output_dr]
        record_op_stats_lst = [record_op_stats]

        # construct and return the record set
        return DataRecordSet(drs, record_op_stats_lst)


    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        # check that query is a string or list of strings, otherwise return output with self.output_attr as None
        query = getattr(candidate, self.search_attr)
        query_is_str = isinstance(query, str)
        query_is_list_of_str = isinstance(query, list) and all(isinstance(q, str) for q in query)
        if not query_is_str and not query_is_list_of_str:
            return self._create_record_set(
                candidate=candidate,
                top_k_results=None,
                generation_stats=GenerationStats(),
                total_time=time.time() - start_time,
            )

        # if query is a string, convert it to a list of strings
        if query_is_str:
            query = [query]

        # compute embedding(s) if the index is a chromadb collection
        gen_stats = GenerationStats()
        if isinstance(self.index, Collection):
            uses_openai_embedding_fcn = isinstance(self.index._embedding_function, OpenAIEmbeddingFunction)
            assert uses_openai_embedding_fcn, "ChromaDB index must use OpenAI embedding function; see: https://docs.trychroma.com/integrations/embedding-models/openai"

            model_name = self.index._embedding_function._model_name
            err_msg = f"For Chromadb, we currently only support `text-embedding-3-small`; your index uses: {model_name}"
            assert model_name == Model.TEXT_EMBEDDING_3_SMALL.value, err_msg

            # compute embeddings
            client = OpenAI()
            embed_start_time = time.time()
            response = client.embeddings.create(input=query, model=model_name)
            embed_total_time = time.time() - embed_start_time

            # extract embedding(s)
            query = [item.embedding for item in response.data]

            # compute cost of embedding(s)
            model_card = MODEL_CARDS[model_name]
            total_input_tokens = response.usage.total_tokens
            total_input_cost = model_card["usd_per_input_token"] * total_input_tokens
            gen_stats = GenerationStats(
                model_name=model_name,
                total_input_tokens=total_input_tokens,
                total_output_tokens=0.0,
                total_input_cost=total_input_cost,
                total_output_cost=0.0,
                cost_per_record=total_input_cost,
                llm_call_duration_secs=embed_total_time,
            )

        try:
            top_results = self.search_func(self.index, query, self.k)
        except Exception:
            top_results = ["error-in-retrieve"]
            os.makedirs("retrieve-errors", exist_ok=True)
            ts = time.time()
            with open(f"retrieve-errors/error-{ts}.txt", "w") as f:
                f.write(str(query))

        # TODO: the user is always right! let's drop this post-processing in the future
        # filter top_results for the top_k_results
        top_k_results = []
        if all([isinstance(result, list) for result in top_results]):
            for result in top_results:
                top_k_results.append(result[:self.k])
        else:
            top_k_results = top_results[:self.k]

        if self.verbose:
            print(f"Top {self.k} results: {top_k_results}")

        # construct and return the record set
        return self._create_record_set(
            candidate=candidate,
            top_k_results=top_k_results,
            generation_stats=gen_stats,
            total_time=time.time() - start_time,
        )
