from __future__ import annotations

import os
import time
from typing import Callable

from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from palimpzest.constants import MODEL_CARDS, Model
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import Schema
from palimpzest.query.operators.physical import PhysicalOperator


class RetrieveOp(PhysicalOperator):
    def __init__(
        self,
        index: Collection,
        search_attr: str,
        output_attrs: list[dict] | type[Schema],
        search_func: Callable | None,
        k: int,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the RetrieveOp object.
        
        Args:
            index (Collection): The PZ index to use for retrieval.
            search_attr (str): The attribute to search on.
            output_attrs (list[dict]): The output fields containing the results of the search.
            search_func (Callable | None): The function to use for searching the index. If None, the default search function will be used.
            k (int): The number of top results to retrieve.
        """
        super().__init__(*args, **kwargs)

        # extract the field names from the output_attrs
        if isinstance(output_attrs, Schema):
            self.output_field_names = output_attrs.field_names()
        elif isinstance(output_attrs, list):
            self.output_field_names = [attr["name"] for attr in output_attrs]
        else:
            raise ValueError("`output_attrs` must be a list of dicts or a Schema object.")

        if len(self.output_field_names) != 1 and search_func is None:
            raise ValueError("If `search_func` is None, `output_attrs` must have a single field.")

        self.index = index
        self.search_attr = search_attr
        self.output_attrs = output_attrs
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
            "output_attrs": self.output_attrs,
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
            "output_attrs": self.output_attrs,
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

    def default_search_func(self, index: Collection, query: list[str] | list[list[float]], k: int) -> list[str] | list[list[str]]:
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
            final_results = results["documents"][0] if is_singleton_list else results["documents"]

            # NOTE: self.output_field_names must be a singleton for default_search_func to be used
            return {self.output_field_names[0]: final_results}

        else:
            raise ValueError("Unsupported index type. Must be either a Collection.")

    def _create_record_set(
        self,
        candidate: DataRecord,
        top_k_results: dict[str, list[str] | list[list[str]]] | None,
        generation_stats: GenerationStats,
        total_time: float,
    ) -> DataRecordSet:
        """
        Given an input DataRecord and the top_k_results, construct the resulting RecordSet.
        """
        # create output DataRecord an set the output attribute
        output_dr, answer = DataRecord.from_parent(self.output_schema, parent_record=candidate), {}
        for output_field_name in self.output_field_names:
            top_k_attr_results = None if top_k_results is None else top_k_results[output_field_name]
            setattr(output_dr, output_field_name, top_k_attr_results)
            answer[output_field_name] = top_k_attr_results

        # get the record_state and generated fields
        record_state = output_dr.to_dict(include_bytes=False)

        # NOTE: this should be equivalent to self.get_fields_to_generate()
        generated_fields = self.output_field_names

        # construct the RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=output_dr.id,
            record_parent_id=output_dr.parent_id,
            record_source_idx=output_dr.source_idx,
            record_state=record_state,
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=total_time,
            cost_per_record=generation_stats.cost_per_record,
            answer=answer,
            input_fields=self.input_schema.field_names(),
            generated_fields=generated_fields,
            fn_call_duration_secs=total_time - generation_stats.llm_call_duration_secs,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        drs = [output_dr]
        record_op_stats_lst = [record_op_stats]

        # construct and return the record set
        return DataRecordSet(drs, record_op_stats_lst)


    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        # check that query is a string or list of strings, otherwise return output with self.output_field_names set to None
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

        # compute input/query embedding(s) if the index is a chromadb collection
        inputs, gen_stats = None, GenerationStats()
        if isinstance(self.index, Collection):
            uses_openai_embedding_fcn = isinstance(self.index._embedding_function, OpenAIEmbeddingFunction)
            uses_sentence_transformer_embedding_fcn = isinstance(self.index._embedding_function, SentenceTransformerEmbeddingFunction)
            error_msg = "ChromaDB index must use OpenAI or SentenceTransformer embedding function; see: https://docs.trychroma.com/integrations/embedding-models/openai"
            assert uses_openai_embedding_fcn or uses_sentence_transformer_embedding_fcn, error_msg

            model_name = self.index._embedding_function._model_name if uses_openai_embedding_fcn else "clip-ViT-B-32"
            err_msg = f"For Chromadb, we currently only support `text-embedding-3-small` and `clip-ViT-B-32`; your index uses: {model_name}"
            embedding_model_names = [model.value for model in Model if model.is_embedding_model()]
            assert model_name in embedding_model_names, err_msg

            # compute embeddings
            try:
                embed_start_time = time.time()
                total_input_tokens = 0.0
                if uses_openai_embedding_fcn:
                    client = OpenAI()
                    response = client.embeddings.create(input=query, model=model_name)
                    total_input_tokens = response.usage.total_tokens
                    inputs = [item.embedding for item in response.data]

                elif uses_sentence_transformer_embedding_fcn:
                    model = SentenceTransformer(model_name)
                    inputs = model.encode(query)

                embed_total_time = time.time() - embed_start_time

                # compute cost of embedding(s)
                model_card = MODEL_CARDS[model_name]
                total_input_cost = model_card["usd_per_input_token"] * total_input_tokens
                gen_stats = GenerationStats(
                    model_name=model_name,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=0.0,
                    total_input_cost=total_input_cost,
                    total_output_cost=0.0,
                    cost_per_record=total_input_cost,
                    llm_call_duration_secs=embed_total_time,
                    total_llm_calls=1,
                    total_embedding_llm_calls=len(query),
                )
            except Exception:
                query = None

        # in the default case, pass string inputs rather than embeddings
        if inputs is None:
            inputs = query

        try:
            assert inputs is not None, "Error: inputs is None (likely because embedding generation failed)"
            top_results = self.search_func(self.index, inputs, self.k)

        except Exception:
            top_results = ["error-in-retrieve"]
            os.makedirs("retrieve-errors", exist_ok=True)
            ts = time.time()
            with open(f"retrieve-errors/error-{ts}.txt", "w") as f:
                f.write(str(query))

        # TODO: the user is always right! let's drop this post-processing in the future
        # filter top_results for the top_k_results
        top_k_results = {output_field_name: [] for output_field_name in self.output_field_names}
        for output_field_name in self.output_field_names:
            if output_field_name in top_results:
                if all([isinstance(result, list) for result in top_results[output_field_name]]):
                    for result in top_results[output_field_name]:
                        top_k_results[output_field_name].append(result[:self.k])
                else:
                    top_k_results[output_field_name] = top_results[output_field_name][:self.k]
            else:
                top_k_results[output_field_name] = []

        if self.verbose:
            print(f"Top {self.k} results: {top_k_results}")

        # construct and return the record set
        return self._create_record_set(
            candidate=candidate,
            top_k_results=top_k_results,
            generation_stats=gen_stats,
            total_time=time.time() - start_time,
        )
