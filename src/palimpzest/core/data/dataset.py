from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, Callable

try:
    from chromadb.api.models.Collection import Collection
except ImportError:
    class Collection:
        pass

from pydantic import BaseModel

from palimpzest.constants import AggFunc, Cardinality
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.lib.schemas import create_schema_from_fields, project, relax_schema, union_schemas
from palimpzest.policy import construct_policy_from_kwargs
from palimpzest.query.operators.logical import (
    Aggregate,
    ConvertScan,
    Distinct,
    FilteredScan,
    GroupByAggregate,
    JoinOp,
    LimitScan,
    LogicalOperator,
    Project,
    TopKScan,
)
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.hash_helpers import hash_for_serialized_dict
from palimpzest.validator.validator import Validator


# TODO?: remove `schema` from `Dataset` and access it from `operator`?
# - Q: how do you handle datasets with multiple sources?
#    - for joins the operator should have the union'ed schema
#    - but for Contexts it may be trickier
class Dataset:
    """
    A `Dataset` represents a collection of structured or unstructured data that can be processed and
    transformed. Each `Dataset` is either a "root" `Dataset` (which yields data items) or it is the
    result of performing data processing operation(s) on root `Dataset(s)`.

    Users can perform computations on a `Dataset` in a lazy or eager fashion. Applying functions
    such as `sem_filter`, `sem_map`, `sem_join`, `sem_agg`, etc. will lazily create a new `Dataset`.
    Users can invoke the `run()` method to execute the computation and retrieve a materialized `Dataset`.
    Materialized `Dataset`s can be processed further, or their results can be retrieved using `.get()`.
    
    A root `Dataset` must subclass at least one of `pz.IterDataset`, `pz.IndexDataset`, or `pz.Context`.
    Each of these classes supports a different access pattern:

        - `pz.IterDataset`: supports accessing data via iteration
            - Ex: iterating over a list of PDFs
            - Ex: iterating over rows in a DataFrame
        - `pz.IndexDataset`: supports accessing data via point lookups / queries
            - Ex: querying a vector database
            - Ex: querying a SQL database
        - `pz.Context`: supports accessing data with an agent
            - Ex: processing a set of CSV files with a data science agent
            - Ex: processing time series data with a data cleaning agent

    A root `Dataset` may subclass more than one of the aforementioned classes. For example, the root
    `Dataset` for a list of files may inherit from `pz.IterDataset` and `pz.IndexDataset` to support
    iterating over the files and performing point lookups for individual files.

    For details on how to create your own root `Dataset`, please see: TODO
    """
    def __init__(
            self,
            sources: list[Dataset] | Dataset | None,
            operator: LogicalOperator,
            schema: type[BaseModel] | None = None,
            id: str | None = None,
        ) -> None:
        """
        Initialize a `Dataset` with one or more `sources` and the operator that is being applied.
        Root `Datasets` subclass `pz.IterDataset`, `pz.IndexDataset`, and/or `pz.Context` and use
        their own constructors.

        Args:
            sources (`list[Dataset] | Dataset`): The (list of) `Dataset(s)` which are input(s) to
                the operator used to compute this `Dataset`.
            operator (`LogicalOperator`): The `LogicalOperator` used to compute this `Dataset`.
            schema (type[`BaseModel`] | None): The schema of this `Dataset`.
            id (str | None): an identifier for this `Dataset` provided by the user

        Raises:
            ValueError: if `sources` is not a `Dataset` or list of `Datasets`
        """
        # set sources
        self._sources = []
        if isinstance(sources, list):
            self._sources = sources
        elif isinstance(sources, Dataset):
            self._sources = [sources]
        elif sources is not None:
            raise ValueError("Dataset sources must be another Dataset or a list of Datasets. For root Datasets, you must subclass pz.IterDataset, pz.IndexDataset, or pz.Context.")

        # set the logical operator and schema
        self._operator: LogicalOperator = operator
        self._schema = schema

        # compute the dataset id
        self._id = self._compute_dataset_id() if id is None else id

    @property
    def id(self) -> str:
        """The string identifier for this `Dataset`"""
        return self._id

    @property
    def schema(self) -> type[BaseModel]:
        """The Pydantic model defining the schema of this `Dataset`"""
        return self._schema

    @property
    def is_root(self) -> bool:
        return len(self._sources) == 0

    def __str__(self) -> str:
        return f"Dataset(schema={self._schema}, id={self._id}, op_id={self._operator.get_logical_op_id()})"

    def __iter__(self) -> Iterator[Dataset]:
        for source in self._sources:
            yield from source
        yield self

    def _compute_dataset_id(self) -> str:
        """
        Compute the identifier for this `Dataset`. The ID is uniquely defined by the operation(s)
        applied to the `Dataset's` sources.
        """
        return hash_for_serialized_dict({
            "source_ids": [source.id for source in self._sources],
            "logical_op_id": self._operator.get_logical_op_id(),
        })

    def _set_root_datasets(self, new_root_datasets: dict[str, Dataset]) -> None:
        """
        Update the root dataset(s) for this dataset with the `new_root_datasets`. This is used during
        optimization to reuse the same physical plan while running it on a train dataset.

        Args:
            new_root_datasets (dict[str, Dataset]): the new root datasets for this dataset.
        """
        new_sources = []
        for old_source in self._sources:
            if old_source.id in new_root_datasets:
                new_sources.append(new_root_datasets[old_source.id])
            else:
                old_source._set_root_datasets(new_root_datasets)
                new_sources.append(old_source)
        self._sources = new_sources

    # TODO: the entire way (unique) logical op ids are computed and stored needs to be rethought
    def _generate_unique_logical_op_ids(self, topo_idx: int | None = None) -> None:
        """
        Generate unique operation IDs for all operators in this dataset and its sources.
        This is used to ensure that each operator can be uniquely identified during execution.
        """
        # generate the unique op ids for all sources' operators
        for source in self._sources:
            topo_idx = source._generate_unique_logical_op_ids(topo_idx)
            topo_idx += 1

        # if topo_idx is None, this is the first call, so we initialize it to 0
        if topo_idx is None:
            topo_idx = 0

        # compute this operator's unique operator id
        this_unique_logical_op_id = f"{topo_idx}-{self._operator.get_logical_op_id()}"

        # update the unique logical op id for this operator
        self._operator.set_unique_logical_op_id(this_unique_logical_op_id)

        # return the current unique full_op_id for this operator
        return topo_idx

    # TODO
    def _resolve_depends_on(self, depends_on: list[str]) -> list[str]:
        """
        TODO: resolve the `depends_on` strings to their full field names ({Dataset.id}.{field_name}).
        """
        return []

    def _get_root_datasets(self) -> dict[str, Dataset]:
        """Return a mapping from the id --> Dataset for all root datasets."""
        if self.is_root:
            return {self.id: self}

        root_datasets = {}
        for source in self._sources:
            child_root_datasets = source._get_root_datasets()
            root_datasets = {**root_datasets, **child_root_datasets}

        return root_datasets

    def relax_types(self) -> None:
        """
        Relax the types in this Dataset's schema and all upstream Datasets' schemas to be more permissive.
        """
        # relax the types in this dataset's schema
        self._schema = relax_schema(self._schema)

        # relax the types in dataset's operator's input and output schemas
        self._operator.input_schema = None if self._operator.input_schema is None else relax_schema(self._operator.input_schema)
        self._operator.output_schema = relax_schema(self._operator.output_schema)

        # recursively relax the types in all upstream datasets
        for source in self._sources:
            source.relax_types()

    def get_upstream_datasets(self) -> list[Dataset]:
        """
        Get the list of all upstream datasets that are sources to this dataset.
        """
        # recursively get the upstream datasets
        upstream = []
        for source in self._sources:
            upstream.extend(source.get_upstream_datasets())
            upstream.append(source)
        return upstream

    def get_limit(self) -> int | None:
        """Get the limit applied to this Dataset, if any."""
        if isinstance(self._operator, LimitScan):
            return self._operator.limit

        source_limits = []
        for source in self._sources:
            source_limit = source.get_limit()
            if source_limit is not None:
                source_limits.append(source_limit)

        if len(source_limits) == 0:
            return None

        return min([limit for limit in source_limits if limit is not None])

    def copy(self):
        return Dataset(
            sources=[source.copy() for source in self._sources],
            operator=self._operator.copy(),
            schema=self._schema,
            id=self.id,
        )

    def join(self, other: Dataset, on: str | list[str], how: str = "inner") -> Dataset:
        """
        Perform the specified join on the specified (list of) column(s)
        """
        # enforce type for on
        if isinstance(on, str):
            on = [on]

        # construct new output schema
        combined_schema = union_schemas([self.schema, other.schema], join=True, on=on)

        # construct logical operator
        operator = JoinOp(
            input_schema=combined_schema,
            output_schema=combined_schema,
            condition="",
            on=on,
            how=how,
            depends_on=on,
        )

        return Dataset(sources=[self, other], operator=operator, schema=combined_schema)

    def sem_join(self, other: Dataset, condition: str, desc: str | None = None, depends_on: str | list[str] | None = None, how: str = "inner") -> Dataset:
        """
        Perform a semantic (inner) join on the specified join predicate
        """
        # enforce type for depends_on
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # construct new output schema
        combined_schema = union_schemas([self.schema, other.schema], join=True)

        # construct logical operator
        operator = JoinOp(
            input_schema=combined_schema,
            output_schema=combined_schema,
            condition=condition,
            how=how,
            desc=desc,
            depends_on=depends_on,
        )

        return Dataset(sources=[self, other], operator=operator, schema=combined_schema)

    def filter(
        self,
        filter: Callable,
        depends_on: str | list[str] | None = None,
    ) -> Dataset:
        """Add a user defined function as a filter to the Set. This filter will possibly restrict the items that are returned later."""
        # construct Filter object
        f = None
        if callable(filter):
            f = Filter(filter_fn=filter)
        else:
            error_str = f"Only support callable for filter, currently got {type(filter)}"
            if isinstance(filter, str):
                error_str += ". Consider using sem_filter() for semantic filters."
            raise Exception(error_str)

        # enforce type for depends_on
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # construct logical operator
        operator = FilteredScan(input_schema=self.schema, output_schema=self.schema, filter=f, depends_on=depends_on)

        return Dataset(sources=[self], operator=operator, schema=self.schema)

    def sem_filter(
        self,
        filter: str,
        desc: str | None = None,
        depends_on: str | list[str] | None = None,
    ) -> Dataset:
        """Add a natural language description of a filter to the Set. This filter will possibly restrict the items that are returned later."""
        # construct Filter object
        f = None
        if isinstance(filter, str):
            f = Filter(filter)
        else:
            raise Exception("sem_filter() only supports `str` input for _filter.", type(filter))

        # enforce type for depends_on
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # construct logical operator
        operator = FilteredScan(input_schema=self.schema, output_schema=self.schema, filter=f, desc=desc, depends_on=depends_on)

        return Dataset(sources=[self], operator=operator, schema=self.schema)

    def _sem_map(self, cols: list[dict] | type[BaseModel] | None,
                 cardinality: Cardinality,
                 desc: str | None = None,
                 depends_on: str | list[str] | None = None) -> Dataset:
        """Execute the semantic map operation with the appropriate cardinality."""
        # construct new output schema
        new_output_schema = None
        if cols is None:
            new_output_schema = self.schema
        elif isinstance(cols, list):
            cols = create_schema_from_fields(cols)
            new_output_schema = union_schemas([self.schema, cols])
        elif issubclass(cols, BaseModel):
            new_output_schema = union_schemas([self.schema, cols])
        else:
            raise ValueError("`cols` must be a list of dictionaries or a BaseModel.")

        # enforce type for depends_on
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # construct logical operator
        operator = ConvertScan(
            input_schema=self.schema,
            output_schema=new_output_schema,
            cardinality=cardinality,
            udf=None,
            desc=desc,
            depends_on=depends_on,
        )

        return Dataset(sources=[self], operator=operator, schema=new_output_schema)

    def sem_add_columns(self, cols: list[dict] | type[BaseModel],
                        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
                        desc: str | None = None,
                        depends_on: str | list[str] | None = None) -> Dataset:
        """
        NOTE: we are renaming this function to `sem_map` and deprecating `sem_add_columns` in the next
        release of PZ. To update your code, simply change your calls from `.sem_add_columns(...)` to `.sem_map(...)`.
        The function arguments will stay the same.

        Add new columns by specifying the column names, descriptions, and types.
        The column will be computed during the execution of the Dataset.
        Example:
            sem_add_columns(
                [{'name': 'greeting', 'desc': 'The greeting message', 'type': str},
                 {'name': 'age', 'desc': 'The age of the person', 'type': int},
                 {'name': 'full_name', 'desc': 'The name of the person', 'type': str}]
            )
        """
        # issue deprecation warning
        warnings.warn(
            "we are renaming this function to `sem_map` and deprecating `sem_add_columns` in the next"
            " release of PZ. To update your code, simply change your calls from `.sem_add_columns(...)`"
            " to `.sem_map(...)`. The function arguments will stay the same.",
            DeprecationWarning,
            stacklevel=2
        )

        return self._sem_map(cols, cardinality, desc, depends_on)

    def sem_map(self, cols: list[dict] | type[BaseModel], desc: str | None = None, depends_on: str | list[str] | None = None) -> Dataset:
        """
        Compute new field(s) by specifying their names, descriptions, and types. For each input there will
        be one output. The field(s) will be computed during the execution of the Dataset.

        Example:
            sem_map(
                [{'name': 'greeting', 'desc': 'The greeting message', 'type': str},
                 {'name': 'age', 'desc': 'The age of the person', 'type': int},
                 {'name': 'full_name', 'desc': 'The name of the person', 'type': str}]
            )
        """
        return self._sem_map(cols, Cardinality.ONE_TO_ONE, desc, depends_on)

    def sem_flat_map(self, cols: list[dict] | type[BaseModel], desc: str | None = None, depends_on: str | list[str] | None = None) -> Dataset:
        """
        Compute new field(s) by specifying their names, descriptions, and types. For each input there will
        be one or more output(s). The field(s) will be computed during the execution of the Dataset.

        Example:
            sem_flat_map(
                cols=[
                    {'name': 'author_name', 'description': 'The name of the author', 'type': str},
                    {'name': 'institution', 'description': 'The institution of the author', 'type': str},
                    {'name': 'email', 'description': 'The author's email', 'type': str},
                ]
            )
        """
        return self._sem_map(cols, Cardinality.ONE_TO_MANY, desc, depends_on)

    def _map(self, udf: Callable,
            cols: list[dict] | type[BaseModel] | None,
            cardinality: Cardinality,
            depends_on: str | list[str] | None = None) -> Dataset:
        """Execute the map operation with the appropriate cardinality."""
        # construct new output schema
        new_output_schema = None
        if cols is None:
            new_output_schema = self.schema
        elif isinstance(cols, list):
            cols = create_schema_from_fields(cols)
            new_output_schema = union_schemas([self.schema, cols])
        elif issubclass(cols, BaseModel):
            new_output_schema = union_schemas([self.schema, cols])
        else:
            raise ValueError("`cols` must be a list of dictionaries, a BaseModel, or None.")

        # enforce type for depends_on
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # construct logical operator
        operator = ConvertScan(
            input_schema=self.schema,
            output_schema=new_output_schema,
            cardinality=cardinality,
            udf=udf,
            depends_on=depends_on,
        )

        return Dataset(sources=[self], operator=operator, schema=new_output_schema)

    def add_columns(self, udf: Callable,
                    cols: list[dict] | type[BaseModel] | None,
                    cardinality: Cardinality = Cardinality.ONE_TO_ONE,
                    depends_on: str | list[str] | None = None) -> Dataset:
        """
        NOTE: we are renaming this function to `map` and deprecating `add_columns` in the next
        release of PZ. To update your code, simply change your calls from `.add_columns(...)` to `.map(...)`.
        The function arguments will stay the same.

        Compute new fields (or update existing ones) with a UDF. For each input, this function will compute one output.

        Set `cols=None` if your add_columns operation is not computing any new fields.

        Examples:
            add_columns(
                udf=compute_personal_greeting,
                cols=[
                    {'name': 'greeting', 'description': 'The greeting message', 'type': str},
                    {'name': 'age', 'description': 'The age of the person', 'type': int},
                    {'name': 'full_name', 'description': 'The name of the person', 'type': str},
                ]
            )
        """
        # issue deprecation warning
        warnings.warn(
            "we are renaming this function to `map` and deprecating `add_columns` in the next"
            " release of PZ. To update your code, simply change your calls from `.add_columns(...)`"
            " to `.map(...)`. The function arguments will stay the same.",
            DeprecationWarning,
            stacklevel=2
        )

        # sanity check inputs
        if udf is None:
            raise ValueError("`udf` must be provided for add_columns.")

        return self._map(udf, cols, cardinality, depends_on)

    def map(self, udf: Callable,
            cols: list[dict] | type[BaseModel] | None,
            depends_on: str | list[str] | None = None) -> Dataset:
        """
        Compute new fields (or update existing ones) with a UDF. For each input, this function will compute one output.

        Set `cols=None` if your map is not computing any new fields.

        Examples:
            map(
                udf=compute_personal_greeting,
                cols=[
                    {'name': 'greeting', 'description': 'The greeting message', 'type': str},
                    {'name': 'age', 'description': 'The age of the person', 'type': int},
                    {'name': 'full_name', 'description': 'The name of the person', 'type': str},
                ]
            )
        """
        # sanity check inputs
        if udf is None:
            raise ValueError("`udf` must be provided for map.")

        return self._map(udf, cols, Cardinality.ONE_TO_ONE, depends_on)

    def flat_map(self, udf: Callable,
            cols: list[dict] | type[BaseModel] | None,
            depends_on: str | list[str] | None = None) -> Dataset:
        """
        Compute new fields (or update existing ones) with a UDF. For each input, this function will compute one or more outputs.

        Set `cols=None` if your flat_map is not computing any new fields.

        Examples:
            flat_map(
                udf=extract_paper_authors,
                cols=[
                    {'name': 'author_name', 'description': 'The name of the author', 'type': str},
                    {'name': 'institution', 'description': 'The institution of the author', 'type': str},
                    {'name': 'email', 'description': 'The author's email', 'type': str},
                ]
            )
        """
        # sanity check inputs
        if udf is None:
            raise ValueError("`udf` must be provided for map.")

        return self._map(udf, cols, Cardinality.ONE_TO_MANY, depends_on)

    def count(self) -> Dataset:
        """Apply a count aggregation to this set"""
        operator = Aggregate(input_schema=self.schema, agg_func=AggFunc.COUNT)
        return Dataset(sources=[self], operator=operator, schema=operator.output_schema)

    def average(self) -> Dataset:
        """Apply an average aggregation to this set"""
        operator = Aggregate(input_schema=self.schema, agg_func=AggFunc.AVERAGE)
        return Dataset(sources=[self], operator=operator, schema=operator.output_schema)

    def sum(self) -> Dataset:
        """Apply a summation to this set"""
        operator = Aggregate(input_schema=self.schema, agg_func=AggFunc.SUM)
        return Dataset(sources=[self], operator=operator, schema=operator.output_schema)

    def min(self) -> Dataset:
        """Apply an min operator to this set"""
        operator = Aggregate(input_schema=self.schema, agg_func=AggFunc.MIN)
        return Dataset(sources=[self], operator=operator, schema=operator.output_schema)

    def max(self) -> Dataset:
        """Apply an max operator to this set"""
        operator = Aggregate(input_schema=self.schema, agg_func=AggFunc.MAX)
        return Dataset(sources=[self], operator=operator, schema=operator.output_schema)

    def groupby(self, groupby: GroupBySig) -> Dataset:
        output_schema = groupby.output_schema()
        operator = GroupByAggregate(input_schema=self.schema, output_schema=output_schema, group_by_sig=groupby)
        return Dataset(sources=[self], operator=operator, schema=output_schema)

    def sem_agg(self, col: dict | type[BaseModel], agg: str, depends_on: str | list[str] | None = None) -> Dataset:
        """
        Apply a semantic aggregation to this set. The `agg` string will be applied using an LLM
        over the entire set of inputs' fields specified in `depends_on` to generate the output `col`.

        Example:
            sem_agg(
                col={'name': 'overall_sentiment', 'desc': 'The overall sentiment of the reviews', 'type': str},
                agg="Compute the overall sentiment of the reviews as POSITIVE or NEGATIVE.",
                depends_on="review_text",
            )
        """
        # construct new output schema
        new_output_schema = None
        if isinstance(col, dict):
            new_output_schema = create_schema_from_fields([col])
        elif issubclass(col, BaseModel):
            assert len(col.model_fields) == 1, "For semantic aggregation, when passing a BaseModel to `col` it must have exactly one field."
            new_output_schema = col
        else:
            raise ValueError("`col` must be a dictionary or a single-field BaseModel.")

        # enforce type for depends_on
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # construct logical operator
        operator = Aggregate(input_schema=self.schema, output_schema=new_output_schema, agg_str=agg, depends_on=depends_on)

        return Dataset(sources=[self], operator=operator, schema=operator.output_schema)

    def sem_topk(
        self,
        index: Collection,
        search_attr: str,
        output_attrs: list[dict] | type[BaseModel],
        search_func: Callable | None = None,
        k: int = -1,
    ) -> Dataset:
        """
        Retrieve the top-k nearest neighbors of the value of the `search_attr` from the `index` and
        use these results to construct the `output_attrs` field(s).
        """
        # construct new output schema
        new_output_schema = None
        if isinstance(output_attrs, list):
            output_attrs = create_schema_from_fields(output_attrs)
            new_output_schema = union_schemas([self.schema, output_attrs])
        elif issubclass(output_attrs, BaseModel):
            new_output_schema = union_schemas([self.schema, output_attrs])
        else:
            raise ValueError("`output_attrs` must be a list of dictionaries or a BaseModel.")

        # TODO: revisit once we can think through abstraction(s)
        # # construct the PZIndex from the user-provided index
        # index = index_factory(index)

        # construct logical operator
        operator = TopKScan(
            input_schema=self.schema,
            output_schema=new_output_schema,
            index=index,
            search_func=search_func,
            search_attr=search_attr,
            output_attrs=output_attrs,
            k=k,
        )

        return Dataset(sources=[self], operator=operator, schema=new_output_schema)

    def limit(self, n: int) -> Dataset:
        """Limit the set size to no more than n rows"""
        operator = LimitScan(input_schema=self.schema, output_schema=self.schema, limit=n)
        return Dataset(sources=[self], operator=operator, schema=self.schema)

    def distinct(self, distinct_cols: list[str] | None = None) -> Dataset:
        """Return a new Dataset with distinct rows based on the current schema."""
        operator = Distinct(input_schema=self.schema, output_schema=self.schema, distinct_cols=distinct_cols)
        return Dataset(sources=[self], operator=operator, schema=self.schema)

    def project(self, project_cols: list[str] | str) -> Dataset:
        """Project the Set to only include the specified columns."""
        project_cols = project_cols if isinstance(project_cols, list) else [project_cols]
        new_output_schema = project(self.schema, project_cols)
        operator = Project(input_schema=self.schema, output_schema=new_output_schema, project_cols=project_cols)
        return Dataset(sources=[self], operator=operator, schema=new_output_schema)

    def chunk(
        self,
        input_col: str = "text",
        output_col: str = "text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        *,
        chunker_kind: str = "recursive_character",
        chunker_params: dict[str, Any] | None = None,
        graph: Any | None = None,
        edge_policy: str | None = None,
        has_chunk_edge_type: str = "overlay:has_chunk",
        next_chunk_edge_type: str = "overlay:next_chunk",
        chunk_node_type: str | None = "chunk",
        overwrite_nodes: bool = False,
        overwrite_edges: bool = False,
    ) -> Dataset:
        """Chunks the text in `input_col` into smaller segments.

        If `graph` and `edge_policy` are set, chunk nodes are upserted into the graph and
        edges are added according to the policy.

        Supported edge_policy values:
        - None / "none": no graph side effects (default)
        - "has_chunk": add has_chunk edges (parent -> chunk)
        - "has_and_next": add has_chunk and next_chunk edges
        """
        from palimpzest.utils.chunkers import get_chunking_udf

        udf = get_chunking_udf(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            input_col=input_col,
            output_col=output_col,
            chunker_kind=chunker_kind,
            chunker_params=chunker_params,
        )
        
        existing_out_field = self.schema.model_fields.get(output_col)
        out_type = existing_out_field.annotation if existing_out_field is not None else str

        cols = [
            {"name": "id", "type": str, "desc": "The ID of the chunk"},
            {"name": output_col, "type": out_type, "desc": "The chunked text"},
            {"name": "chunk_index", "type": int, "desc": "Index of the chunk"},
            {"name": "source_node_id", "type": str, "desc": "ID of the source node"},
            {"name": "prev_chunk_id", "type": str | None, "desc": "ID of the previous chunk (if any)"},
        ]
        
        chunked = self.flat_map(udf=udf, cols=cols)
        # ConvertScan's ONE_TO_MANY semantics always emit at least one output record per
        # input record, even when the UDF returns an empty list. For chunking, those
        # placeholder outputs will have null chunk fields; filter them out.
        chunked = chunked.filter(lambda r: r.get("chunk_index") is not None, depends_on=["chunk_index"])

        policy = (edge_policy or "none").strip().lower()
        if policy in {"none", ""}:
            return chunked

        if graph is None:
            raise ValueError("Must provide `graph` when `edge_policy` is set.")

        from palimpzest.query.operators.logical import LinkFromField, UpsertGraphNodes

        # 1) Upsert chunk nodes into the graph (must happen before adding edges).
        upsert = UpsertGraphNodes(
            graph=graph,
            text_field=output_col,
            node_type=chunk_node_type,
            overwrite=overwrite_nodes,
            input_schema=chunked.schema,
            output_schema=chunked.schema,
        )
        chunked = Dataset(sources=[chunked], operator=upsert, schema=chunked.schema)

        # 2) source_node_id -> chunk_id edges
        if policy in {"has_chunk", "has_and_next"}:
            link_has = LinkFromField(
                graph=graph,
                edge_type=has_chunk_edge_type,
                src_field="source_node_id",
                dst_field=None,
                overwrite=overwrite_edges,
                input_schema=chunked.schema,
                output_schema=chunked.schema,
            )
            chunked = Dataset(sources=[chunked], operator=link_has, schema=chunked.schema)

        # 3) prev_chunk_id -> this_chunk_id
        if policy == "has_and_next":
            link_next = LinkFromField(
                graph=graph,
                edge_type=next_chunk_edge_type,
                src_field="prev_chunk_id",
                dst_field=None,
                ensure_src_node=True,
                placeholder_node_type=chunk_node_type,
                overwrite=overwrite_edges,
                input_schema=chunked.schema,
                output_schema=chunked.schema,
            )
            chunked = Dataset(sources=[chunked], operator=link_next, schema=chunked.schema)

        return chunked

    def embed(self, input_col: str = "text", output_col: str = "embedding", model_name: str = "openai", config: Any = None) -> Dataset:
        """
        Embeds the text in `input_col` using the specified embedding model.
        """
        from palimpzest.utils.embeddings import get_embedding_udf
        
        udf = get_embedding_udf(model_name=model_name, input_col=input_col, output_col=output_col, config=config)
        
        return self.map(
            udf=udf,
            cols=[{"name": output_col, "type": list[float], "desc": "Embedding of the text"}]
        )

    def summarize(
        self, 
        prompt: str, 
        output_col: str = "summary", 
        depends_on: str | list[str] | None = "text",
        aggregate: bool = False,
        graph: Any | None = None,
        edge_type: str | None = None
    ) -> Dataset:
        """
        Summarizes the content of the records using an LLM.
        
        Args:
            prompt: The instruction prompt for summarization.
            output_col: The name of the output field for the summary.
            depends_on: The input field(s) to summarize. Defaults to "text".
            aggregate: If True, aggregates all input records into a single summary (N-to-1).
                       If False (default), summarizes each record individually (1-to-1).
            graph: The GraphDataset to update with edges (required if edge_type is set).
            edge_type: If set, creates edges from the summary node(s) to the source node(s)
                       with this type (e.g. "SUMMARIZES").
        """
        # 1. Perform Summarization
        summary_ds = None
        if aggregate:
            summary_ds = self.sem_agg(
                col={"name": output_col, "type": str, "desc": "The summary generated by the LLM"},
                agg=prompt,
                depends_on=depends_on
            )
        else:
            summary_ds = self.sem_map(
                cols=[{"name": output_col, "type": str, "desc": "The summary generated by the LLM"}],
                desc=prompt,
                depends_on=depends_on
            )
            
        # 2. Optionally Link to Children
        if edge_type:
            if graph is None:
                # Try to infer graph from sources if possible, otherwise raise error
                # For now, we require explicit graph argument as Dataset doesn't strictly track the GraphDataset object
                raise ValueError("Must provide `graph` argument when `edge_type` is set.")
                
            from palimpzest.query.operators.logical import LinkToChildren
            
            link_op = LinkToChildren(
                graph=graph,
                edge_type=edge_type,
                input_schema=summary_ds.schema,
                output_schema=summary_ds.schema
            )
            
            return Dataset(sources=[summary_ds], operator=link_op, schema=summary_ds.schema)
            
        return summary_ds

    def run(self, config: QueryProcessorConfig | None = None, **kwargs):
        """Invoke the QueryProcessor to execute the query. `kwargs` will be applied to the QueryProcessorConfig."""
        # TODO: this import currently needs to be here to avoid a circular import; we should fix this in a subsequent PR
        from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

        # as syntactic sugar, we will allow some keyword arguments to parameterize our policies
        policy = construct_policy_from_kwargs(**kwargs)
        if policy is not None:
            kwargs["policy"] = policy

        # construct unique logical op ids for all operators in this dataset
        self._generate_unique_logical_op_ids()

        return QueryProcessorFactory.create_and_run_processor(self, config)

    def optimize_and_run(self, config: QueryProcessorConfig | None = None, train_dataset: dict[str, Dataset] | Dataset | None = None, validator: Validator | None = None, **kwargs):
        """Optimize the PZ program using the train_dataset and validator before running the optimized plan."""
        # TODO: this import currently needs to be here to avoid a circular import; we should fix this in a subsequent PR
        from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

        # confirm that either train_dataset or validator is provided
        assert train_dataset is not None or validator is not None, "Must provide at least one of train_dataset or validator to use optimize_and_run()"

        # validate the train_dataset has one input for each source dataset and normalize its type to be a dict
        if train_dataset is not None:
            root_datasets = self._get_root_datasets()
            if isinstance(train_dataset, Dataset) and len(root_datasets) > 1:
                raise ValueError(
                    "For plans with more than one root dataset, `train_dataset` must be a dictionary mapping"
                    " {'dataset_id' --> Dataset} for all root Datasets"
                )

            elif isinstance(train_dataset, Dataset):
                root_dataset_id = list(root_datasets.values())[0].id
                if train_dataset.id != root_dataset_id:
                    warnings.warn(
                        f"train_dataset.id={train_dataset.id} does not match root dataset id={root_dataset_id}\n"
                        f"Setting train_dataset to be the training data for root dataset with id={root_dataset_id} anyways.",
                        stacklevel=2,
                    )
                train_dataset = {root_dataset_id: train_dataset}

            elif not all(dataset_id in train_dataset for dataset_id in root_datasets):
                missing_ids = [dataset_id for dataset_id in root_datasets if dataset_id not in train_dataset]
                raise ValueError(
                    f"`train_dataset` is missing the following root dataset id(s): {missing_ids}"
                )

        # as syntactic sugar, we will allow some keyword arguments to parameterize our policies
        policy = construct_policy_from_kwargs(**kwargs)
        if policy is not None:
            kwargs["policy"] = policy

        # construct unique logical op ids for all operators in this dataset
        self._generate_unique_logical_op_ids()

        return QueryProcessorFactory.create_and_run_processor(self, config, train_dataset, validator)
