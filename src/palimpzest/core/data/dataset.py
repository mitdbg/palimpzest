from __future__ import annotations

from typing import Callable

from chromadb.api.models.Collection import Collection

from palimpzest.constants import AggFunc, Cardinality
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.lib.schemas import Number, Schema
from palimpzest.policy import construct_policy_from_kwargs
from palimpzest.query.operators.logical import (
    Aggregate,
    ConvertScan,
    FilteredScan,
    GroupByAggregate,
    LimitScan,
    LogicalOperator,
    MapScan,
    Project,
    RetrieveScan,
)
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.hash_helpers import hash_for_serialized_dict


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
            schema: type[Schema] | None = None,
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
            schema (type[`Schema`] | None): The `Schema`
            id (str | None): an identifier for this Dataset provided by the user

        Raises:
            ValueError: if `sources` is not a `Dataset` or list of `Datasets`
        """
        # set sources
        self._sources = None
        if isinstance(sources, list):
            self._sources = sources
        elif isinstance(sources, Dataset):
            self._sources = [sources]
        elif sources is not None:
            raise ValueError("Dataset sources must be another Dataset or a list of Datasets. For root Datasets, you must subclass pz.IterDataset, pz.IndexDataset, or pz.Context.")

        # set the logical operator
        self._operator: LogicalOperator = operator

        # compute the schema
        self._schema = schema
        if self._schema is None:
            for source in self._sources.values():
                self._schema = source.schema if self._schema is None else self._schema.union(source.schema)

        # compute the dataset id
        self._id = self._compute_dataset_id() if id is None else id

    @property
    def id(self) -> str:
        """The string identifier for this `Dataset`"""
        return self._id

    @property
    def schema(self) -> Schema:
        """The `Schema` of this `Dataset`"""
        return self._schema

    @property
    def is_root(self) -> bool:
        return self._sources is None

    def __str__(self) -> str:
        return f"Dataset(schema={self._schema}, id={self._id}, op_id={self._operator.get_logical_op_id()})"

    def _compute_dataset_id(self) -> str:
        """
        Compute the identifier for this `Dataset`. The ID is uniquely defined by the operation(s)
        applied to the `Dataset's` sources.
        """
        return hash_for_serialized_dict({
            "source_ids": None if self._sources is None else [source.id for source in self._sources],
            "logical_op_id": self._operator.get_logical_op_id(),
        })

    def _set_data_source(self, id: str, source: Dataset) -> None:
        """
        Update the source with the given `id` to use the new `source`. This is used during optimization
        to re-use the same physical plan while running it on a validation dataset. If the `id` is not
        present in this `Dataset`'s sources, pass the update through to its sources.

        Args:
            id (str): the identifier for the source `Dataset`
            source (`Dataset`): the (validation) dataset
        """
        new_sources = []
        for old_source in self._sources:
            if id == old_source.id:
                new_sources.append(source)
            else:
                old_source._set_data_source(id, source)
                new_sources.append(old_source)

    # TODO
    def _resolve_depends_on(self, depends_on: list[str]) -> list[str]:
        """
        TODO: resolve the `depends_on` strings to their full field names ({Dataset.id}.{field_name}).
        """
        return []

    def copy(self):
        return Dataset(
            sources=None if self._sources is None else [source.copy() for source in self._sources],
            operator=self._operator,
            schema=self._schema,
        )

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
        operator = FilteredScan(input_schema=self.schema, output_schema=self.schema, filter=f, depends_on=depends_on)

        return Dataset(sources=[self], operator=operator, schema=self.schema)

    def sem_add_columns(self, cols: list[dict] | type[Schema],
                        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
                        depends_on: str | list[str] | None = None) -> Dataset:
        """
        Add new columns by specifying the column names, descriptions, and types.
        The column will be computed during the execution of the Dataset.
        Example:
            sem_add_columns(
                [{'name': 'greeting', 'desc': 'The greeting message', 'type': str},
                 {'name': 'age', 'desc': 'The age of the person', 'type': int},
                 {'name': 'full_name', 'desc': 'The name of the person', 'type': str}]
            )
        """
        # construct new output schema
        new_output_schema = None
        if isinstance(cols, list):
            new_output_schema = self.schema.add_fields(cols)
        elif issubclass(cols, Schema):
            new_output_schema = self.schema.union(cols)
        else:
            raise ValueError("`cols` must be a list of dictionaries or a Schema.")

        # enforce type for depends_on
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # construct logical operator
        operator = ConvertScan(
            input_schema=self.schema,
            output_schema=new_output_schema,
            cardinality=cardinality,
            udf=None,
            depends_on=depends_on,
        )

        return Dataset(sources=[self], operator=operator, schema=new_output_schema)

    def add_columns(self, udf: Callable,
                    cols: list[dict] | type[Schema],
                    cardinality: Cardinality = Cardinality.ONE_TO_ONE,
                    depends_on: str | list[str] | None = None) -> Dataset:
        """
        Add new columns by specifying UDFs.

        Examples:
            add_columns(
                udf=compute_personal_greeting,
                cols=[
                    {'name': 'greeting', 'desc': 'The greeting message', 'type': str},
                    {'name': 'age', 'desc': 'The age of the person', 'type': int},
                    {'name': 'full_name', 'desc': 'The name of the person', 'type': str},
                ]
            )
        """
        # sanity check inputs
        if udf is None or cols is None:
            raise ValueError("`udf` and `cols` must be provided for add_columns.")

        # construct new output schema
        new_output_schema = None
        if isinstance(cols, list):
            updated_cols = []
            for col_dict in cols:
                assert isinstance(col_dict, dict), "each entry in `cols` must be a dictionary"
                assert "name" in col_dict, "each type must contain a 'name' key specifying the column name"
                assert "type" in col_dict, "each type must contain a 'type' key specifying the column type"
                col_dict["desc"] = col_dict.get("desc", "New column: " + col_dict["name"])
                updated_cols.append(col_dict)
            new_output_schema = self.schema.add_fields(updated_cols)

        elif issubclass(cols, Schema):
            new_output_schema = self.schema.union(cols)

        else:
            raise ValueError("`cols` must be a list of dictionaries or a Schema.")

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

    def map(self, udf: Callable) -> Dataset:
        """
        Apply a UDF map function.

        Examples:
            map(udf=clean_column_values)
        """
        # sanity check inputs
        if udf is None:
            raise ValueError("`udf` must be provided for map.")

        # construct logical operator
        operator = MapScan(input_schema=self.schema, output_schema=self.schema, udf=udf)

        return Dataset(sources=[self], operator=operator, schema=self.schema)

    def count(self) -> Dataset:
        """Apply a count aggregation to this set"""
        operator = Aggregate(input_schema=self.schema, output_schema=Number, agg_func=AggFunc.COUNT)
        return Dataset(sources=[self], operator=operator, schema=Number)

    def average(self) -> Dataset:
        """Apply an average aggregation to this set"""
        operator = Aggregate(input_schema=self.schema, output_schema=Number, agg_func=AggFunc.AVERAGE)
        return Dataset(sources=[self], operator=operator, schema=Number)

    def groupby(self, groupby: GroupBySig) -> Dataset:
        output_schema = groupby.output_schema()
        operator = GroupByAggregate(input_schema=self.schema, output_schema=output_schema, group_by_sig=groupby)
        return Dataset(sources=[self], operator=operator, schema=output_schema)

    def retrieve(
        self,
        index: Collection,
        search_attr: str,
        output_attrs: list[dict] | type[Schema],
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
            new_output_schema = self.schema.add_fields(output_attrs)
        elif issubclass(output_attrs, Schema):
            new_output_schema = self.schema.union(output_attrs)
        else:
            raise ValueError("`cols` must be a list of dictionaries or a Schema.")

        # TODO: revisit once we can think through abstraction(s)
        # # construct the PZIndex from the user-provided index
        # index = index_factory(index)

        # construct logical operator
        operator = RetrieveScan(
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

    def project(self, project_cols: list[str] | str) -> Dataset:
        """Project the Set to only include the specified columns."""
        project_cols = project_cols if isinstance(project_cols, list) else [project_cols]
        new_output_schema = self.schema.project(project_cols)
        operator = Project(input_schema=self.schema, output_schema=new_output_schema, project_cols=project_cols)
        return Dataset(sources=[self], operator=operator, schema=new_output_schema)

    def run(self, config: QueryProcessorConfig | None = None, **kwargs):
        """Invoke the QueryProcessor to execute the query. `kwargs` will be applied to the QueryProcessorConfig."""
        # TODO: this import currently needs to be here to avoid a circular import; we should fix this in a subsequent PR
        from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

        # as syntactic sugar, we will allow some keyword arguments to parameterize our policies
        policy = construct_policy_from_kwargs(**kwargs)
        if policy is not None:
            kwargs["policy"] = policy

        return QueryProcessorFactory.create_and_run_processor(self, config, **kwargs)
