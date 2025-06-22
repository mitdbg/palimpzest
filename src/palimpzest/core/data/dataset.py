from __future__ import annotations

from pathlib import Path
from typing import Callable, NewType

import pandas as pd
from chromadb.api.models.Collection import Collection

from palimpzest.constants import AggFunc, Cardinality
from palimpzest.core.data.datasource import DataSource, resolve_datasource
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.lib.schemas import Number, Schema
from palimpzest.policy import construct_policy_from_kwargs
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.hash_helpers import hash_for_serialized_dict

# DEFINITIONS
DatasetType = NewType("Dataset", None)
DataSourceType = DataSource | DatasetType | str | Path | pd.DataFrame | list

class Dataset:
    """
    A `Dataset` represents a collection of structured or unstructured data that can be processed and
    transformed.
    
    Each `Dataset` contains a collection of `DataSource`s. Each `DataSource` supports being accessed
    as an iterable (allowing PZ to iterate over the items in the `Dataset`) and may optionally support
    indexing, which can be used to perform exact or fuzzy point lookups.
    
    Users can perform computations on the `Dataset` in a lazy or eager fashion. Applying functions
    such as `sem_filter`, `sem_map`, `sem_join`, `sem_agg`, etc. will lazily create a new `Dataset`.
    Users can invoke the `run()` method to execute the computation and retrieve a materialized `Dataset`.
    Materialized `Dataset`s can be processed further, or their results can be retrieved using `.get()`.
    """
    def __init__(self, sources: dict[str, DataSourceType] | list[DataSourceType] | DataSourceType, **kwargs) -> None:
        """
        Initialize a Dataset with a collection of DataSource(s) or existing Dataset(s).

        Args:
            sources (dict[str, DataSourceType] | list[DataSourceType] | DataSourceType):
                A dictionary of DataSource(s) or Dataset(s). The keys are identifiers for the sources,
                and the values are either DataSource objects or Dataset objects.

                Alternatively, the user may provide: 
                    - a list of DataSource or Dataset objects
                    - a single Datasource
                    - a single Dataset
                    - a string or Path to a local file or directory, which will be converted to a DataSource.
                    - a pandas DataFrame, which will be converted to a MemorySource.
                    - a list of objects, which will be converted to a MemorySource.

                If any of the alternatives are provided, they will be given ordinal identifiers relative
                to the order in which they are provided (e.g., "source_0", "source_1", etc.).

                If the user optimizes their computation, and the user provided a dictionary of sources, then
                the dictionary keys of the validation data must match the keys of the sources. If the user
                provided a list of sources, then the order of the validation data must match the order of the
                sources.

        Raises:
            ValueError: If the provided sources are not of the expected type.
        """
        # TODO: source identifiers should map one-to-one to the underlying data
        # TODO: we should automatically check for users providing the same source_id twice and warn the user
        #       that PZ will treat their underlying data as being identical during optimization
        # TODO: instantly translate all `field_names` in `depends_on` to `source_key.field_name`
        # TODO: node ids are based on `source_key.[field_names]`
        # load sources
        self._sources = {}
        if isinstance(sources, dict):
            for source_id, source in sources.items():
                source = source if isinstance(source, (DataSource, Dataset)) else resolve_datasource(source)
                # TODO: what if source is Dataset? should Dataset also have set_id()? should DataSource and Dataset share a base class?
                source.set_id(source_id)
                self._sources[source_id] = source

        elif isinstance(sources, (list, tuple)) and all([isinstance(source, (DataSource, Dataset, str, Path, pd.DataFrame)) for source in sources]):
            self._sources = {
                f"source_{idx}": source if isinstance(source, (DataSource, Dataset)) else resolve_datasource(source)
                for idx, source in enumerate(sources)
            }
        else:
            self._sources = {"source_0": resolve_datasource(sources)}

        # sort sources by key
        self._sources = dict(sorted(self._sources.items()))

        # set schema
        self._schema = None
        for source in self._sources.values():
            self._schema = source.schema if self._schema is None else self._schema.union(source.schema)

        # initialize the dictionary of attributes
        self._attrs = {**kwargs}

    @property
    def schema(self) -> Schema:
        return self._schema

    def _set_data_source(self, source_key: str, source: DataSourceType) -> None:
        """
        Update the given `source_key` to use the given `source`. This is used during optimization
        to re-use the same physical plan while running it on new validation data. If the `source_key`
        is not present in this `Dataset`'s sources, pass the update through to its sources.

        Args:
            source_key (str): the identifier for the source
            source (DataSourceType): the (validation) data source 
        """
        if source_key in self._sources:
            self._sources[source_key] = source
        else:
            for _, source in self._sources.items():
                if isinstance(source, Dataset):
                    source._set_data_source(source_key, source)

    def serialize(self):
        # NOTE: I needed to remove depends_on from the serialization dictionary because
        # the optimizer changes the name of the depends_on fields to be their "full" name.
        # This created an issue with the node.universal_identifier() not being consistent
        # after changing the field to its full name.
        d = {
            "schema": self.schema.json_schema(),
            "sources": {k: source.serialize() for k, source in self._sources.items()},
            "filter": None if self._filter is None else self._filter.serialize(),
            "udf": None if self._udf is None else self._udf.__name__,
            "agg_func": None if self._agg_func is None else self._agg_func.value,
            "cardinality": self._cardinality,
            "limit": self._limit,
            "group_by": None if self._group_by is None else self._group_by.serialize(),
            "project_cols": None if self._project_cols is None else self._project_cols,
            "index": None if self._index is None else self._index.__class__.__name__,
            "search_func": None if self._search_func is None else self._search_func.__name__,
            "search_attr": self._search_attr,
            "output_attrs": None if self._output_attrs is None else str(self._output_attrs),
            "k": self._k,
        }

        return d

    def universal_identifier(self):
        """Return a unique identifier for this Set."""
        return hash_for_serialized_dict(self.serialize())

    def json_schema(self):
        """Return the JSON schema for this Set."""
        return self.schema.json_schema()


#####################################################
#
#####################################################
class Set:
    """
    """

    def __init__(
        self,
        source: Set | DataSource,
        schema: Schema,
        desc: str | None = None,
        filter: Filter | None = None,
        udf: Callable | None = None,
        agg_func: AggFunc | None = None,
        group_by: GroupBySig | None = None,
        project_cols: list[str] | None = None,
        index: Collection | None = None,
        search_func: Callable | None = None,
        search_attr: str | None = None,
        output_attrs: list[dict] | None = None,
        k: int | None = None,  # TODO: disambiguate `k` to be something like `retrieve_k`
        limit: int | None = None,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        depends_on: list[str] | None = None,
        cache: bool = False,
    ):
        self._schema = schema
        self._source = source
        self._desc = desc
        self._filter = filter
        self._udf = udf
        self._agg_func = agg_func
        self._group_by = group_by
        self._project_cols = None if project_cols is None else sorted(project_cols)
        self._index = index
        self._search_func = search_func
        self._search_attr = search_attr
        self._output_attrs = output_attrs
        self._k = k
        self._limit = limit
        self._cardinality = cardinality
        self._depends_on = [] if depends_on is None else sorted(depends_on)
        self._cache = cache

    


class Dataset(Set):
    """
    A Dataset is the intended abstraction for programmers to interact with when writing PZ programs.

    Users instantiate a Dataset by specifying a `source` that either points to a DataSource
    or an existing Dataset. Users can then perform computations on the Dataset in a lazy fashion
    by leveraging functions such as `filter`, `sem_filter`, `sem_add_columns`, `aggregate`, etc.
    Underneath the hood, each of these operations creates a new Dataset. As a result, the Dataset
    defines a lineage of computation.
    """

    def __init__(
        self,
        source: str | Path | list | pd.DataFrame | DataSource | Dataset,
        schema: Schema | None = None,
        *args,
        **kwargs,
    ) -> None:
        # NOTE: this function currently assumes that DataSource will always be provided with a schema;
        #       we will relax this assumption in a subsequent PR
        # convert source into a DataSource
        updated_source = get_local_datasource(source, **kwargs) if isinstance(source, (str, Path, list, pd.DataFrame)) else source

        # get the schema
        schema = updated_source.schema if schema is None else schema

        # intialize class
        super().__init__(updated_source, schema, *args, **kwargs)

    def copy(self):
        return Dataset(
            source=self._source.copy() if isinstance(self._source, Set) else self._source,
            schema=self._schema,
            desc=self._desc,
            filter=self._filter,
            udf=self._udf,
            agg_func=self._agg_func,
            group_by=self._group_by,
            project_cols=self._project_cols,
            index=self._index,
            search_func=self._search_func,
            search_attr=self._search_attr,
            output_attrs=self._output_attrs,
            k=self._k,
            limit=self._limit,
            cardinality=self._cardinality,
            depends_on=self._depends_on,
            cache=self._cache,
        )

    def filter(
        self,
        _filter: Callable,
        depends_on: str | list[str] | None = None,
    ) -> Dataset:
        """Add a user defined function as a filter to the Set. This filter will possibly restrict the items that are returned later."""
        f = None
        if callable(_filter):
            f = Filter(filter_fn=_filter)
        else:
            error_str = f"Only support callable for filter, currently got {type(_filter)}"
            if isinstance(_filter, str):
                error_str += ". Consider using sem_filter() for semantic filters."
            raise Exception(error_str)

        if isinstance(depends_on, str):
            depends_on = [depends_on]

        return Dataset(
            source=self,
            schema=self.schema,
            filter=f,
            depends_on=depends_on,
            cache=self._cache,
        )

    def sem_filter(
        self,
        _filter: str,
        depends_on: str | list[str] | None = None,
    ) -> Dataset:
        """Add a natural language description of a filter to the Set. This filter will possibly restrict the items that are returned later."""
        f = None
        if isinstance(_filter, str):
            f = Filter(_filter)
        else:
            raise Exception("sem_filter() only supports `str` input for _filter.", type(_filter))

        if isinstance(depends_on, str):
            depends_on = [depends_on]

        return Dataset(
            source=self,
            schema=self.schema,
            filter=f,
            depends_on=depends_on,
            cache=self._cache,
        )

    def sem_add_columns(self, cols: list[dict] | type[Schema],
                        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
                        depends_on: str | list[str] | None = None,
                        desc: str = "Add new columns via semantic reasoning") -> Dataset:
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
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        new_output_schema = None
        if isinstance(cols, list):
            new_output_schema = self.schema.add_fields(cols)
        elif issubclass(cols, Schema):
            new_output_schema = self.schema.union(cols)
        else:
            raise ValueError("`cols` must be a list of dictionaries or a Schema.")

        return Dataset(
            source=self,
            schema=new_output_schema,
            udf=None,
            cardinality=cardinality,
            depends_on=depends_on,
            desc=desc,
            cache=self._cache,
        )

    def add_columns(self, udf: Callable,
                    cols: list[dict] | type[Schema],
                    cardinality: Cardinality = Cardinality.ONE_TO_ONE,
                    depends_on: str | list[str] | None = None,
                    desc: str = "Add new columns via UDF") -> Dataset:
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
        if udf is None or cols is None:
            raise ValueError("`udf` and `cols` must be provided for add_columns.")

        if isinstance(depends_on, str):
            depends_on = [depends_on]

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

        return Dataset(
            source=self,
            schema=new_output_schema,
            udf=udf,
            cardinality=cardinality,
            desc=desc,
            depends_on=depends_on,
            cache=self._cache,
        )

    def map(self, udf: Callable) -> Dataset:
        """
        Apply a UDF map function.

        Examples:
            map(udf=clean_column_values)
        """
        if udf is None:
            raise ValueError("`udf` must be provided for map.")

        return Dataset(
            source=self,
            schema=self.schema,
            udf=udf,
            cache=self._cache,
        )

    def count(self) -> Dataset:
        """Apply a count aggregation to this set"""
        return Dataset(
            source=self,
            schema=Number,
            desc="Count results",
            agg_func=AggFunc.COUNT,
            cache=self._cache,
        )

    def average(self) -> Dataset:
        """Apply an average aggregation to this set"""
        return Dataset(
            source=self,
            schema=Number,
            desc="Average results",
            agg_func=AggFunc.AVERAGE,
            cache=self._cache,
        )

    def groupby(self, groupby: GroupBySig) -> Dataset:
        return Dataset(
            source=self,
            schema=groupby.output_schema(),
            desc="Group By",
            group_by=groupby,
            cache=self._cache,
        )

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

        return Dataset(
            source=self,
            schema=new_output_schema,
            desc="Retrieve",
            index=index,
            search_func=search_func,
            search_attr=search_attr,
            output_attrs=output_attrs,
            k=k,
            cache=self._cache,
        )

    def limit(self, n: int) -> Dataset:
        """Limit the set size to no more than n rows"""
        return Dataset(
            source=self,
            schema=self.schema,
            desc="LIMIT " + str(n),
            limit=n,
            cache=self._cache,
        )

    def project(self, project_cols: list[str] | str) -> Dataset:
        """Project the Set to only include the specified columns."""
        return Dataset(
            source=self,
            schema=self.schema.project(project_cols),
            project_cols=project_cols if isinstance(project_cols, list) else [project_cols],
            cache=self._cache,
        )

    def run(self, config: QueryProcessorConfig | None = None, **kwargs):
        """Invoke the QueryProcessor to execute the query. `kwargs` will be applied to the QueryProcessorConfig."""
        # TODO: this import currently needs to be here to avoid a circular import; we should fix this in a subsequent PR
        from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

        # as syntactic sugar, we will allow some keyword arguments to parameterize our policies
        policy = construct_policy_from_kwargs(**kwargs)
        if policy is not None:
            kwargs["policy"] = policy

        return QueryProcessorFactory.create_and_run_processor(self, config, **kwargs)
