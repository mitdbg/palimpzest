from __future__ import annotations

import json
from typing import Callable

import pandas as pd

from palimpzest.constants import AggFunc, Cardinality
from palimpzest.core.data.datasources import DataSource
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.lib.fields import ListField, StringField
from palimpzest.core.lib.schemas import DefaultSchema, Number, Schema
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.hash_helpers import hash_for_id
from palimpzest.utils.index_helpers import get_index_str


#####################################################
#
#####################################################
class Set:
    """
    A Set is the logical abstraction for a set of DataRecords matching some Schema. It is
    also a node in the computation graph of a Dataset.

    Each Dataset consists of one or more Sets. The "initial" Set in a Dataset can be thought
    of as the Set that results from reading each DataRecord unaltered from the source. For each
    filter or transformation that is applied to the Dataset, a new Set is created which defines
    the set of DataRecords that result from applying that filter or transformation. In brief,
    the Sets define a Dataset's computation graph. Sets can also be cached to maximize the reuse
    of past computation.

    Sets are initialized with a dataset_id, a schema, and a source. The source is either an
    existing Set or a raw data source (such as a directory or S3 prefix). Sets may be initialized
    with a Filter (which defines the filtering performed on the source to obtain *this* Set),
    and a description of what this Set is meant to represent.
    """

    SET_VERSION = 0.1

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
        index=None,  # TODO(Siva): Abstract Index and add a type here and elsewhere
        search_func: Callable | None = None,
        search_attr: str | None = None,
        output_attr: str | None = None,
        k: int | None = None,  # TODO: disambiguate `k` to be something like `retrieve_k`
        limit: int | None = None,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        depends_on: list[str] | None = None,
        nocache: bool = False,
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
        self._output_attr = output_attr
        self._k = k
        self._limit = limit
        self._cardinality = cardinality
        self._depends_on = [] if depends_on is None else sorted(depends_on)
        self._nocache = nocache

    def __str__(self):
        return (
            f"{self.__class__.__name__}(schema={self.schema}, desc={self._desc}, "
            f"filter={str(self._filter)}, udf={str(self._udf)}, agg_func={str(self._agg_func)}, limit={str(self._limit)}, "
            f"project_cols={str(self._project_cols)}, uid={self.universal_identifier()})"
        )

    @property
    def schema(self) -> Schema:
        return self._schema

    def serialize(self):
        # NOTE: I needed to remove depends_on from the serialization dictionary because
        # the optimizer changes the name of the depends_on fields to be their "full" name.
        # This created an issue with the node.universal_identifier() not being consistent
        # after changing the field to its full name.
        d = {
            "version": Set.SET_VERSION,
            "schema": self.schema.json_schema(),
            "source": self._source.serialize(),
            "desc": repr(self._desc),
            "filter": None if self._filter is None else self._filter.serialize(),
            "udf": None if self._udf is None else str(self._udf),
            "agg_func": None if self._agg_func is None else self._agg_func.value,
            "cardinality": self._cardinality,
            "limit": self._limit,
            "group_by": (None if self._group_by is None else self._group_by.serialize()),
            "project_cols": (None if self._project_cols is None else self._project_cols),
            "index": None if self._index is None else get_index_str(self._index),
            "search_func": None if self._search_func is None else str(self._search_func),
            "search_attr": self._search_attr,
            "output_attr": self._output_attr,
            "k": self._k,
        }

        return d

    def universal_identifier(self):
        """Return a unique identifier for this Set."""
        d = self.serialize()
        ordered = json.dumps(d, sort_keys=True)
        result = hash_for_id(ordered)
        return result

    def json_schema(self):
        """Return the JSON schema for this Set."""
        return self.schema.json_schema()


class Dataset(Set):
    """
    A Dataset is the intended abstraction for programmers to interact with when manipulating Sets.

    Users instantiate a Dataset by specifying a `source` that either points to a
    DataSource or an existing cached Set. Users can then perform computations on
    the Dataset in an imperative fashion by leveraging functions such as `filter`,
    `convert`, `aggregate`, etc. Underneath the hood, each of these operations creates
    a new Set which is cached by the DataManager. As a result, the Sets define the
    lineage of computation on a Dataset, and this enables programmers to re-use
    previously cached computation by providing it as a `source` to some future Dataset.
    """

    def __init__(self, source: str | list | pd.DataFrame | DataSource, schema: Schema | None = None, *args, **kwargs):
        # convert source (str) -> source (DataSource) if need be
        updated_source = DataDirectory().get_or_register_dataset(source) if isinstance(source, (str, list, pd.DataFrame)) else source

        if schema is None:
            # This is mainly for DataSource with a schema. 
            if updated_source.schema is not None:
                schema = updated_source.schema
            else:
                schema = Schema.from_df(source) if isinstance(source, pd.DataFrame) else DefaultSchema
        # intialize class
        super().__init__(updated_source, schema, *args, **kwargs)

    def copy(self) -> Dataset:
        source_copy = self._source.copy()
        dataset_copy = Dataset(
            schema=self.schema,
            source=source_copy,
            desc=self._desc,
            filter=self._filter,
            udf=self._udf,
            agg_func=self._agg_func,
            group_by=self._group_by,
            index=self._index,
            search_func=self._search_func,
            search_attr=self._search_attr,
            output_attr=self._output_attr,
            k=self._k,
            limit=self._limit,
            cardinality=self._cardinality,
            depends_on=self._depends_on,
            nocache=self._nocache,
        )
        return dataset_copy

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
            nocache=self._nocache,
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
            nocache=self._nocache,
        )        

    # TODO(Jun): Remove in https://github.com/mitdbg/palimpzest/issues/94
    def convert(
        self,
        output_schema: Schema,
        udf: Callable | None = None,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        depends_on: str | list[str] | None = None,
        desc: str = "Convert to new schema",
    ) -> Dataset:
        """Convert the Set to a new schema.

        Deprecated: This method will be removed in a future version. Please use add_columns() or sem_add_columns() instead.
        """
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        return Dataset(
            source=self,
            schema=output_schema,
            udf=udf,
            cardinality=cardinality,
            depends_on=depends_on,
            desc=desc,
            nocache=self._nocache,
        )

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
            desc="Add new columns " + str(cols),
            nocache=self._nocache,
        )

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
            desc="Add new columns via UDF",
            depends_on=depends_on,
            nocache=self._nocache,
        )

    def count(self) -> Dataset:
        """Apply a count aggregation to this set"""
        return Dataset(
            source=self,
            schema=Number,
            desc="Count results",
            agg_func=AggFunc.COUNT,
            nocache=self._nocache,
        )

    def average(self) -> Dataset:
        """Apply an average aggregation to this set"""
        return Dataset(
            source=self,
            schema=Number,
            desc="Average results",
            agg_func=AggFunc.AVERAGE,
            nocache=self._nocache,
        )

    def groupby(self, groupby: GroupBySig) -> Dataset:
        return Dataset(
            source=self,
            schema=groupby.output_schema(),
            desc="Group By",
            group_by=groupby,
            nocache=self._nocache,
        )

    def retrieve(
        self, index, search_func: Callable, search_attr: str, output_attr: str, output_attr_desc: str, k=-1
    ) -> Dataset:
        """
        Retrieve the top k nearest neighbors of the value of the `search_attr` from the index and
        stores it in the `output_attr` field. The output schema is a union of the current schema
        and the `output_attr` with type ListField(StringField). `search_func` is a function of
        type (index, query: str | list(str), k: int) -> list[str]. It should implement the lookup
        logic for the index and return the top k results. The value of the `search_attr` field is
        used as the query to lookup in the index. The results are stored in the `output_attr`
        field. `output_attr_desc` is the description of the `output_attr` field.
        """
        # Output schema is a union of the current schema and the output_attr
        attributes = {output_attr: ListField(StringField)(desc=output_attr_desc)}
        output_schema = self.schema().union(type("temp_class", (Schema,), attributes))
        return Dataset(
            source=self,
            schema=output_schema,
            desc="Retrieve",
            index=index,
            search_func=search_func,
            search_attr=search_attr,
            output_attr=output_attr,
            k=k,
            nocache=self._nocache,
        )

    def limit(self, n: int) -> Dataset:
        """Limit the set size to no more than n rows"""
        return Dataset(
            source=self,
            schema=self.schema,
            desc="LIMIT " + str(n),
            limit=n,
            nocache=self._nocache,
        )

    def project(self, project_cols: list[str] | str) -> Dataset:
        """Project the Set to only include the specified columns."""
        return Dataset(
            source=self,
            schema=self.schema.project(project_cols),
            project_cols=project_cols if isinstance(project_cols, list) else [project_cols],
            nocache=self._nocache,
        )

    def run(self, config: QueryProcessorConfig | None = None, **kwargs):  # noqa: F821
        from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

        return QueryProcessorFactory.create_and_run_processor(self, config, **kwargs)
