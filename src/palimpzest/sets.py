from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from chromadb.api.models.Collection import Collection
from ragatouille.RAGPretrainedModel import RAGPretrainedModel

from palimpzest.constants import AggFunc, Cardinality
from palimpzest.core.data.datareaders import DataReader
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.lib.fields import ListField, StringField
from palimpzest.core.lib.schemas import Number, Schema
from palimpzest.policy import construct_policy_from_kwargs
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.datareader_helpers import get_local_datareader
from palimpzest.utils.hash_helpers import hash_for_serialized_dict


#####################################################
#
#####################################################
class Set:
    """
    """

    def __init__(
        self,
        source: Set | DataReader,
        schema: Schema,
        desc: str | None = None,
        filter: Filter | None = None,
        udf: Callable | None = None,
        agg_func: AggFunc | None = None,
        group_by: GroupBySig | None = None,
        project_cols: list[str] | None = None,
        index: Collection | RAGPretrainedModel | None = None,
        search_func: Callable | None = None,
        search_attr: str | None = None,
        output_attr: str | None = None,
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
        self._output_attr = output_attr
        self._k = k
        self._limit = limit
        self._cardinality = cardinality
        self._depends_on = [] if depends_on is None else sorted(depends_on)
        self._cache = cache

    @property
    def schema(self) -> Schema:
        return self._schema

    def _set_data_source(self, source: DataReader):
        if isinstance(self._source, Set):
            self._source._set_data_source(source)
        else:
            self._source = source

    def serialize(self):
        # NOTE: I needed to remove depends_on from the serialization dictionary because
        # the optimizer changes the name of the depends_on fields to be their "full" name.
        # This created an issue with the node.universal_identifier() not being consistent
        # after changing the field to its full name.
        d = {
            "schema": self.schema.json_schema(),
            "source": self._source.serialize(),
            "desc": repr(self._desc),
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
            "output_attr": self._output_attr,
            "k": self._k,
        }

        return d

    def universal_identifier(self):
        """Return a unique identifier for this Set."""
        return hash_for_serialized_dict(self.serialize())

    def json_schema(self):
        """Return the JSON schema for this Set."""
        return self.schema.json_schema()


class Dataset(Set):
    """
    A Dataset is the intended abstraction for programmers to interact with when writing PZ programs.

    Users instantiate a Dataset by specifying a `source` that either points to a DataReader
    or an existing Dataset. Users can then perform computations on the Dataset in a lazy fashion
    by leveraging functions such as `filter`, `sem_filter`, `sem_add_columns`, `aggregate`, etc.
    Underneath the hood, each of these operations creates a new Dataset. As a result, the Dataset
    defines a lineage of computation.
    """

    def __init__(
        self,
        source: str | Path | list | pd.DataFrame | DataReader | Dataset,
        schema: Schema | None = None,
        *args,
        **kwargs,
    ) -> None:
        # NOTE: this function currently assumes that DataReader will always be provided with a schema;
        #       we will relax this assumption in a subsequent PR
        # convert source into a DataReader
        updated_source = get_local_datareader(source, **kwargs) if isinstance(source, (str, Path, list, pd.DataFrame)) else source

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
            output_attr=self._output_attr,
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
        index: Collection | RAGPretrainedModel,
        search_attr: str,
        output_attr: str,
        search_func: Callable | None = None,
        output_attr_desc: str | None = None,
        k: int = -1,
    ) -> Dataset:
        """
        Retrieve the top-k nearest neighbors of the value of the `search_attr` from the index and
        stores it in the `output_attr` field.

        The output schema is a union of the current schema and the `output_attr` with type ListField(StringField).
        `search_func` is a function of type (index, query: str | list(str), k: int) -> list[str]. It should
        implement the lookuplogic for the index and return the top-k results. The value of the `search_attr`
        field is used as the query to lookup in the index. The results are stored in the `output_attr` field.
        `output_attr_desc` is the description of the `output_attr` field.
        """
        # Output schema is a union of the current schema and the output_attr
        attributes = {output_attr: ListField(StringField)(desc=output_attr_desc)}
        output_schema = self.schema().union(type("Schema", (Schema,), attributes))

        # TODO: revisit once we can think through abstraction(s)
        # # construct the PZIndex from the user-provided index
        # index = index_factory(index)

        return Dataset(
            source=self,
            schema=output_schema,
            desc="Retrieve",
            index=index,
            search_func=search_func,
            search_attr=search_attr,
            output_attr=output_attr,
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
