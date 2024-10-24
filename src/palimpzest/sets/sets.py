from __future__ import annotations

import hashlib
import json
from typing import Callable, List, Optional, Type, Union

from palimpzest.constants import MAX_ID_CHARS, AggFunc, Cardinality
from palimpzest.corelib import Number, Schema
from palimpzest.datamanager import DataDirectory
from palimpzest.datasources import DataSource
from palimpzest.elements import (
    Filter,
    GroupBySig,
)


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
        source: Union[Set, DataSource],
        schema: Schema,
        desc: str = None,
        filter: Filter = None,
        udf: Callable = None,
        aggFunc: AggFunc = None,
        groupBy: GroupBySig = None,
        limit: int = None,
        fnid: str = None,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        image_conversion: bool = None,
        depends_on: List[str] | None = None,
        nocache: bool = False,
    ):
        self._schema = schema
        self._source = source
        self._desc = desc
        self._filter = filter
        self._udf = udf
        self._aggFunc = aggFunc
        self._groupBy = groupBy
        self._limit = limit
        self._fnid = fnid
        self._cardinality = cardinality
        self._image_conversion = image_conversion
        self._depends_on = depends_on
        self._nocache = nocache

    def __str__(self):
        return f"{self.__class__.__name__}(schema={self.schema}, desc={self._desc}, filter={str(self._filter)}, udf={str(self._udf)}, aggFunc={str(self._aggFunc)}, limit={str(self._limit)}, uid={self.universalIdentifier()})"

    @property
    def schema(self) -> Type[Schema]:
        return self._schema

    def serialize(self):
        # NOTE: I needed to remove depends_on from the serialization dictionary because
        # the optimizer changes the name of the depends_on fields to be their "full" name.
        # This created an issue with the node.universalIdentifier() not being consistent
        # after changing the field to its full name.
        d = {
            "version": Set.SET_VERSION,
            "schema": self.schema.jsonSchema(),
            "source": self._source.serialize(),
            "desc": repr(self._desc),
            "filter": None if self._filter is None else self._filter.serialize(),
            "udf": None if self._udf is None else str(self._udf),
            "aggFunc": None if self._aggFunc is None else self._aggFunc.serialize(),
            "fnid": self._fnid,
            "cardinality": self._cardinality,
            "image_conversion": self._image_conversion,
            "limit": self._limit,
            "groupBy": (None if self._groupBy is None else self._groupBy.serialize()),
        }

        return d

    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        d = self.serialize()
        ordered = json.dumps(d, sort_keys=True)
        result = hashlib.sha256(ordered.encode()).hexdigest()
        return result[:MAX_ID_CHARS]

    def jsonSchema(self):
        """Return the JSON schema for this Set."""
        return self.schema.jsonSchema()


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

    def __init__(self, source: Union[str, DataSource], *args, **kwargs):
        # convert source (str) -> source (DataSource) if need be
        source = DataDirectory().getRegisteredDataset(source) if isinstance(source, str) else source

        # intialize class
        super().__init__(source, *args, **kwargs)

        if self._depends_on is None:
            self._depends_on = []

        elif type(self._depends_on) is str:
            self._depends_on = [self._depends_on]

    def filter(
        self,
        _filter: Union[str, callable],
        depends_on: Union[str, List[str]] | None = None,
        desc: str = "Apply filter(s)",
    ) -> Dataset:
        """Add a filter to the Set. This filter will possibly restrict the items that are returned later."""
        f = None
        if type(_filter) is str:
            f = Filter(_filter)
        elif callable(_filter):
            f = Filter(filterFn=_filter)
        else:
            raise Exception("Filter type not supported.", type(_filter))

        return Dataset(
            source=self,
            schema=self.schema,
            desc=desc,
            filter=f,
            depends_on=depends_on,
            nocache=self._nocache,
        )

    def convert(
        self,
        outputSchema: Schema,
        udf: Optional[Callable] = None,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        image_conversion: bool = False,
        depends_on: Union[str, List[str]] | None = None,
        desc: str = "Convert to new schema",
    ) -> Dataset:
        """Convert the Set to a new schema."""
        return Dataset(
            source=self,
            schema=outputSchema,
            udf=udf,
            cardinality=cardinality,
            image_conversion=image_conversion,
            depends_on=depends_on,
            desc=desc,
            nocache=self._nocache,
        )

    def count(self) -> Dataset:
        """Apply a count aggregation to this set"""
        return Dataset(
            source=self,
            schema=Number,
            desc="Count results",
            aggFunc=AggFunc.COUNT,
            nocache=self._nocache,
        )

    def average(self) -> Dataset:
        """Apply an average aggregation to this set"""
        return Dataset(
            source=self,
            schema=Number,
            desc="Average results",
            aggFunc=AggFunc.AVERAGE,
            nocache=self._nocache,
        )

    def groupby(self, groupBy: GroupBySig) -> Dataset:
        return Dataset(
            source=self,
            schema=groupBy.outputSchema(),
            desc="Group By",
            groupBy=groupBy,
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
