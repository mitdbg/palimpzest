from palimpzest.datamanager import DataDirectory # TODO: DataManager
from palimpzest.elements import AggregateFunction, File, Filter, Number, Schema
from palimpzest.operators import (
    ApplyAggregateFunction,
    BaseScan,
    CacheScan,
    ConvertScan,
    FilteredScan,
    LimitScan,
    LogicalOperator,
)
from palimpzest.sets import DataSource, DirectorySource, FileSource, MemorySource

from __future__ import annotations
from typing import Union

import hashlib
import json
import os


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

    def __init__(self, dataset_id: str, schema: Schema, source: Union[Set, DataSource], desc: str=None, filter: Filter=None, aggFunc=None, limit=None):
        self._dataset_id = dataset_id
        self._schema = schema
        self._source = source
        self._desc = desc
        self._filter = filter
        self._aggFunc = aggFunc
        self._limit = limit

    def __str__(self):
        return f"{self.__class__.__name__}(dataset_id={self._dataset_id}, schema={self._schema}, desc={self._desc}, filter={str(self._filter)}, aggFunc={str(self._aggFunc)}, limit={str(self._limit)}, uid={self.universalIdentifier()})"

    def serialize(self):
        d = {"version": Set.SET_VERSION, 
             "dataset_id": self._dataset_id,
             "schema": self._schema.jsonSchema(),
             "source": self._source.serialize(),
             "desc": self._desc, 
             "filter": self._filter.serialize(),
             "aggFunc": None if self._aggFunc is None else self._aggFunc.serialize(),
             "limit": self._limit}

        return d

    def deserialize(inputObj):
        if inputObj["version"] != Set.SET_VERSION:
            raise Exception("Cannot deserialize Set because it is the wrong version")

        # TODO: I don't believe this would actually work for the DataSources,
        #       as inputObj["schema"] is a dict. not a Schema
        # deserialize source depending on whether it's a Set or DataSource
        source = None
        if "version" in inputObj["source"]:
            source = Set.deserialize(inputObj["source"])

        elif inputObj["source_type"] == "directory":
            source = DirectorySource(inputObj["schema"])

        elif inputObj["source_type"] == "file":
            source = FileSource(inputObj["schema"])

        # deserialize agg. function
        aggFuncStr = inputObj.get("aggFunc", None)
        aggFunc = None if aggFuncStr is None else AggregateFunction.deserialize(aggFuncStr)

        # deserialize limit
        limitStr = inputObj.get("limit", None)
        limit = None if limitStr is None else int(limitStr)

        return Set(dataset_id=inputObj["dataset_id"],
                   schema=inputObj["schema"].jsonSchema(), 
                   source=source, 
                   desc=inputObj["desc"], 
                   filter=Filter.deserialize(inputObj["filter"]),
                   aggFunc=aggFunc,
                   limit=limit)

    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        d = self.serialize()
        ordered = json.dumps(d, sort_keys=True)
        result = hashlib.sha256(ordered.encode()).hexdigest()
        return result

    def schema(self):
        """The Set's schema"""
        return self._schema

    def jsonSchema(self):
        """Return the JSON schema for this Set."""
        return self._schema.jsonSchema()

    def dumpSyntacticTree(self):
        """Return the syntactic tree of this Set."""
        if isinstance(self._source, DataSource):
            return (self, None)

        return (self, self._source.dumpSyntacticTree())

    ### TODO --> make sure it's consistent w/new ConcreteDataset in main
    def getLogicalTree(self) -> LogicalOperator:
        """Return the logical tree of operators on Sets."""
        # first, check to see if this set has previously been cached
        uid = self.universalIdentifier()
        if DataDirectory().hasCachedAnswer(uid):
            return CacheScan(self._schema, uid)

        # otherwise, if this Set's source is a DataSource
        if isinstance(self._source, DataSource):
            source_id = self._source.universalIdentifier()
            sourceSchema = Number if isinstance(self._source, MemorySource) else File
            if self._schema == sourceSchema:
                return BaseScan(self._schema, source_id)
            else:
                return ConvertScan(self._schema, BaseScan(sourceSchema, source_id), targetCacheId=uid)

        # if the Set's source is another Set, apply the appropriate scan to the Set
        if self._filter is not None:
            return FilteredScan(self._schema, self._source.getLogicalTree(), self._filter, targetCacheId=uid)
        elif self._aggFunc is not None:
            return ApplyAggregateFunction(self._schema, self._source.getLogicalTree(), self._aggFunc, targetCacheId=uid)
        elif self._limit is not None:
            return LimitScan(self._schema, self._source.getLogicalTree(), self._limit, targetCacheId=uid)
        elif not self._schema == self._source._schema:
            return ConvertScan(self._schema, self._source.getLogicalTree(), targetCacheId=uid)
        else:
            return self._source.getLogicalTree()


class Dataset(Set):
    """
    A Dataset is the intended abstraction for programmers to interact with when manipulating Sets.

    Users instantiate a Dataset by specifying a `source` that either points to a
    DataSource or an existing cached Set. Users can then perform computations on
    the Dataset in an imperative fashion by leveraging functions such as `filterByStr`,
    `convert`, `aggregate`, etc. Underneath the hood, each of these operations creates
    a new Set which is cached by the DataManager. As a result, the Sets define the
    lineage of computation on a Dataset, and this enables programmers to re-use
    previously cached computation by providing it as a `source` to some future Dataset.

    To construct a Dataset, users must provide three pieces of information. First, they must
    give the Dataset a unique name using the dataset_id. Second, they need to provide a source
    or source_id, depending on whether the Dataset takes its input from a raw source (e.g. a
    directory, an S3 prefix, etc.) or from another Dataset, respectively. Third, users must
    provide a Schema for the Dataset. This Schema will be enforced when the Dataset iterates
    over the source in its __iter__ method and constructs DataRecords.
    """
    def __init__(self, dataset_id: str, source: str, schema: Schema, desc: str=None):
        # convert source (str) -> source (Union[Set, DataSource])
        sourceSet = None
        if os.path.exists(source) and os.path.isdir(source):
            sourceSet = DirectorySource(source)
        elif os.path.exists(source) and os.path.isfile(source):
            sourceSet = FileSource(source)
        else:
            # TODO: this will fetch the appropriate source assuming it's a Dataset;
            #       can we make it possible for source to refer to the intermediate
            #       computation of a previous Dataset (i.e. a Set)?
            sourceSet = DataDirectory().getRegisteredDataset(source)

        super().__init__(schema, sourceSet, desc=desc, filter=None)
        self.dataset_id = dataset_id

    def deserialize(inputObj):
        # TODO: this deserialize operation will not work; I need to finish the deserialize impl. for Schema
        if inputObj["version"] != Set.SET_VERSION:
            raise Exception("Cannot deserialize Set because it is the wrong version")

        # deserialize source depending on whether it's a Set or DataSource
        source = None
        if "version" in inputObj["source"]:
            source = Set.deserialize(inputObj["source"])

        elif inputObj["source_type"] == "directory":
            source = DirectorySource(inputObj["schema"])

        elif inputObj["source_type"] == "file":
            source = FileSource(inputObj["schema"])

        return Dataset(inputObj["dataset_id"], source, inputObj["schema"], desc=inputObj["desc"]) 

    def filter(self, f: Filter, desc: str="Apply filter(s)") -> Dataset:
        """
        This function creates and returns a new Set. The newly created Set uses this Set
        as its source and applies the provided filter to it.
        """
        filteredSet = Set(self.dataset_id, self.schema(), self.source, desc=desc, filter=f)

        # TODO: should I update self._desc to reflect the filter operation?
        return Dataset(self.dataset_id, source=filteredSet, schema=self.schema(), desc=self._desc)

    def filterByStr(self, filterCondition: str, desc: str="Apply filter(s)") -> Dataset:
        """Add a filter to the Set. This filter will possibly restrict the items that are returned later."""
        f = Filter(filterCondition)

        return self.filter(f, desc)

    def convert(self, newSchema: Schema, desc: str="Convert to new schema") -> Dataset:
        """Convert the Set to a new schema."""
        convertedSet = Set(self.dataset_id, newSchema, self.source, desc=desc) # TODO: add conversion fcn. ?

        # TODO: should I update self._desc to reflect the conversion operation?
        return Dataset(self.dataset_id, source=convertedSet, schema=newSchema, desc=self._desc)

    def aggregate(self, aggFuncDesc: str) -> Dataset:
        """Apply an aggregate function to this set"""
        aggregatedSet = Set(self.dataset_id, Number, self.source, desc="Aggregate results", aggFunc=AggregateFunction(aggFuncDesc))

        # TODO: should I update self._desc to reflect the Agg. operation?
        return Dataset(self.dataset_id, source=aggregatedSet, schema=Number, desc=self._desc)
    
    def limit(self, n: int) -> Dataset:
        """Limit the set size to no more than n rows"""
        limitSet = Set(self.dataset_id, self.schema(), self.source, desc="LIMIT " + str(n), limit=n)

        # TODO: should I update self._desc to reflect the limit operation?
        return Dataset(self.dataset_id, source=limitSet, schema=self.schema(), desc=self._desc)
