import logging

from palimpzest.constants import Cardinality
from palimpzest.core.data.context import Context, TextFileContext
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.data.iter_dataset import (
    HTMLFileDataset,
    ImageFileDataset,
    IterDataset,
    MemoryDataset,
    PDFFileDataset,
    TextFileDataset,
    XLSFileDataset,
)
from palimpzest.policy import (
    MaxQuality,
    MaxQualityAtFixedCost,
    MaxQualityAtFixedTime,
    MinCost,
    MinCostAtFixedQuality,
    MinTime,
    MinTimeAtFixedQuality,
    PlanCost,
    Policy,
)
from palimpzest.query.processor.config import QueryProcessorConfig

# Initialize the root logger
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # constants
    "Cardinality",
    # core
    "Context",
    "TextFileContext",
    "Dataset",
    "IterDataset",
    "MemoryDataset",
    "HTMLFileDataset",
    "ImageFileDataset",
    "PDFFileDataset",
    "TextFileDataset",
    "XLSFileDataset",
    # policy
    "MaxQuality",
    "MaxQualityAtFixedCost",
    "MaxQualityAtFixedTime",
    "MinCost",
    "MinCostAtFixedQuality",
    "MinTime",
    "MinTimeAtFixedQuality",
    "PlanCost",
    "Policy",
    # query
    "QueryProcessorConfig",
    # sets
    "Dataset",
]
