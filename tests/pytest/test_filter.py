"""This script contains tests for physical operators for filter."""

import os

from pydantic import BaseModel, Field

from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineFilter
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsFilter
from palimpzest.query.operators.rag import RAGFilter
from palimpzest.query.operators.split import SplitFilter

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


class SimpleSchema(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")

class SimpleSchemaTwo(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    height: int | float = Field(description="The height of the person in cm")
