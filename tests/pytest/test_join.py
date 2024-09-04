from palimpzest.constants import Model
from palimpzest.cost_model import CostModel
from palimpzest.optimizer import LogicalExpression, Group, Optimizer
from palimpzest.operators import *
from palimpzest.policy import *
import palimpzest as pz
import pytest

"""
SELECT * FROM title AS t JOIN movie_info_idx AS mi_idx
ON t.id = mi_idx.movie_id
WHERE (t.year > 2000 AND mi_idx.score > ’7.0’)
OR (t.year > 1980 AND mi_idx.score > ’8.0’)
"""

class Movie(pz.TextFile):
    """Represents a movie extracted from a text file"""
    movie_id = pz.Field(desc="The unique identifier of the movie", required=True)
    movie_title = pz.Field(desc="The natural language title of the movie", required=True)
    movie_year = pz.Field(desc="The year of the movie", required=True)

class MovieRating(pz.TextFile):
    """Represents the rating of a movie from a website like IMDB"""
    movie_id = pz.Field(desc="The unique identifier of the movie", required=True)
    rating = pz.Field(desc="The rating of the movie from 1 to 10", required=True)
    source = pz.Field(desc="The source of the rating", required=False)


movies = pz.Dataset("movies-title", schema=Movie)
ratings = pz.Dataset("movies-rating", schema = MovieRating)

movierating = ratings.join(movies, on="movie_id")
movierating = movierating.filter("The movie is rated more than 8 and from the 2010s")
output = movierating

class TestJoinOptimizer:

    def test_basic_functionality(self, enron_eval_tiny):
        policy = MinCost()
        cost_model = CostModel(enron_eval_tiny, sample_execution_data=[])
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            allow_code_synth=False,
            allow_token_reduction=False,
            no_cache=True,
            verbose=True,
            available_models=[Model.GPT_4, Model.GPT_3_5],
        )
        physical_plans = optimizer.optimize(output)
        physical_plan = physical_plans[0]
        print(physical_plan)
        assert isinstance(physical_plan[0], MarshalAndScanDataOp)

        engine = pz.SequentialSingleThreadNoSentinelExecution

        records, plan, stats =  pz.Execute(output,
                        policy = policy,
                        nocache=True,
                        allow_code_synth=False,
                        allow_token_reduction=False,
                        execution_engine=engine)

        for record in records:
            print(record.title, record.year, record.score)