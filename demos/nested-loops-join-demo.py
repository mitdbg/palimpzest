from pydantic import BaseModel, Field
import pandas as pd
import palimpzest as pz
from palimpzest.query.operators.join import NestedLoopsJoin

# datasets of movie reviews and actor descriptions
movie_reviews = [
    {"review": "Inception is a mind-bending thriller that blurs the lines between dreams and reality. A must-watch!"},
    {"review": "The Devil Wears Prada is a sharp and witty look into the fashion industry, with standout performances."},
    {"review": "Titanic is a heartbreaking love story set against the backdrop of a historic tragedy. Truly moving."},
    {"review": "The Dark Knight redefined the superhero genre with its intense action and complex characters."},
    {"review": "Training Day is a gritty crime drama that keeps you on the edge of your seat from start to finish."},
]
actor_descriptions = [
    {"actor": "Tom Cruise is an American actor and producer known for his roles in action films such as 'Top Gun' and the 'Mission: Impossible' series."},
    {"actor": "Meryl Streep is an acclaimed American actress recognized for her versatility and roles in films like 'The Devil Wears Prada' and 'Sophie's Choice'."},
    {"actor": "Leonardo DiCaprio is an American actor and film producer known for his performances in 'Titanic', 'Inception', and 'The Revenant'."},
    {"actor": "Scarlett Johansson is an American actress and singer, famous for her roles in 'Lost in Translation' and as Black Widow in the Marvel Cinematic Universe."},
    {"actor": "Denzel Washington is an American actor and director known for his powerful performances in films like 'Training Day' and 'Malcolm X'."},
]

# create DataRecords for movie reviews
movie_ds = pz.MemoryDataset(id="movies", vals=movie_reviews)
output = movie_ds.run()
left_candidates = [dr for dr in output]

# create DataRecords for actor descriptions
actor_ds = pz.MemoryDataset(id="actors", vals=actor_descriptions)
output = actor_ds.run()
right_candidates = [dr for dr in output]

# execute semantic join with your operator
class JoinSchema(BaseModel):
    review: str = Field(description="A movie review")
    actor: str = Field(description="A sentence about an actor")

join_op = NestedLoopsJoin(
    input_schema=JoinSchema,
    output_schema=JoinSchema,
    model=pz.Model.GPT_4o_MINI,
    condition="The actor appears in the movie being reviewed",
    logical_op_id="abc123"
)
output, _ = join_op(left_candidates, right_candidates)
output_df = pd.DataFrame([dr.to_dict() for dr in output if dr._passed_operator])
print(output_df)