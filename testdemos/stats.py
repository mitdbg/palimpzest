import argparse
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import palimpzest as pz
from palimpzest.core.data.datareaders import DataReader
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BooleanField, ListField, ImageFilepathField, NumericField, StringField
from palimpzest.core.lib.schemas import Schema
from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.sets import Dataset
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.demo_helpers import print_table


import requests
import http.client


import cv2
import os
import numpy as np

def extract_10_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle case where video has fewer than 10 frames
    num_frames_to_extract = min(10, total_frames)

    # Compute frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames_to_extract, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        # Set the video position to the selected frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        success, frame = video.read()

        if success:
            # Save the frame with a zero-padded filename
            frame_filename = os.path.join(output_folder, f"frame_{i:02d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
        else:
            print(f"Failed to read frame at index {frame_idx}")

    # Release the video resource
    video.release()
    print("Done extracting 10 frames.")

# 🧪 Example usage:




passer_rating_context = ''' The NFL passer rating is a statistic that assesses a quarterback's passing performance, considering completion percentage, yards per attempt, touchdown percentage, and interception percentage. It's calculated by taking four individual components, each scaled to a value between 0 and 2.375, adding them together, dividing by 6, and then multiplying by 100. A score of 1.0 for each component is considered average. 
Here's a breakdown of how each component is calculated:
Completion Percentage: (Completions / Attempts - 0.3) * 5.
Yards per Attempt: (Yards / Attempts - 3) * 0.25.
Touchdown Percentage: (Touchdowns / Attempts) * 20.
Interception Percentage: 2.375 - ((Interceptions / Attempts) * 25). 
Important Notes:
No component can be less than zero or greater than 2.375. 
An average quarterback is expected to complete 50% of their passes, average 7 yards per attempt, throw 5% of their passes for touchdowns, and throw an interception 5.5% of the time. 
The highest possible passer rating is 158.3, which is achieved when all components are at their maximum value of 2.375'''


production_rating_description = '''
1-10 scale

the more games played the better, the more passing yards, passing touchdowns, completions, passing long, rushing yards, rushing touchdowns rushig long the better
the less interceptions, fumbles and fumbles lost the better 

the more pass yards per game, touchdowns per game, touchdown-int, topuchdown-turnover, passer rating, complryions per game yards per attemot and yards per completion the better

the less interceptions per game the better

more interceptions than touchdowns or more turnovers than touchdowns is really really bad (below 4)

'''

class Player(Schema):
    


    description = StringField(desc='Input description of the player')


    film_images = ListField(
        element_type=ImageFilepathField,
        desc="A list of the attached images from the film of the player",
    )

class PlayerAdvanced (Schema):

    name = StringField(desc='Name of the Player as String')
    games_played = NumericField(desc="number of games played")
    passing_yards = NumericField(desc='Number of passing yards')
    passing_touchdowns = NumericField(desc='Number of passing Touchdowns')
    passing_completions = NumericField(desc='Number of passing completions')
    passing_attempts = NumericField(desc= "number of passing attempts")
    passing_long = NumericField(desc='Longest passing play')
    passing_interceptions = NumericField (desc='Interceptions')
    rushing_attempts = NumericField(desc='Number of rushing attempts')
    rushing_yards = NumericField (desc= 'Numner of Rushing yards')
    rushing_touchdowns = NumericField (desc='Number of rushing touchdowns')
    rushing_long = NumericField(desc='Rushing Longest Play')
    fumbles = NumericField(desc='Fumbles')
    fumbles_lost = NumericField(desc='Fumbles Lost')
    height = NumericField(desc='Height in inches')
    weight = NumericField(desc='Weight in Pounds')


    pass_yards_per_game = NumericField (desc="passing yards per game (pass yards divided by game)")
    pass_touchdowns_per_game = NumericField (desc="passing yards per game (pass yards divided by game)")
    pass_completions_per_game = NumericField(desc="pass completions per game")
    pass_interceptions_per_game = NumericField(desc="interceptions per game")
    pass_completion_percentage = NumericField(desc='pass completion percentage based off of the attempts')
    pass_yards_per_attempt = NumericField(desc="passing yards per attempt")
    pass_yards_per_completion = NumericField(desc="passing yards per completion")
    touchdown_interception_ratio = NumericField(desc='Passing touchdown to passing interception ratio')
    
    passer_rating = NumericField(desc=passer_rating_context)


    rushing_yards_per_carry = NumericField(desc='Rushing yards per attempt')
    rushing_yards_per_game = NumericField(desc='Rushing yards per game')
    rush_attempts_per_game = NumericField(desc='rushing attempts per game')


    total_yards_per_game = NumericField(desc='total rush and pass yards per game')
    touchdowns_per_game = NumericField(desc='total Touchdowns per game (rush and pass)')
    turnovers_per_game  = NumericField(desc='total turnovers per game (fumbles lody and interceptions)')
    total_touchdown_to_turnover_ratio = NumericField(desc='total touchdown (pass and rush) to turnover(fumbles lost plus interceptions) ratio')


    physical_rating = NumericField (desc = "1-10 rating of a player's weight and height and longest play")
    production_rating = NumericField(desc=production_rating_description)
    film_rating = NumericField(desc="rating based off of the film images and quality of the play(s) on a 1-10 scale. Touchdowns and big plays shouild be worth a lot! turnovers should be worth very little. Avewrage plays should be average. Should only be based off of the images and not the stats. Crowd reaction and celebration should play a positive factor. It is based off of the quarterback's ability")
    overall_rating = NumericField(desc='1-10 rating of all attributes')

    summary = StringField(desc='one sentence descriptn of QB based off of all stats and ratings')


class PlayerDataReader(pz.DataReader):
    def __init__(self, players):
        super().__init__(Player)
        self.players = players

        # self.shoes = sorted([dir for dir in os.listdir(self.shoes_dir) if "shoe" in dir])

        # self.listings_dir = listings_dir
        # self.listings = sorted([dir for dir in os.listdir(self.listings_dir) if "listing" in dir])

    def __len__(self):
        return len(self.players)

    def __getitem__(self, idx: int):
        # fetch listing
       
        # output = {}
        # for i in sneaker:
        #     output[i] = sneaker[i]
        
        player = self.players[idx]
        return player.copy()


        # dr = DataRecord(self.schema, idx)
        # dr.shoe = shoe
        # dr.image_filepaths = []
        # shoe_dir = os.path.join(self.shoes, shoe)
        # for file in os.listdir(shoe_dir):
        #     if file.endswith(".txt"):
        #         with open(os.path.join(shoe_dir, file), "rb") as f:
        #             dr.text_content = f.read().decode("utf-8")
        #     elif file.endswith(".png"):
        #         dr.image_filepaths.append(os.path.join(shoe_dir, file))

        # return dr





def is_star_player (player_ds):
    # print(active_ds)
    if player_ds is None or player_ds['overall_rating'] is None:
        return False
    return player_ds['overall_rating'] >= 1

if __name__ == "__main__":
    ##MY CODE BEGINS

    

    # players =quarterbacks = [
    # {  # Elite
    #     'name': 'Patrick Blaze',
    #     'games_played': 17,
    #     'passing_yards': 5100,
    #     'passing_touchdowns': 45,
    #     'passing_completions': 430,
    #     'passing_attempts': 620,
    #     'passing_long': 80,
    #     'passing_interceptions': 7,
    #     'rushing_attempts': 60,
    #     'rushing_yards': 400,
    #     'rushing_touchdowns': 4,
    #     'rushing_long': 30,
    #     'fumbles': 4,
    #     'fumbles_lost': 1,
    #     'height': 76,
    #     'weight': 225
    # },
    # {  # Average
    #     'name': 'Jake Turner',
    #     'games_played': 16,
    #     'passing_yards': 3600,
    #     'passing_touchdowns': 23,
    #     'passing_completions': 320,
    #     'passing_attempts': 510,
    #     'passing_long': 65,
    #     'passing_interceptions': 11,
    #     'rushing_attempts': 35,
    #     'rushing_yards': 200,
    #     'rushing_touchdowns': 2,
    #     'rushing_long': 18,
    #     'fumbles': 6,
    #     'fumbles_lost': 3,
    #     'height': 74,
    #     'weight': 218
    # },
    # {  # Elite
    #     'name': 'Derrick Cole',
    #     'games_played': 17,
    #     'passing_yards': 4800,
    #     'passing_touchdowns': 38,
    #     'passing_completions': 405,
    #     'passing_attempts': 600,
    #     'passing_long': 75,
    #     'passing_interceptions': 8,
    #     'rushing_attempts': 50,
    #     'rushing_yards': 320,
    #     'rushing_touchdowns': 5,
    #     'rushing_long': 25,
    #     'fumbles': 5,
    #     'fumbles_lost': 2,
    #     'height': 75,
    #     'weight': 230
    # },
    # {  # Bad
    #     'name': 'Matt Griggs',
    #     'games_played': 13,
    #     'passing_yards': 2100,
    #     'passing_touchdowns': 9,
    #     'passing_completions': 195,
    #     'passing_attempts': 390,
    #     'passing_long': 48,
    #     'passing_interceptions': 15,
    #     'rushing_attempts': 20,
    #     'rushing_yards': 70,
    #     'rushing_touchdowns': 0,
    #     'rushing_long': 9,
    #     'fumbles': 10,
    #     'fumbles_lost': 5,
    #     'height': 73,
    #     'weight': 225
    # },
    # {  # Average
    #     'name': 'Connor Blake',
    #     'games_played': 15,
    #     'passing_yards': 3400,
    #     'passing_touchdowns': 21,
    #     'passing_completions': 310,
    #     'passing_attempts': 495,
    #     'passing_long': 60,
    #     'passing_interceptions': 12,
    #     'rushing_attempts': 42,
    #     'rushing_yards': 180,
    #     'rushing_touchdowns': 2,
    #     'rushing_long': 17,
    #     'fumbles': 7,
    #     'fumbles_lost': 2,
    #     'height': 74,
    #     'weight': 220
    # },
    # {  # Elite
    #     'name': 'Malik Stone',
    #     'games_played': 17,
    #     'passing_yards': 5000,
    #     'passing_touchdowns': 42,
    #     'passing_completions': 425,
    #     'passing_attempts': 610,
    #     'passing_long': 82,
    #     'passing_interceptions': 6,
    #     'rushing_attempts': 55,
    #     'rushing_yards': 370,
    #     'rushing_touchdowns': 6,
    #     'rushing_long': 34,
    #     'fumbles': 3,
    #     'fumbles_lost': 1,
    #     'height': 76,
    #     'weight': 228
    # },
    # {  # Bad
    #     'name': 'Kyle Darnell',
    #     'games_played': 12,
    #     'passing_yards': 1900,
    #     'passing_touchdowns': 8,
    #     'passing_completions': 180,
    #     'passing_attempts': 360,
    #     'passing_long': 40,
    #     'passing_interceptions': 17,
    #     'rushing_attempts': 18,
    #     'rushing_yards': 60,
    #     'rushing_touchdowns': 0,
    #     'rushing_long': 6,
    #     'fumbles': 11,
    #     'fumbles_lost': 6,
    #     'height': 72,
    #     'weight': 215
    # },
    # {  # Average
    #     'name': 'Ryan Beck',
    #     'games_played': 16,
    #     'passing_yards': 3500,
    #     'passing_touchdowns': 22,
    #     'passing_completions': 325,
    #     'passing_attempts': 500,
    #     'passing_long': 62,
    #     'passing_interceptions': 13,
    #     'rushing_attempts': 38,
    #     'rushing_yards': 190,
    #     'rushing_touchdowns': 1,
    #     'rushing_long': 21,
    #     'fumbles': 6,
    #     'fumbles_lost': 2,
    #     'height': 75,
    #     'weight': 222
    # },
    # {  # Bad
    #     'name': 'Trevor Hodge',
    #     'games_played': 14,
    #     'passing_yards': 2200,
    #     'passing_touchdowns': 10,
    #     'passing_completions': 200,
    #     'passing_attempts': 400,
    #     'passing_long': 49,
    #     'passing_interceptions': 14,
    #     'rushing_attempts': 22,
    #     'rushing_yards': 75,
    #     'rushing_touchdowns': 1,
    #     'rushing_long': 10,
    #     'fumbles': 9,
    #     'fumbles_lost': 4,
    #     'height': 73,
    #     'weight': 219
    # },
    # {  # Average
    #     'name': 'Nick Harrow',
    #     'games_played': 17,
    #     'passing_yards': 3700,
    #     'passing_touchdowns': 24,
    #     'passing_completions': 330,
    #     'passing_attempts': 520,
    #     'passing_long': 67,
    #     'passing_interceptions': 10,
    #     'rushing_attempts': 40,
    #     'rushing_yards': 210,
    #     'rushing_touchdowns': 2,
    #     'rushing_long': 20,
    #     'fumbles': 5,
    #     'fumbles_lost': 2,
    #     'height': 76,
    #     'weight': 224
    # }
    # ]

    extract_10_frames("/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/plays/play1.mp4", "/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/quarterback-sample/qb1")
    extract_10_frames("/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/plays/play2.mp4", "/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/quarterback-sample/qb2")
    extract_10_frames("/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/plays/play3.mp4", "/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/quarterback-sample/qb3")
    extract_10_frames("/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/plays/play4.mp4", "/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/quarterback-sample/qb4")




    players = []

    with os.scandir('/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/quarterback-sample') as entries:
            for entry in entries:
                if entry.is_dir():
                    temp = {}
                    with os.scandir(entry) as sub_entries:
                        for file in sub_entries:
                            # print(file.path)
                            if file.path.endswith(".txt"):
                                with open(file, "r") as file:
                                    content = file.read()
                                    temp['description'] = content
                            elif file.path.endswith(".png"):
                                if temp.get('attached_images') is None:
                                    temp['attached_images'] = []
                                temp['attached_images'].append(os.path.join(entry, file))
                    
                    players.append(temp)

    # try:
    # with os.scandir('/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/sneakers-info') as entries:
    #         for entry in entries:
    #             if entry.is_dir():
    #                 temp = {}
    #                 with os.scandir(entry) as sub_entries:
    #                     for file in sub_entries:
    #                         # print(file.path)
    #                         if file.path.endswith(".txt"):
    #                             with open(file, "r") as file:
    #                                 content = file.read()
    #                                 temp['description'] = content
    #                         elif file.path.endswith(".png"):
    #                             if temp.get('attached_images') is None:
    #                                 temp['attached_images'] = []
    #                             temp['attached_images'].append(os.path.join(entry, file))
                    
    #                 shoes.append(temp)
    # except Exception as e:
    #      print(f"An error occurred: {e}")



    # print(shoes)
    
    ds = Dataset(PlayerDataReader(players)) # list of city names
    ds = ds.sem_add_columns(PlayerAdvanced)

    
    # ds = ds.filter(is_star_player, depends_on="overall_rating")

    config = QueryProcessorConfig(
        cache=False, #changed from nocache=True
        verbose=True,
        policy=MaxQuality(),
        execution_strategy="parallel",
        available_models=[Model.MIXTRAL, Model.GPT_4o_MINI, Model.GPT_4o_MINI_V]
    )
    data_record_collection = ds.run(config)
    cols_needed = ['name', 'overall_rating', 'production_rating', 'physical_rating', 'film_rating', 'summary']


    # print(data_record_collection.to_df(cols_needed))
    # print(data_record_collection.to_df(cols_needed)['description'].iloc[2])
    data_record_collection.to_df()[cols_needed[:]].iloc[:].to_csv("output.csv")