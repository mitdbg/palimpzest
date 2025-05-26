# class Bike:
#     def __init__(self):

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



class Shoe(Schema):
    description = StringField(desc="Description of the shoe and its quality")
    attached_images = ListField(
        element_type=ImageFilepathField,
        desc="A list of the attached images of the shoe",
    )

    # brand = StringField(desc="The brand of the shoe")
    # price = NumericField(desc='The cost of the sneakers in US Dollars')
    
    # location = StringField(desc="City/Town/Region where to pickup/receive the shoes once ordered")
   
    
    # application = StringField(desc='application used to send the send the message (ex. iMessage, gmail, instagram etc.)')


    # attached_images = ListField(
    #     element_type=ImageFilepathField,
    #     desc="A list of the attached images of the shoe",
    # )

    # time_passed = NumericField(desc = 'minutes passed since the message has come through')
    # size = NumericField(desc="Size of the shoes")

    






class ShoeInformation(Schema):
    brand = StringField(desc="The brand of the shoe")
    price = NumericField(desc='The cost of the sneakers in US Dollars')
    

    time_passed = NumericField(desc = 'days passed since the offer has come through')
    size = NumericField(desc="Size of the shoes")
    quality = NumericField(desc="the quality of the shoe based off solely the image and not the description. On a 0-100 scale")



    rating = NumericField(desc="A value quantitively expressing the value of the sneakers based off of all the information that the user is looking for. He is looking for size 10 sneakers. This rating is impacted by all factors including the price. Scaled on a 0-10 scale")


class ShoeDataReader(pz.DataReader):
    def __init__(self, shoes_dir):
        super().__init__(Shoe)
        self.shoes_dir = shoes_dir

        # self.shoes = sorted([dir for dir in os.listdir(self.shoes_dir) if "shoe" in dir])

        # self.listings_dir = listings_dir
        # self.listings = sorted([dir for dir in os.listdir(self.listings_dir) if "listing" in dir])

    def __len__(self):
        return len(self.shoes_dir)

    def __getitem__(self, idx: int):
        # fetch listing
       
        # output = {}
        # for i in sneaker:
        #     output[i] = sneaker[i]
        
        shoe = self.shoes_dir[idx]
        return shoe.copy()


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





def good_deal (shoe_ds):
    # print(active_ds)
    if shoe_ds is None or shoe_ds['rating'] is None:
        return False
    return shoe_ds['rating'] >= 3

if __name__ == "__main__":
    ##MY CODE BEGINS

    

    shoes = []
    # try:
    with os.scandir('/Users/barilebari/Desktop/UROP2025/palimpzest/testdemos/sneakers-info') as entries:
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
                    
                    shoes.append(temp)
    # except Exception as e:
    #      print(f"An error occurred: {e}")



    # print(shoes)
    
    ds = Dataset(ShoeDataReader(shoes)) # list of city names
    ds = ds.sem_add_columns(ShoeInformation)

    #do i need this line
    ds = ds.filter(good_deal, depends_on="rating")

    config = QueryProcessorConfig(
        cache=False, #changed from nocache=True
        verbose=True,
        policy=MaxQuality(),
        execution_strategy="parallel",
        available_models=[Model.MIXTRAL, Model.GPT_4o_MINI, Model.GPT_4o_MINI_V]
    )
    data_record_collection = ds.run(config)
    print(data_record_collection.to_df())