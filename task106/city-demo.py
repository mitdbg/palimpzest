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




tripadvisor_location_ids = {
    "New York City": "60763",
    "Los Angeles": "32655",
    "Chicago": "35805",
    "Houston": "56003",
    "San Francisco": "60713",
    "Las Vegas": "45963",
    "Nashville": "55229",
    "Miami": "34438",
    "Atlanta": "60898",
    "Denver": "33388",
    "Louisville": "39604",
    "San Diego": "60750",
    "Dallas": "55711",
    "Seattle": "60878",
    "Phoenix": "31310",
    "Philadelphia": "60795",
    "San Antonio": "60956",
    "Honolulu": "60982",
    "New Orleans": "60864",
    "Orlando": "34515"
}

city_to_airport = { #maps city to airport code
    "New York City": "JFK",   # John F. Kennedy International Airport
    "Los Angeles": "LAX",     # Los Angeles International Airport
    "Chicago": "ORD",         # O'Hare International Airport
    "Houston": "IAH",         # George Bush Intercontinental Airport
    "San Francisco": "SFO",   # San Francisco International Airport
    "Las Vegas": "LAS",       # Harry Reid International Airport
    "Nashville": "BNA",       # Nashville International Airport
    "Miami": "MIA",           # Miami International Airport
    "Atlanta": "ATL",         # Hartsfield-Jackson Atlanta International Airport
    "Denver": "DEN",          # Denver International Airport
    "Louisville": "SDF",      # Louisville Muhammad Ali International Airport
    "San Diego": "SAN",       # San Diego International Airport
    "Dallas": "DFW",          # Dallas/Fort Worth International Airport
    "Seattle": "SEA",         # Seattle-Tacoma International Airport
    "Phoenix": "PHX",         # Phoenix Sky Harbor International Airport
    "Philadelphia": "PHL",    # Philadelphia International Airport
    "San Antonio": "SAT",     # San Antonio International Airport
    "Honolulu": "HNL",        # Daniel K. Inouye International Airport
    "New Orleans": "MSY",     # Louis Armstrong New Orleans International Airport
    "Orlando": "MCO"          # Orlando International Airport
}


class City(Schema):
    name = StringField(desc="The name of the city")
    weather = StringField(desc="Weather information of the city")
    # flight_prices = NumericField(desc="The prices of the flights to the city")
    hotel_prices = NumericField(desc="The prices of the hotels in the city")
    reviews = StringField(desc="String of 5 latest reviews of city")


class CityRating(Schema):
    rating = NumericField(desc="The calculated rating of a city based off its characteristics on a 0-10 scale")


def weather_api(city): # should work
    api_key = os.getenv("WEATHER_STACK_KEY")  # Replace with your actual API key
    base_url = "https://api.weatherstack.com/current?access_key={" + api_key + "}"

    params = {
        # "access_key": api_key,
        "query": city
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if "error" in data:
            return f"Error: {data['error']['info']}"

        # Extract relevant weather data
        weather_info = {
            "temperature": data["current"]["temperature"],
            "description": data["current"]["weather_descriptions"][0],
            "humidity": data["current"]["humidity"],
            "wind_speed": data["current"]["wind_speed"]
        }
        return weather_info

    except Exception as e:
        return f"API Error: {str(e)}"

# def flight_api(city): #not sure
#     # import requests

#     url = "https://compare-flight-prices.p.rapidapi.com/GetPricesAPI/GetPrices.aspx"

#     headers = {
#         "x-rapidapi-key": "225d8d56bemsha4362cd3d01c951p1ef7b7jsnc8edeaa460a5",
#         "x-rapidapi-host": "compare-flight-prices.p.rapidapi.com",
#         "from": "BOS",  # Replace with your actual departure airport code
#         "to": city_to_airport[city],  # Destination city (Needs to be an airport code, e.g., "LAX" for Los Angeles)
#         "date1": "2024-04-01",  # Replace with actual date (YYYY-MM-DD)
#         "isOneWay": "true",
#         "currency": "USD",
#         "languageCode": "en"
#     }


#     params = {
        
#     }

#     response = requests.get(url, headers=headers)

#     return response.json()
    # pass

def hotel_api(city): #should work
    import requests

    key = os.getenv("TRIP_ADVISORY_KEY")
    
    url = None
    if len(city.split()) == 2:
        url = f"https://api.content.tripadvisor.com/api/v1/location/search?key={key}&searchQuery={city.split()[0]}%20{city.split()[1]}&language=en"
    else:
        url = f"https://api.content.tripadvisor.com/api/v1/location/search?key={key}&searchQuery={city}&language=en"




   

    temp = "https://api.content.tripadvisor.com/api/v1/location/search?language=en"


    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)
    return response.json()



def reviews_api(city): #should work
     #not sure if it works
    key = os.getenv("TRIP_ADVISORY_KEY")
    city_id = tripadvisor_location_ids.get(city)
    url = f"https://api.content.tripadvisor.com/api/v1/location/{city_id}/reviews?key={key}&language=en"


    




    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)

    return response.text

class CityDataReader(pz.DataReader):
    def __init__(self, potential_cities):
        super().__init__(City)
        self.potential_cities = potential_cities
        # self.listings_dir = listings_dir
        # self.listings = sorted([dir for dir in os.listdir(self.listings_dir) if "listing" in dir])

    def __len__(self):
        return len(self.potential_cities)

    def __getitem__(self, idx: int):
        # fetch listing
        city = self.potential_cities[idx]
        weather = weather_api(city)
        # flight_prices = flight_api(city)
        hotel_prices = hotel_api(city)
        reviews = reviews_api(city)


        output = {}
        output['name'] = city
        output['weather'] = weather
        # output['flight prices'] = flight_prices
        output['reviews'] = reviews
        output['hotel_prices'] = hotel_prices
        return output






def is_good_vacation (active_ds):
    return active_ds['rating'] >= 1.0

if __name__ == "__main__":
    ##MY CODE BEGINS
    cities = ['New York City', "Los Angeles", "Chicago", "Houston", "San Francisco", 
                                 "Las Vegas", "Nashville", "Miami", "Atlanta", "Denver",
                                   "Louisville", "San Diego", "Dallas", "Seattle", "Pheonix",
                                   "Philidelphia", "San Antonio", "Honolulu", "New Orleans", "Orlando"]
    ds = Dataset(CityDataReader(cities)) # list of city names
    ds = ds.sem_add_columns(CityRating)

    #do i need this line
    ds = ds.filter(is_good_vacation, depends_on="rating")

    config = QueryProcessorConfig(
        cache=False, #changed from nocache=True
        verbose=True,
        policy=MaxQuality(),
        execution_strategy="parallel",
        available_models=[Model.MIXTRAL, Model.GPT_4o_MINI, Model.GPT_4o_MINI_V]
    )
    data_record_collection = ds.run(config)
    print(data_record_collection.to_df())

   