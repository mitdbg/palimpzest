
import requests

import os

api_key = os.getenv("WEATHER_STACK_KEY")
url = "https://api.weatherstack.com/current?access_key=c528fa00a3c004942381a5aeada60ca4"

querystring = {"query":"New Delhi"}

response = requests.get(url, params=querystring)

print(response.json())