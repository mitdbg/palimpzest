from palimpzest.constants import Model
from palimpzest.tools.profiler import Profiler
from pathlib import Path
from openai import OpenAI
import google.generativeai as genai

from typing import Union
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import base64
import requests
import time
import io
from PIL import Image

# retry LLM executions 2^x * (multiplier) for up to 10 seconds and at most 4 times
RETRY_MULTIPLIER = 2
RETRY_MAX_SECS = 10
RETRY_MAX_ATTEMPTS = 1

def log_attempt_number(retry_state):
    """return the result of the last call attempt"""
    print(f"Retrying: {retry_state.attempt_number}...")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def decode_image(base64_string):
    return base64.b64decode(base64_string)


PROMPT = """You are a image analysis bot. Analyze the supplied image and return a complete description of its contents including a description of the general scene as well as a list of all of the people, animals, and objects you see and what they are doing."""

def make_payload(base64_image):
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": PROMPT
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 4000
    }
    return payload



@retry(
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    after=log_attempt_number,
)
def describe_image(model_name:str, image_b64: str) -> Union[str,str]:
    """Method essentially based on do_image_analysis from the original codebase.
    Key differences:
    1. API key is not passed, rather it is fetched from the environment variables based on the llm service chosen
    2. The API is not called using REST but using the proprietary python libraries
    """
    if model_name == Model.GPT_4V.value:
      api_key = os.environ["OPENAI_API_KEY"]
      client = OpenAI(api_key=api_key)
      payload = make_payload(image_b64)
      start_time = time.time()
      completion = client.chat.completions.create(**payload)
      end_time = time.time()
      candidate = completion.choices[-1]
      content_str = candidate.message.content
      finish_reason = candidate.finish_reason
      usage = completion.usage
    elif model_name == Model.GEMINI_1V.value:
      api_key = os.environ["GOOGLE_API_KEY"]
      genai.configure(api_key=api_key)
      client = genai.GenerativeModel('gemini-pro-vision')
      img = Image.open(io.BytesIO(decode_image(image_b64)))
      start_time = time.time()
      response = client.generate_content([PROMPT,img])
      end_time = time.time()
      candidate = response.candidates[-1]
      content_str = candidate.content.parts[0].text
      finish_reason = candidate.finish_reason
      usage = 0
    else:
      raise ValueError(f"Unknown model name: {model_name}")

    stats = {}
    if Profiler.profiling_on():
      stats['api_call_duration'] = end_time - start_time
      stats['prompt'] = PROMPT
      stats['usage'] = usage
      stats['finish_reason'] = finish_reason

    return content_str, stats