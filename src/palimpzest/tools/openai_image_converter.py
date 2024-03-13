from palimpzest.tools.profiler import Profiler

from tenacity import retry, stop_after_attempt, wait_exponential

import base64
import requests
import time

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

def make_payload(base64_image):
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": '''
              You are a image analysis bot.  Analyze the supplied image and return a complete description of its contents including 
              a description of the general scene as well as a list of all of the people, animals, and objects you see and what they are doing.
              '''
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
def do_image_analysis(api_key, image_bytes):
    # Getting the base64 string
    base64_image = image_bytes

    payload = make_payload(base64_image)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    start_time = time.time()
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    end_time = time.time()

    # Your JSON blob
    json_blob = response.json()

    # Accessing the content
    content_str = json_blob['choices'][-1]['message']['content']

    # get usage data if profiling is on
    stats = {}
    if Profiler.profiling_on():
        stats['api_call_duration'] = end_time - start_time
        stats['prompt'] = payload['messages'][0]["content"][0]["text"]
        stats['usage'] = json_blob['usage']
        stats['finish_reason'] = json_blob['choices'][-1]['finish_reason']

    return content_str, stats
