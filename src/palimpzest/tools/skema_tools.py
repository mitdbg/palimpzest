#####################################################
#
#####################################################
# Description: This file contains the functions that are from ASKEM skema tools at endpoints:
# https://api.askem.lum.ai/docs
import base64
import requests
import time


def equations_to_latex(image_content):
    url = "https://api.askem.lum.ai/workflows/images/equations-to-latex"
    files = {
      "data": image_content,
    }
    r = requests.post(url, files=files)
    return r.text

def equations_to_latex_base64(image_content):
    url = "https://api.askem.lum.ai/workflows/images/base64/equations-to-latex"
    start_time = time.time()
    r = requests.post(url, data=image_content)
    return r.text
