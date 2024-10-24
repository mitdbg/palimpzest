#####################################################
#
#####################################################
# Description: This file contains the functions that are from ASKEM skema tools at endpoints:
# https://api.askem.lum.ai/docs

import requests


def equations_to_latex(image_content):
    url = "https://api.askem.lum.ai/workflows/images/equations-to-latex"
    files = {
        "data": image_content,
    }
    r = requests.post(url, files=files)
    return r.text


def equations_to_latex_base64(image_content):
    url = "https://api.askem.lum.ai/workflows/images/base64/equations-to-latex"
    r = requests.post(url, data=image_content)
    return r.text
