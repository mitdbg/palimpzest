#####################################################
#
#####################################################
# Description: This file contains the functions that are from ASKEM skema tools at endpoints:
# https://api.askem.lum.ai/docs
import base64
import json
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


if __name__ == "__main__":
    img_bytes = open("../../../testdata/equation-tiny/dEdt.png", "rb").read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    r_b64 = equations_to_latex_base64(img_b64)
    print("b64 api test: ", r_b64)


    r = equations_to_latex(img_bytes)
    print("binary api test: ", r)
