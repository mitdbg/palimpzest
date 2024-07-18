import palimpzest as pz
from palimpzest.corelib import URL, WebPage, Download
from palimpzest.elements import DataRecord, Schema

import requests
from requests_html import HTMLSession # for downloading JavaScript content
import datetime
from bs4 import BeautifulSoup

# TODO: I think this might all go away in favor of simply having users provide python functions / lambdas
#       the way we did for Filter(s) in the real-estate workloads

#
# REMIND: I don't love the syntax of UDFs right now. 
# It's too verbose and clumsy.
#

##
# The class for user-defined functions
#
class UserFunctionSingletonMeta(type):
    """Functions are always singletons"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class UserFunction(metaclass=UserFunctionSingletonMeta):
    def __init__(self, udfid: str, inputSchema: Schema, outputSchema: Schema):
        self.udfid = udfid
        self.inputSchema = inputSchema
        self.outputSchema = outputSchema

    def map(self, data):
        raise Exception("Not implemented")


class DownloadHTMLFunction(UserFunction):
    """DownloadHTMLFunction downloads the HTML content of a web page."""
    def __init__(self):
        super().__init__("localfunction:webgetter", URL, WebPage)

    def html_to_text_with_links(self, html):
        # Parse the HTML content
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all hyperlink tags
        for a in soup.find_all('a'):
            # Check if the hyperlink tag has an 'href' attribute
            if a.has_attr('href'):
                # Replace the hyperlink with its text and URL in parentheses
                a.replace_with(f"{a.text} ({a['href']})")
        
        # Extract text from the modified HTML
        text = soup.get_text(separator='\n', strip=True)        
        return text

    def get_page_text(self, url):
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        }

        session = HTMLSession()
        response = session.get(url, headers=headers)
        return response.text

    # Someday we should introduce an abstraction that lets us
    # coalesce many requests into a bulk operation. This code is
    # fine for a dozen requests, but not for a million.
    def map(self, dr: DataRecord):
        textcontent = self.get_page_text(dr.url)
        dr2 = DataRecord(self.outputSchema, parent_uuid=dr._uuid)
        dr2.url = dr.url

        html = textcontent
        tokens = html.split()[:5000]
        dr2.html = " ".join(tokens)

        strippedHtml = self.html_to_text_with_links(textcontent)
        tokens = strippedHtml.split()[:5000]
        dr2.text = " ".join(tokens)

        # get current timestamp, in nice ISO format
        dr2.timestamp = datetime.datetime.now().isoformat()
        return dr2
    
class DownloadBinaryFunction(UserFunction):
    """DownloadBinaryFunction downloads binary content from a URL."""
    def __init__(self):
        super().__init__("localfunction:downloader", URL, Download)

    # Someday we should introduce an abstraction that lets us
    # coalesce many requests into a bulk operation. This code is
    # fine for a dozen requests, but not for a million.
    def map(self, dr: DataRecord):
        content = requests.get(dr.url).content
        dr2 = DataRecord(pz.File, parent_uuid=dr._uuid)
        dr2.url = dr.url
        dr2.content = content
        dr2.timestamp = datetime.datetime.now().isoformat()
        return dr2
