from palimpzest.elements import DataRecord, UserFunction
from palimpzest.datamanager import DataDirectory
from palimpzest.corelib import URL, WebPage, Download

import requests
import datetime

#
# REMIND: I don't love the syntax of UDFs right now. 
# It's too verbose and clumsy.
#
class DownloadHTMLFunction(UserFunction):
    """DownloadHTMLFunction downloads the HTML content of a web page."""
    def __init__(self):
        super().__init__("localfunction:webgetter", URL, WebPage)

    # Someday we should introduce an abstraction that lets us
    # coalesce many requests into a bulk operation. This code is
    # fine for a dozen requests, but not for a million.
    def map(self, dr: DataRecord):
        textcontent = requests.get(dr.url).text
        dr2 = DataRecord(self.outputSchema)
        dr2.url = dr.url
        dr2.html = textcontent
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
        dr2 = DataRecord(self.outputSchema)
        dr2.url = dr.url
        dr2.content = content
        dr2.timestamp = datetime.datetime.now().isoformat()
        return dr2

DataDirectory().registerUserFunction(DownloadHTMLFunction())
DataDirectory().registerUserFunction(DownloadBinaryFunction())

