from palimpzest.datasources import FileSource
from palimpzest.corelib import Schema
from palimpzest.datamanager import DataDirectory
import logging




class ValidationData:
    """
    Base class for reading validation dataset

    """

    # TODO: currently only support reading validation dataset from xlsx files.
    def __init__(
        self,
        input_file_id: str = None,
        output_file_id: str = None,
        verbose: bool = False,
    ):
        
        # Make sure reading data from input is deterministic 
        # TODO(chjun): currently we only support File in and File out. 
        # e.g. If we have three input files, we should have 3 output files to indicate the output for each file.
        # Currently we don't consider file1 JOIN file2 -> file3. case.
        self.input = DataDirectory().getRegisteredDataset(input_file_id)
        self.output = DataDirectory().getRegisteredDataset(output_file_id)

    def get_input(self):
        return self.input
    
    def get_output(self):
        return self.output

