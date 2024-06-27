from palimpzest.datasources import FileSource, RecordsFromXLSFile
from palimpzest.corelib import Schema, XLSFile,Table
from palimpzest.sets import Set, Dataset
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
import logging




class ValidationData:
    """
    Base class for reading validation dataset

    """

    # TODO: currently only support reading validation dataset from xlsx files.
    def __init__(
        self,
        input_file: str = None, # should be able to read image/file/xlsx
        input_schema: Schema = None,
        output_file: str = None, # output should be in DataRecord.
        output_schema: str = None,
        verbose: bool = False,
    ):
        if input_file =="":
            logging.CRITICAL("!!We'll skip validation when validation input is empty.")

        # TODO(chjun): ability to read file to Dataset with schema directly. Deterministic actions.
        #   Input needs to be dataset so that it can work with the system well. The system only accept Dataset as input for the computation.
        #   Output is list of DataRecord as this is what the system outputs.
        # Is it still true in the future?
        # import os.path
        # if not os.path.isfile(input_file):
        #     raise Exception("Cannot find input file ", input_file)
        # if not os.path.isfile(output_file):
        #     raise Exception("Cannot find output file ", output_file)
        
        # Make sure reading data from input is deterministic 
        # TODO(chjun): currently we only support File in and File out. 
        # e.g. If we have three input files, we should have 3 output files to indicate the output for each file.
        # Currently we dobn't consider file1 JOIN file2 -> file3. case.
        input_file_path = "/Users/chjun/Documents/GitHub/code/palimpzest/testdata/biofabric-tiny-quality-estimation/dou_mmc1_input.xlsx"
        output_file_path = "/Users/chjun/Documents/GitHub/code/palimpzest/testdata/biofabric-tiny-quality-estimation/dou_mmc1_output.xlsx"
        input_source = FileSource(path=input_file_path, dataset_id=input_file)
        self.input = input_source

        output_source = FileSource(path=output_file_path, dataset_id=output_file)
        self.output = output_source

    def get_input(self):
        return self.input
    
    def get_output(self):
        return self.output

