from palimpzest.datasources import FileSource
from palimpzest.corelib import Schema
import logging




class ValidationData:
    """
    Base class for reading validation dataset

    """

    # TODO: currently only support reading validation dataset from xlsx files.
    def __init__(
        self,
        input_file: str = None,
        input_schema: Schema = None,
        output_file: str = None,
        output_schema: str = None,
        verbose: bool = False,
    ):
        if input_file =="":
            logging.CRITICAL("!!We'll skip validation when validation input is empty.")

        # Make sure reading data from input is deterministic 
        # TODO(chjun): currently we only support File in and File out. 
        # e.g. If we have three input files, we should have 3 output files to indicate the output for each file.
        # Currently we don't consider file1 JOIN file2 -> file3. case.
        # TODO(chjun): I don't think users need to provide both path and dataset_id for the FileSource at the same time, 
        #              so I just hack a path here.
        input_file_path = "testdata/biofabric-tiny-quality-estimation/dou_mmc1_input.xlsx"
        output_file_path = "testdata/biofabric-tiny-quality-estimation/dou_mmc1_output.xlsx"
        self.input = FileSource(path=input_file_path, dataset_id=input_file)

        self.output = FileSource(path=output_file_path, dataset_id=output_file)

    def get_input(self):
        return self.input
    
    def get_output(self):
        return self.output

