"""
This module is responsible for building schemas dynamically, taking an input file and generating a schema for it.
The class offers a general class method, from_file, that takes a file and generates a schema for it.
This method is a simple wrapper for different methods, e.g., from_csv, from_yml, etc.

"""

import json
import os

import pandas as pd
import yaml
from pyld import jsonld

import palimpzest as pz
from palimpzest.corelib.fields import Field
from palimpzest.corelib.schemas import Schema


class SchemaBuilder:

    @classmethod
    def from_file(cls,
        schema_file: str,
        schema_name: str = "",
        schema_description: str = "",
        include_attributes: list = None,
        schema_type: Schema = None,
        ):
        """
        Inputs:
            schema_file: str - the path to the file
            description (optional): str - the description of the schema
            name (optional): str - the name of the schema
            target_attributes (optional): list - a list of attribute names to include in the schema. If None, all attributes are included.
            schema_type (optional): Schema - the parent type of the schema to generate, e.g. ScientificPapers have a schema_type of PDFFile. If None, a generic Schema type is used.
        Outputs:
            A class object - the dynamically generated class
        """

        # Get the file extension
        filename = os.path.basename(schema_file)
        basename, file_extension = os.path.splitext(filename)
      
        if file_extension == ".csv":
            schema_data = cls.from_csv(schema_file)
        elif file_extension == ".json":
            schema_data = cls.from_json(schema_file)
        elif file_extension == ".jsonld":
            schema_data = cls.from_jsonld(schema_file)
        elif file_extension == ".yml":
            schema_data = cls.from_yml(schema_file)
        else:
            raise ValueError("Unsupported file format!")

        # If additional metadata is not provided, read it from the file. 
        # If not available, generate it from the filename
        if not schema_name:
            if schema_data['name']:
                schema_name = schema_data['name']
            else:
                schema_name = "".join([word.capitalize() for word in basename.split("_")])

        if not schema_description:
            if schema_data['description']:
                schema_description = schema_data['description']
            else:
                schema_description = f"A schema generated from the {file_extension} file {basename}."

        if include_attributes is None:
           include_attributes = []

        if schema_type is None:
            if schema_data.get('type', None):
                # Find if the schema type is a valid class in pz
                parsed_type = getattr(pz, schema_data['type'], Schema)
                schema_type = parsed_type if issubclass(parsed_type, Schema) else Schema
            else:
                schema_type = Schema
           
        # Generate the schema class dynamically
        attributes = {"__doc__": schema_description}
        for field in schema_data['fields']:
            if len(include_attributes) and field['name'] not in include_attributes:
                continue
            name = field['name']
            description = field.get('description', '')
            required = field.get('required', False)
            field_type = field.get('type', 'Field')
            field_type = getattr(pz, field_type, Field)
            if not issubclass(field_type, Field):
                field_type = Field
                  
            attributes[name] = field_type(desc=description, required=required)

        # Create the class dynamically
        return type(schema_name, (schema_type,), attributes)

    @classmethod
    def from_csv(
        cls,
        schema_file: str,
    ) -> dict:
        """
        The attributes are extracted from the column names of the CSV file.
        If columns contain null values, they are marked as optional.
        TODO: Find a way to infer the description of the field.
        """
        
        # Use pandas to read the CSV file
        df = pd.read_csv(schema_file)
        columns = df.columns.tolist()

        # Generate the schema class dynamically
        fields = []
        for col in columns:
            required = not df[col].isnull().values.any()

            field_type = df[col].dtype
            if field_type == float or field_type == int:  # noqa
                field_type = "NumericField"
            else:
                field_type = "Field"

            fields.append({"name":col,
                           "description":"",
                           "type":field_type,
                           "required":required})
        
        return {
            "name": '',
            "description": '',
            "fields": fields,
        }

    @classmethod
    def from_jsonld(
        cls,
        schema_file: str,
    ) -> dict:
        """JSON-LD schema builder.
        The attributes are extracted from the JSON-LD objects of type 'rdfs:Class'.
        If they contain a 'comment' field, this is used to populate a schema descripton.
        If they contain a 'rangeIncludes' field, this is used within the description to
        signal the list of valid values.
       """

        # Load the schema from the JSONLD file
        with open(schema_file) as file:
            jsonld_data = json.load(file)
        context = jsonld_data.get("@context")
        compacted_data = jsonld.compact(jsonld_data, context)
        compacted_graph = compacted_data["@graph"]

        fields = []

        for node in compacted_graph:
            if node.get("@type") != "rdfs:Class":
                continue
            name = node.get("rdfs:label")

            values = []
            if "schema:rangeIncludes" in node:
                values = [val["@id"].split(":")[-1] for val in node["schema:rangeIncludes"]]
            
            description = node.get("rdfs:comment", "")
            if values:
                description += " The only valid values are: " + ", ".join(values)
            fields.append({
                "name": name,
                "description": description, 
                "values": values,
                "required": True})

        return {
            "name": '',
            "description": '',
            "fields": fields,
        }
    
    @classmethod
    def from_json(
            cls,
            schema_file: str,
            schema_name: str = "",
            schema_description: str = "",
            include_attributes: list = None,
            schema_type: Schema = None,
    )-> dict:
        """
        The attributes are extracted from the JSON objects.
        The format of the json has to be in the form:
        {
            "attribute1": {
                "description": "description",
                "required": True
            },
            ...
        }
        """

        # Load the schema from the JSON file
        with open(schema_file) as file:
            schema_data = json.load(file)

        return schema_data
    
    @classmethod
    def from_yml(
            cls,
            schema_file: str,
    )-> dict:
        """
        The attributes are extracted from the YAML file.
        The schema name and description are extracted from the YAML file if it contains them, overwriting the input parameters.

        The format of the yaml has to be in the form:
        schema:
          name:
          description:
          fields:
            - name: attribute_name
              description: description
              required: True
        ...
        """

        # Load the schema from the YAML file
        with open(schema_file) as file:
            schema_data = yaml.safe_load(file)

        schema_data = schema_data["schema"]
        return {
            "name": schema_data.get("name", ""),
            "description": schema_data.get("description", ""),
            "fields": schema_data.get("fields", []),
            "type": schema_data.get("type", "")
        }