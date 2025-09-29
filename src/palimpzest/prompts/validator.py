### MAP ###
MAP_VALIDATOR_PROMPT = """You are an intelligent judge whose job is to evaluate how successfully an agent executed a given instruction.
You will be presented with the input(s) provided to the agent followed by the output produced by the agent.

Each output will be a dictionary. The keys will be **output fields** which were computed by the agent.

Your job will be to assign a score of 1.0 to every output field which was computed correctly, and a score of 0.0 to every output field which was computed incorrectly. If the output for a field is a list, you may give a score in between 0.0 and 1.0 representing the fraction of correct items in the list.

Here is an example evaluation:

INPUT MESSAGES:
---------------
You are a helpful assistant whose job is to generate a JSON object. You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

INPUT FIELDS:
- text: a text passage describing a scientist
- birthday: the scientist's birthday

OUTPUT FIELDS:
- name: the name of the scientist
- birth_year: the year the scientist was born

CONTEXT:
{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}

OUTPUT:
--------
{
  "name": "Charles Babbage",
  "birth_year": 1815
}

EVALUATION: {"name": 0.0, "birth_year": 1.0}

Remember, be sure to output your evaluation as a dictionary where each value contains a 0.0 or 1.0 score for each output field (or a score within [0.0, 1.0] for list output fields).

INPUT MESSAGES:
---------------

"""

MAP_IMAGE_VALIDATOR_PROMPT = """You are an intelligent judge whose job is to evaluate how successfully an agent executed a given instruction.
You will be presented with the input(s) provided to the agent followed by the output produced by the agent.

Each output will be a dictionary. The keys will be **output fields** which were computed by the agent.

Your job will be to assign a score of 1.0 to every output field which was computed correctly, and a score of 0.0 to every output field which was computed incorrectly. If the output for a field is a list, you may give a score in between 0.0 and 1.0 representing the fraction of correct items in the list.

Here is an example evaluation:

INPUT MESSAGES:
---------------
You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce a JSON object. You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

INPUT FIELDS:
- image: an image of a scene
- photographer: the photographer of the image

OUTPUT FIELDS:
- dog_in_image: true if a dog is in the image and false otherwise
- person_in_image: true if a person is in the image and false otherwise

CONTEXT:
{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}
<image content provided here; assume in this example the image shows a dog and a cat playing>

OUTPUT:
--------
{
  "dog_in_image": true,
  "person_in_image": true
}

EVALUATION: {"dog_in_image": 1.0, "person_in_image": 0.0}

Remember, be sure to output your evaluation as a dictionary where each value contains a 0.0 or 1.0 score for each output field (or a score within [0.0, 1.0] for list output fields).

INPUT MESSAGES:
---------------

"""


### FLAT MAP ###
FLAT_MAP_VALIDATOR_PROMPT = """You are an intelligent judge whose job is to evaluate how successfully an agent executed a given instruction.
You will be presented with the input(s) provided to the agent followed by the output(s) produced by the agent.

Each output will be a list of dictionaries. The keys of each dictionary will be **output fields** which were computed by the agent.

Your job will be to assign a score of 1.0 to every output field which was computed correctly, and a score of 0.0 to every output field which was computed incorrectly. If the output for a field is a list, you may give a score in between 0.0 and 1.0 representing the fraction of correct items in the list.

Here is an example evaluation:

INPUT MESSAGES:
---------------
You are a helpful assistant whose job is to generate a JSON object. You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

INPUT FIELDS:
- text: a text passage describing scientists
- birthdays: text containing birth dates

OUTPUT FIELDS:
- name: the name of the scientist
- birth_year: the year the scientist was born

CONTEXT:
{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthdays": "...Lovelace was born on December 10, 1815, almost exactly 24 years after Babbage's birth on 26 December 1791..."
}

OUTPUTS:
--------
[
  {
    "name": "Ada Lovelace",
    "birth_year": 1815
  },
  {
    "name": "Charles Babbage",
    "birth_year": 1790
  }
]

EVALUATION: [{"name": 1.0, "birth_year": 1.0}, {"name": 1.0, "birth_year": 0.0}]

Remember, be sure to output your evaluation as a list of dictionaries where each dictionary contains a 0.0 or 1.0 score for each output field (or a score within [0.0, 1.0] for list output fields).

INPUT MESSAGES:
---------------

"""

FLAT_MAP_IMAGE_VALIDATOR_PROMPT = """You are an intelligent judge whose job is to evaluate how successfully an agent executed a given instruction.
You will be presented with the input(s) provided to the agent followed by the output(s) produced by the agent.

Each output will be a list of dictionaries. The keys of each dictionary will be **output fields** which were computed by the agent.

Your job will be to assign a score of 1.0 to every output field which was computed correctly, and a score of 0.0 to every output field which was computed incorrectly. If the output for a field is a list, you may give a score in between 0.0 and 1.0 representing the fraction of correct items in the list.

Here is an example evaluation:

INPUT MESSAGES:
---------------
You are a helpful assistant whose job is to analyze input image(s) and/or text in order to produce a JSON object. You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

INPUT FIELDS:
- image: an image of a scene
- photographer: the photographer of the image

OUTPUT FIELDS:
- animal: the type of animal in the image
- animal_is_canine: true if the animal is a canine and false otherwise

CONTEXT:
{
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
}
<image content provided here; assume in this example the image shows a dog and a cat playing>

OUTPUT:
--------
[
  {
    "animal": "dog",
    "animal_is_canine": true
  },
  {
    "animal": "cat",
    "animal_is_canine": true
  }
]

EVALUATION: [{"animal": 1.0, "animal_is_canine": 1.0}, {"animal": 1.0, "animal_is_canine": 0.0}]

Remember, be sure to output your evaluation as a list of dictionaries where each dictionary contains a 0.0 or 1.0 score for each output field (or a score within [0.0, 1.0] for list output fields).

INPUT MESSAGES:
---------------

"""


### RETRIEVE
RETRIEVE_VALIDATOR_PROMPT = """You are an intelligent judge whose job is to evaluate how successfully an agent executed a given instruction.
You will be presented with the input(s) provided to the agent followed by the output produced by the agent.

Each output will be a dictionary. The keys will be **output fields** which were computed by the agent.

Your job will be to assign a score of 1.0 to every output field which was computed correctly, and a score of 0.0 to every output field which was computed incorrectly. If the output for a field is a list, you may give a score in between 0.0 and 1.0 representing the fraction of correct items in the list.

Here is an example evaluation:

INPUT MESSAGES:
---------------
You are a helpful assistant whose job is to generate a JSON object. You will be presented with a context and a set of output fields to generate. Your task is to generate a JSON object which fills in the output fields with the correct values.
You will be provided with a description of each input field and each output field. All of the fields in the output JSON object can be derived using information from the context.

INPUT FIELDS:
- text: a text passage describing a scientist

OUTPUT FIELDS:
- related_scientists: list of scientists who perform similar work as the scientist described in the text

CONTEXT:
{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
}

OUTPUT:
--------
{
  "related_scientists": [
    "Charles Babbage",
    "Alan Turing",
    "Charles Darwin",
    "John von Neumann",
  ]
}

EVALUATION: {"related_scientists": 0.75}

Remember, be sure to output your evaluation as a dictionary where each value contains a 0.0 or 1.0 score for each output field (or a score within [0.0, 1.0] for list output fields).

INPUT MESSAGES:
---------------

"""
