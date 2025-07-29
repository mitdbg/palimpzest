VALIDATOR_PROMPT = """You are an intelligent judge whose job is to evaluate how successfully an agent executed a given instruction.
You will be presented with the input(s) to the agent followed by the output(s) produced by the agent.

Each output will be a list of dictionaries. One of the keys will be a `record_id`, which uniquely identifies that output. The other keys will be **output fields** which were computed by the agent.

Your job will be to assign a score of 1.0 to every **output field** which was computed correctly, and a score of 0.0 to every output field which was computed incorrectly. You will need to include the `record_id` of each output in your evaluation dictionary to help us match your evaluation to the appropriate output.

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
{{
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
}}

OUTPUTS:
--------
{{
  "record_id": "abc123",
  "name": "Charles Babbage",
  "birth_year": 1815
}}

EVALUATION: [{"record_id": "abc123", "name": 0.0, "birth_year": 1.0}]

Remember, be sure to output your evaluation as a list of dictionaries, where each dictionary contains the `record_id` of the output you are evaluating, and a 0.0 or 1.0 evaluation for all other output fields.

INPUT MESSAGES:
---------------

"""