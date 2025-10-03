"""This file contains utility format strings which are templated into many of our prompts."""

### FORMATTING INSTRUCTIONS ###
ONE_TO_ONE_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON dictionary. The dictionary should only have the specified output fields."
ONE_TO_MANY_OUTPUT_FORMAT_INSTRUCTION = "Remember, your answer must be a valid JSON list of dictionaries. The list may contain one or more dictionaries, and each dictionary should only have the specified output fields."

### USER-PROVIDED DESCRIPTION FOR MAPS / FILTERS / JOINS ###
DESC_SECTION = """
The user has additionally provided you with this description of the task you need to perform:
{desc}
"""

### JOB INSTRUCTIONS ###
AGG_JOB_INSTRUCTION = """analyze input {modalities} in order to perform an aggregation and generate a JSON object"""
MAP_JOB_INSTRUCTION = """analyze input {modalities} in order to produce a JSON object"""
FILTER_JOB_INSTRUCTION = """analyze input {modalities} in order to answer a TRUE / FALSE question"""
JOIN_JOB_INSTRUCTION = """analyze input {modalities} in order to determine whether two data records satisfy a join condition"""
PROPOSER_JOB_INSTRUCTION = """analyze input {modalities} in order to produce an answer to a question"""

### AGG / FILTER / JOIN CONDITIONS ###
EXAMPLE_AGG_INSTRUCTION = "Count the distinct number of scientists in the input."
EXAMPLE_FILTER_CONDITION = "The subject of the input is a foundational computer scientist."
EXAMPLE_JOIN_CONDITION = "The two inputs are scientists in the same academic field."

### EXAMPLE INPUT FIELDS ###
TEXT_EXAMPLE_INPUT_FIELDS = """
- text: a text passage describing a scientist
- birthday: the scientist's birthday
"""
IMAGE_EXAMPLE_INPUT_FIELDS = """
- image: an image of the scientist
- photographer: the photographer of the image
"""
AUDIO_EXAMPLE_INPUT_FIELDS = """
- recording: an audio recording of a newscast about the scientist's contributions to their field
- speaker: the speaker in the recording
"""
RIGHT_TEXT_EXAMPLE_INPUT_FIELDS = """
- contents: the contents of a text file
"""
RIGHT_IMAGE_EXAMPLE_INPUT_FIELDS = """
- headshot: a headshot of a famous scientist
"""
RIGHT_AUDIO_EXAMPLE_INPUT_FIELDS = """
- podcast: an audio recording of a podcast about historic scientists
"""

### EXAMPLE OUTPUT FIELDS ###
TEXT_EXAMPLE_OUTPUT_FIELDS = """- name: the name of the scientist
- birth_year: the year the scientist was born"""
IMAGE_EXAMPLE_OUTPUT_FIELDS = """- is_bald: true if the scientist is bald and false otherwise"""
AUDIO_EXAMPLE_OUTPUT_FIELDS = """- birthplace: the city where the scientist was born"""
AGG_EXAMPLE_OUTPUT_FIELDS = """- num_distinct_scientists: the number of distinct scientists mentioned in the input"""

### EXAMPLE CONTEXTS ###
TEXT_EXAMPLE_CONTEXT = """
  "text": "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.",
  "birthday": "December 10, 1815"
"""
IMAGE_EXAMPLE_CONTEXT = """
  "image": <bytes>,
  "photographer": "CameraEnthusiast1"
"""
AUDIO_EXAMPLE_CONTEXT = """
  "recording": <bytes>,
  "speaker": "Walter Cronkite"
"""
RIGHT_TEXT_EXAMPLE_CONTEXT = """
  "content": "Alan Turing was a pioneering computer scientist and mathematician. He is widely considered to be the father of theoretical computer science and artificial intelligence."
"""
RIGHT_IMAGE_EXAMPLE_CONTEXT = """
  "headshot": <bytes>
"""
RIGHT_AUDIO_EXAMPLE_CONTEXT = """
  "podcast": <bytes>
"""
SECOND_TEXT_EXAMPLE_CONTEXT = """
  "text": "Alan Turing was a pioneering computer scientist and mathematician. He is widely considered to be the father of theoretical computer science and artificial intelligence.",
  "birthday": "June 23, 1912"
"""
SECOND_IMAGE_EXAMPLE_CONTEXT = """
  "image": <bytes>,
  "photographer": "PhotoPro42"
"""
SECOND_AUDIO_EXAMPLE_CONTEXT = """
  "recording": <bytes>,
  "speaker": "Barbara Walters"
"""
THIRD_TEXT_EXAMPLE_CONTEXT = """
  "text": "Ada Lovelace is a historically significant computer scientist.",
  "birthday": "December 10, 1815"
"""
THIRD_IMAGE_EXAMPLE_CONTEXT = """
  "image": <bytes>,
  "photographer": "PicturePerfect"
"""
THIRD_AUDIO_EXAMPLE_CONTEXT = """
  "recording": <bytes>,
  "speaker": "Anderson Cooper"
"""

### DISCLAIMERS ###
IMAGE_DISCLAIMER = """
\n<image content provided here; assume in this example the image shows Ada Lovelace wearing a hat on top of her hair>
"""
AUDIO_DISCLAIMER = """
\n<audio content provided here; assume in this example the recording is about Ada Lovelace's upbringing in London>
"""
RIGHT_IMAGE_DISCLAIMER = """
\n<image content provided here; assume in this example the image shows Alan Turing working at his desk>
"""
RIGHT_AUDIO_DISCLAIMER = """
\n<audio content provided here; assume in this example the podcast is discussing Alan Turing's work on the Enigma code>
"""
AGG_IMAGE_DISCLAIMER = """
\n<image content provided here; assume in this example the first image shows Ada Lovelace, the second image shows Alan Turing, and the third image shows Ada Lovelace again>
"""
AGG_AUDIO_DISCLAIMER = """
\n<audio content provided here; assume in this example the first recording is about Ada Lovelace, the second recording is about Alan Turing, and the third recording is about Ada Lovelace again>
"""

### EXAMPLE REASONINGS ###
TEXT_EXAMPLE_REASONING = """The text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815."""
IMAGE_EXAMPLE_REASONING = """The image shows hair on top of the scientist's head, so the is_bald field should be false."""
AUDIO_EXAMPLE_REASONING = """The newscast recording discusses Ada Lovelace's upbringing in London, so the birthplace field should be "London"."""
AGG_EXAMPLE_REASONING = """The input contains two distinct scientists: "Augusta Ada King" and "Alan Turing". Although "Ada Lovelace" is mentioned twice, she should only be counted once. Therefore, the number of distinct scientists mentioned in the input is 2."""
FILTER_EXAMPLE_REASONING = """Ada Lovelace is a foundational computer scientist, therefore the answer is TRUE."""
JOIN_EXAMPLE_REASONING = """The subject of the left record is Ada Lovelace and the subject of the right record is Alan Turing. Since both inputs are about computer scientists, they satisfy the join condition. Therefore, the answer is TRUE."""

### EXAMPLE ANSWERS ###
AGG_EXAMPLE_ANSWER = """
  "num_distinct_scientists": 2
"""
TEXT_EXAMPLE_ANSWER = """
  "name": "Augusta Ada King",
  "birth_year": 1815
"""
IMAGE_EXAMPLE_ANSWER = """
  "is_bald": false,
"""
AUDIO_EXAMPLE_ANSWER = """
  "birthplace": "London",
"""
TEXT_SENTENCE_EXAMPLE_ANSWER = """the text passage mentions the scientist's name as "Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace" and the scientist's birthday as "December 10, 1815". Therefore, the name of the scientist is "Augusta Ada King" and the birth year is 1815."""
IMAGE_SENTENCE_EXAMPLE_ANSWER = """The image shows hair on top of the woman's head, so the is_bald field should be false."""
AUDIO_SENTENCE_EXAMPLE_ANSWER = """The newscast recording discusses Ada Lovelace's upbringing in London, so her birthplace is "London"."""
