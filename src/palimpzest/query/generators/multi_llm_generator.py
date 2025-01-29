from __future__ import annotations
import json
import os
from dotenv import load_dotenv
import base64
import re
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from string import Formatter
from typing import Any, Generic, Tuple, TypeVar

from colorama import Fore, Style
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
from together.types.chat_completions import ChatCompletionResponse

import palimpzest.prompts as prompts
from palimpzest.constants import (
    MODEL_CARDS,
    TOKENS_PER_CHARACTER,
    # RETRY_MAX_ATTEMPTS,
    # RETRY_MAX_SECS,
    # RETRY_MULTIPLIER,
    Cardinality,
    Model,
    PromptStrategy,
)
from palimpzest.core.data.dataclasses import GenerationStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BytesField, ImageBase64Field, ImageFilepathField, ImageURLField, ListField
from palimpzest.utils.generation_helpers import get_json_from_answer
from palimpzest.utils.sandbox import API

# Load environment variables from .env
load_dotenv()

# Ensure OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Initialize OpenAI client
openai = OpenAI(api_key=api_key)


def query_gpt4(prompt, model="gpt-4", temperature=0.7, max_tokens=500) -> ChatCompletion:
    """
    Query GPT-4 with a given prompt.
    """
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response


def generate_critique(user_prompt, original_output, critique_prompt_template) -> str:
    """
    Generate a critique of the original JSON output.
    """
    critique_prompt = critique_prompt_template.format(
        user_prompt=user_prompt, original_output=original_output
    )
    return query_gpt4(critique_prompt).choices[0].message.content.strip()


def refine_output(user_prompt, original_output, critique_output, refinement_prompt_template) -> ChatCompletion:
    """
    Refine the JSON output based on the critique.
    """
    refinement_prompt = refinement_prompt_template.format(
        user_prompt=user_prompt,
        original_output=original_output,
        critique_output=critique_output,
    )
    return query_gpt4(refinement_prompt)


def validate_json(json_output, output_fields):
    """
    Validate the generated JSON to ensure it meets the specified output fields.
    """
    try:
        parsed_json = json.loads(json_output)
        missing_fields = [field for field in output_fields if field not in parsed_json]
        if missing_fields:
            return False, f"Missing fields: {', '.join(missing_fields)}"
        return True, "JSON is valid."
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"

def verify(chat_payload, completion, prompt_strategy: PromptStrategy) -> ChatCompletion:
    """
    Runs the Multi-LLM Testing Framework
    """
    user_prompt = chat_payload['messages']
    original_output = completion.choices[0].message.content.strip()
    promptStrategyMap = {
        PromptStrategy.COT_QA: (prompts.COT_QA_BASE_SYSTEM_PROMPT_CRITIQUE, prompts.COT_QA_BASE_SYSTEM_PROMPT_REFINEMENT),
        PromptStrategy.COT_BOOL: (prompts.COT_BOOL_SYSTEM_PROMPT_CRITIQUE, prompts.COT_BOOL_SYSTEM_PROMPT_REFINEMENT),
        PromptStrategy.COT_BOOL_IMAGE: (prompts.COT_BOOL_IMAGE_SYSTEM_PROMPT_CRITIQUE, prompts.COT_BOOL_IMAGE_SYSTEM_PROMPT_REFINEMENT),
        PromptStrategy.COT_MOA_PROPOSER: (prompts.COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT_CRITIQUE, prompts.COT_MOA_PROPOSER_BASE_SYSTEM_PROMPT_REFINEMENT),
        PromptStrategy.COT_MOA_AGG: (prompts.COT_MOA_AGG_BASE_SYSTEM_PROMPT_CRITIQUE, prompts.COT_MOA_AGG_BASE_SYSTEM_PROMPT_REFINEMENT),
        PromptStrategy.COT_QA_IMAGE: (prompts.COT_QA_IMAGE_BASE_SYSTEM_PROMPT_CRITIQUE, prompts.COT_QA_IMAGE_BASE_SYSTEM_PROMPT_REFINEMENT),
    }

    if prompt_strategy not in promptStrategyMap:
        return completion
    
    critique_prompt_template, refinement_prompt_template = promptStrategyMap[prompt_strategy]

    critique_output = generate_critique(user_prompt, original_output, critique_prompt_template)
    print(f"Critique Output:\n{critique_output}\n")

    refined_output = refine_output(user_prompt, original_output, critique_output, refinement_prompt_template)
    print(f"Refined Output:\n{refined_output.choices[0].message.content.strip()}\n")

    return refined_output

# Example framework workflow
if __name__ == "__main__":
    # Initial user prompt
    user_prompt = """
    You are a helpful assistant whose job is to generate a JSON object.
    You will be presented with a context and a set of output fields to generate. Your task is to 
    generate a JSON object which fills in the output fields with the correct values.
    You will be provided with a description of each input field and each output field. All of the fields
    in the output JSON object can be derived using information from the context.

    Remember, your answer must be a valid JSON dictionary. The dictionary should only have the specified
    output fields. Finish your response with a newline character followed by ---
    ---

    CONTEXT:
    {'filename': 'zipper-a-espeed-28.txt', 'contents': 'Message-ID: 
    <19460654.1075851675082.JavaMail.evans@thyme>\nDate: Wed, 29 Nov 2000 11:34:00 -0800 (PST)\nFrom: 
    travis.mccullough@enron.com\nTo: andy.zipper@enron.com\nSubject: Redraft of the Exclusivity 
    Agreement\nMime-Version: 1.0\nContent-Type: text/plain; charset=us-ascii\nContent-Transfer-Encoding:
    7bit\nX-From: Travis McCullough\nX-To: Andy Zipper\nX-cc: \nX-bcc: \nX-Folder: 
    \\Andrew_Zipper_Nov2001\\Notes Folders\\Espeed\nX-Origin: ZIPPER-A\nX-FileName: azipper.nsf\n\nHeard
    the Skilling meeting was postponed.  Here\'s the redraft of the \nExclusivity; I\'m sending a draft 
    of the consent to Raptor transactions down \nshortly.\n\nTravis McCullough\nEnron North America 
    Corp.\n1400 Smith Street EB 3817\nHouston Texas 77002\nPhone:  (713) 853-1575\nFax: (713) 646-3490  
    \n----- Forwarded by Travis McCullough/HOU/ECT on 11/29/2000 07:34 PM -----\n\n\t"Stockbridge, 
    Edward T" <tstockbridge@velaw.com>\n\t11/29/2000 03:56 PM\n\t\t \n\t\t To: "McCullough, Travis 
    (Enron)" <travis.mccullough@enron.com>\n\t\t cc: "Collins, Christopher" <chriscollins@velaw.com>, 
    "Wills, Anthony" \n<awills@velaw.com>\n\t\t Subject: Redraft of the Exclusivity 
    Agreement\n\n\n\n\nTravis:\n\n Please find attached a clean and redline version of the 
    revised\nExclusivity Agreement, with the changes that we discussed.  The redline\nshows changes from
    the draft submitted to us by eSpeed.\n\n Per our earlier conversation, you said that Andy was still 
    thinking\nabout one open point, namely: Whether the definition of the Enron Platform\nshould be 
    modified to anticipate some changes that may occur.  Specifically\nI suggested that the requirement 
    that the Enron Platform be "operated" by an\nEnron Party be deleted, and leaving only the limitation
    of ownership.\n\n I have given Anthony comments on the series designation.  He will be\nmaking other
    changes to the designation and will forward it to you directly.\n\n Please let me know if you have 
    any questions or comments, or if I\ncan be of any further assistance.\n\n Best regards,\n\n Ted\n   
    ++++++CONFIDENTIALITY NOTICE+++++\nThe information in this email may be confidential and/or 
    privileged.  This\nemail is intended to be reviewed by only the individual or organization\nnamed 
    above.  If you are not the intended recipient or an authorized\nrepresentative of the intended 
    recipient, you are hereby notified that any\nreview, dissemination or copying of this email and its 
    attachments, if any,\nor the information contained herein is prohibited.  If you have received\nthis
    email in error, please immediately notify the sender by return email\nand delete this email from 
    your system.  Thank You\n <<Nov 29 Exclusivity Agreement [by VE].DOC>>   <<Redline.rtf>>\n\n - Nov 
    29 Exclusivity Agreement [by VE].DOC\n - Redline.rtf'}

    INPUT FIELDS:
    - contents: The contents of the file
    - filename: The UNIX-style name of the file

    OUTPUT FIELDS:
    - sender: The email address of the sender
    - subject: The subject of the email
    """

    # Critique prompt template
    critique_prompt_template = """
    The following prompt was used to generate a JSON object:
    {user_prompt}

    Here is the JSON output generated by the model:
    {original_output}

    Critique the JSON output. Are the values accurate based on the context? Is the JSON structure correct?
    Highlight any issues and suggest corrections.
    """

    # Refinement prompt template
    refinement_prompt_template = """
    The following prompt was used to generate a JSON object:
    {user_prompt}

    Here is the original JSON output:
    {original_output}

    Here is the critique of the output:
    {critique_output}

    Refine the original JSON output to address the critique. Ensure it is accurate and valid.
    """

    # Step 1: Generate initial JSON
    original_output = query_gpt4(user_prompt)
    print(f"Original Output:\n{original_output}\n")

    # Validate JSON
    #is_valid, validation_message = validate_json(original_output, ["sender", "subject"])
    #print(f"Validation: {validation_message}\n")
    #if not is_valid:
    #    raise ValueError("Generated JSON is invalid!")

    # Step 2: Generate critique
    critique_output = generate_critique(user_prompt, original_output, critique_prompt_template)
    print(f"Critique Output:\n{critique_output}\n")

    # Step 3: Refine the JSON
    refined_output = refine_output(user_prompt, original_output, critique_output, refinement_prompt_template)
    print(f"Refined Output:\n{refined_output}\n")

    # Final Validation
    ##is_valid, validation_message = validate_json(refined_output, ["sender", "subject"])
    #print(f"Final Validation: {validation_message}")
