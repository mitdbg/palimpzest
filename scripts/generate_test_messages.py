#!/usr/bin/env python3
"""
Script to generate test messages for each provider/modality combination.

This script creates messages using the Generator's prompt factory for each supported
provider/modality cross product and saves them to JSON files for use in provider testing.

The messages include a static context that is just long enough to trigger caching
(~1200 tokens, above the 1024 token threshold for OpenAI/Gemini).

Supported provider/modality combinations:
- Anthropic: text-only, image-only, text-image (no audio support)
- OpenAI: text-only, image-only, text-image
- OpenAI-Audio: audio-only, text-audio
- Gemini: all 7 modality combinations
- Vertex AI: all 7 modality combinations

Output files are saved to: tests/pytest/data/generator_messages/
Format: {modality}_{provider}.json (e.g., text-only_anthropic.json)
"""

import json
import os
import sys

from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.query.generators.generators import Generator

STATIC_CONTEXT = """
PACHYDERM SANCTUARY & RESEARCH CENTER: OPERATIONS MANUAL (v2025.1)

SECTION 1: INTRODUCTION AND MISSION
The Pachyderm Sanctuary & Research Center (PSRC) is dedicated to the preservation, study, and rehabilitation of elephant species.
All staff members, researchers, and volunteers must adhere to these protocols to ensure the safety of both the animals and human personnel.
Our mission combines advanced veterinary science with compassionate care to support endangered populations.

SECTION 2: SPECIES IDENTIFICATION PROTOCOLS
2.1 African Savanna Elephant (Loxodonta africana):
    - Characteristics: Larger ears (shaped like Africa), concave back, two fingers on trunk tip.
    - Identification: Staff must document tusk shape, ear notch patterns, and tail hair density.
2.2 African Forest Elephant (Loxodonta cyclotis):
    - Characteristics: Smaller stature, oval-shaped ears, straighter tusks pointing downward.
    - Identification: specialized genetic sampling required for distinct lineage verification.
2.3 Asian Elephant (Elephas maximus):
    - Characteristics: Smaller ears, convex or level back, one finger on trunk tip, twin domes on head.
    - Subspecies: Indian, Sri Lankan, Sumatran, and Borneo Pygmy.

SECTION 3: NUTRITIONAL REQUIREMENTS AND FEEDING
3.1 Daily Intake:
    - Adult males: 200-300 kg of forage per day.
    - Adult females: 150-200 kg of forage per day.
    - Water: 100-200 liters per day, depending on ambient temperature.
3.2 Approved Forage:
    - Grasses: Timothy, Bermuda, Napier.
    - Browse: Acacia branches, Bamboo, Ficus leaves.
    - Supplements: Mineral blocks, specialized pellets (Protocol 3.2.A).
3.3 Prohibited Items:
    - High sugar fruits (limited to training rewards).
    - Fermented grains.
    - Toxic plants: Oleander, Rhododendron, Lantana.

SECTION 4: MEDICAL AND HUSBANDRY PROCEDURES
4.1 Routine Examinations:
    - Foot care: Daily inspection of pads and nails. Weekly trimming required.
    - Skin care: Daily dust baths or mud wallows to prevent sunburn and insect bites.
    - Musth Management: Adult males in musth must be isolated in reinforced enclosures (Zone C).
4.2 Emergency Protocols:
    - Colic: Immediate veterinary notification. Monitor behavior for lethargy or rolling.
    - Trauma: Isolate injured animal. Prepare mobile X-ray unit.
    - Anesthesia: Only senior veterinarians may administer etorphine. Reversal agent (diprenorphine) must be on hand.

SECTION 5: SOCIAL STRUCTURE AND ENRICHMENT
5.1 Herd Dynamics:
    - Matriarchal groups must not be disrupted. New introductions require a 30-day fence-line integration period.
    - Bull groups (bachelor herds) require larger ranging areas (Zone B).
5.2 Cognitive Enrichment:
    - Puzzle feeders must be rotated daily.
    - Sensory enrichment (scents, sounds) applied twice weekly.
    - Social play: Monitoring required during pool sessions.

SECTION 6: RESEARCH DATA COLLECTION
6.1 Vocalization Analysis:
    - Infrasound (< 20 Hz) recording devices active 24/7 in Sector 4.
    - Trumpets, rumbles, and roars must be tagged with behavioral context (e.g., "greeting", "distress").
6.2 Biometrics:
    - Shoulder height recorded quarterly.
    - Body condition scoring (1-5 scale) performed monthly.
    - Tusk length and circumference measured annually.

SECTION 7: HUMAN-ANIMAL INTERACTION
7.1 Protected Contact (PC):
    - All interactions occur through a protective barrier. No free contact permitted.
    - Staff must maintain a 2-meter safety zone from the barrier when not actively training.
7.2 Training:
    - Positive reinforcement (operant conditioning) only.
    - Bullhooks (ankus) are strictly prohibited on sanctuary grounds.
    - Targets and whistles used for medical behaviors (trunk lift, ear presentation).

You are an AI Research Assistant for the PSRC. Your job is to analyze data inputs regarding elephant specimens and determine if they match specific biological criteria or sanctuary records.
Analyze the input and provide the requested identification details.
"""


# =============================================================================
# INPUT SCHEMAS
# =============================================================================
class TextInputSchema(BaseModel):
    """Schema for text-only input."""
    text: str = Field(description="Description of an animal")
    age: int = Field(description="The age of the animal in years")


class ImageInputSchema(BaseModel):
    """Schema for image-only input."""
    image_file: ImageFilepath = Field(description="File path to an image of an animal")
    height: float = Field(description="The estimated height of the animal in cm")


class AudioInputSchema(BaseModel):
    """Schema for audio-only input."""
    audio_file: AudioFilepath = Field(description="File path to an audio recording of an animal")
    year: float = Field(description="The year the recording was made")


# Union schemas for multi-modal inputs
TextImageInputSchema = union_schemas([TextInputSchema, ImageInputSchema])
TextAudioInputSchema = union_schemas([TextInputSchema, AudioInputSchema])
ImageAudioInputSchema = union_schemas([ImageInputSchema, AudioInputSchema])
TextImageAudioInputSchema = union_schemas([TextInputSchema, ImageInputSchema, AudioInputSchema])


class OutputSchema(BaseModel):
    """Output schema for animal identification."""
    animal: str = Field(description="The animal in the input")

MODALITY_CONFIGS = {
    "text-only": {
        "input_schema": TextInputSchema,
        "data_item": {
            "text": "An elephant is a large gray animal with a trunk and big ears. It makes a trumpeting sound.",
            "age": 15,
        },
    },
    "image-only": {
        "input_schema": ImageInputSchema,
        "data_item": {
            "image_file": "tests/pytest/data/elephant.png",
            "height": 304.5,
        },
    },
    "audio-only": {
        "input_schema": AudioInputSchema,
        "data_item": {
            "audio_file": "tests/pytest/data/elephant.wav",
            "year": 2020,
        },
    },
    "text-image": {
        "input_schema": TextImageInputSchema,
        "data_item": {
            "text": "An elephant is a large gray animal with a trunk and big ears. It makes a trumpeting sound.",
            "age": 15,
            "image_file": "tests/pytest/data/elephant.png",
            "height": 304.5,
        },
    },
    "text-audio": {
        "input_schema": TextAudioInputSchema,
        "data_item": {
            "text": "An elephant is a large gray animal with a trunk and big ears. It makes a trumpeting sound.",
            "age": 15,
            "audio_file": "tests/pytest/data/elephant.wav",
            "year": 2020,
        },
    },
    "image-audio": {
        "input_schema": ImageAudioInputSchema,
        "data_item": {
            "image_file": "tests/pytest/data/elephant.png",
            "height": 304.5,
            "audio_file": "tests/pytest/data/elephant.wav",
            "year": 2020,
        },
    },
    "text-image-audio": {
        "input_schema": TextImageAudioInputSchema,
        "data_item": {
            "text": "An elephant is a large gray animal with a trunk and big ears. It makes a trumpeting sound.",
            "age": 15,
            "image_file": "tests/pytest/data/elephant.png",
            "height": 304.5,
            "audio_file": "tests/pytest/data/elephant.wav",
            "year": 2020,
        },
    },
}

# Maps provider name to (Model enum, supported modalities)
PROVIDER_CONFIGS = {
    "anthropic": {
        "model": Model.CLAUDE_4_5_SONNET,
        "supported_modalities": ["text-only", "image-only", "text-image"],
    },
    "openai": {
        "model": Model.GPT_4o,
        "supported_modalities": ["text-only", "image-only", "text-image"],
    },
    "openai-audio": {
        "model": Model.GPT_4o_AUDIO_PREVIEW,
        "supported_modalities": ["audio-only", "text-audio"],
    },
    "gemini": {
        "model": Model.GOOGLE_GEMINI_2_5_FLASH,
        "supported_modalities": [
            "text-only", "image-only", "audio-only",
            "text-image", "text-audio", "image-audio", "text-image-audio",
        ],
    },
    "vertex_ai": {
        "model": Model.GEMINI_2_5_FLASH,
        "supported_modalities": [
            "text-only", "image-only", "audio-only",
            "text-image", "text-audio", "image-audio", "text-image-audio",
        ],
    },
}

def main():
    """Generate and save messages for all provider/modality combinations."""
    display_output_dir = "tests/pytest/data/generator_messages/"
    total_combinations = sum(
        len(provider_config["supported_modalities"])
        for provider_config in PROVIDER_CONFIGS.values()
    )

    # Count total combinations
    total_combinations = sum(
        len(provider_config["supported_modalities"])
        for provider_config in PROVIDER_CONFIGS.values()
    )

    print(f"Generating test messages for {total_combinations} provider/modality combinations...")
    print(f"Output directory: {display_output_dir}")

    generated_count = 0

    for provider, provider_config in PROVIDER_CONFIGS.items():
        model = provider_config["model"]
        supported_modalities = provider_config["supported_modalities"]

        print(f"Provider: {provider} (model: {model.value})")
        print(f"  Supported modalities: {supported_modalities}")

        for modality in supported_modalities:
            config = MODALITY_CONFIGS[modality]
            print(f"  Generating: {modality}_{provider}")

            try:
                input_schema = config["input_schema"]
                data_item = config["data_item"]
                input_record = DataRecord(input_schema(**data_item), source_indices=[0])

                # instantiate Generator
                generator = Generator(
                    model=model,
                    prompt_strategy=PromptStrategy.MAP,
                    reasoning_effort=None,
                    desc=STATIC_CONTEXT
                )

                # call generator
                generator
                output_path = save_messages(modality, provider, messages, output_dir)
                print(f"    Saved to: {output_path}")
                print(f"    Messages: {len(messages)}")

                # Print message summary
                for i, msg in enumerate(messages):
                    role = msg.get("role", "unknown")
                    msg_type = msg.get("type", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        content_len = len(content)
                    else:
                        content_len = len(str(content))
                    print(f"      [{i}] role={role}, type={msg_type}, len={content_len}")

                generated_count += 1

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        print()

    print(f"Done! Generated {generated_count}/{total_combinations} message files.")


if __name__ == "__main__":
    main()
