#!/usr/bin/env python3
"""
Script to generate test messages for each provider/modality combination.

This script uses the Generator class directly to create message payloads.
It uses the 'generating_messages_only' flag to retrieve the exact messages
that would be sent to the provider without making an actual API call.

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

def generate_session_id(provider: str, modality: str) -> str:
    """
    Generate a unique 12-character session ID for a provider/modality combination.
    This ensures each modality test has a unique prompt prefix, preventing cross-modality cache hits.
    The ID is deterministic based on provider+modality so regenerating produces consistent results.
    """
    import hashlib
    hash_input = f"{provider}_{modality}"
    hash_hex = hashlib.md5(hash_input.encode()).hexdigest()
    return hash_hex[:12].upper()

STATIC_CONTEXT = """
WILDLIFE CONSERVATION & RESEARCH CENTER: SPECIES IDENTIFICATION MANUAL (v2025.1)

SECTION 1: INTRODUCTION AND MISSION
The Wildlife Conservation & Research Center (WCRC) is dedicated to the preservation, study, and rehabilitation of diverse wildlife species.
All staff members, researchers, and volunteers must adhere to these protocols for accurate species identification and data collection.
Our mission combines advanced biological sciences with conservation efforts to protect endangered and threatened populations worldwide.

SECTION 2: MAMMAL IDENTIFICATION PROTOCOLS

2.1 ELEPHANTS (Family Elephantidae):
    - African Savanna Elephant: Larger ears (shaped like Africa), concave back, two fingers on trunk tip. Weight: 5,000-14,000 lbs.
    - African Forest Elephant: Smaller stature, oval-shaped ears, straighter tusks pointing downward.
    - Asian Elephant: Smaller ears, convex back, one finger on trunk tip, twin domes on head. Weight: 4,000-11,000 lbs.
    - Vocalizations: Trumpeting (alarm/excitement), rumbling (long-distance communication), roaring (distress).

2.2 BIG CATS (Family Felidae):
    - Lion (Panthera leo): Tawny coat, males have distinctive mane. Social, live in prides. Height: 3.5-4 ft at shoulder.
    - Tiger (Panthera tigris): Orange coat with black stripes, white underbelly. Solitary hunters. Largest cat species.
    - Leopard (Panthera pardus): Golden-yellow coat with rosette patterns. Excellent climbers, often cache prey in trees.
    - Cheetah (Acinonyx jubatus): Spotted coat, black "tear marks" from eyes to mouth. Fastest land animal (70 mph).
    - Vocalizations: Roaring (lions, tigers, leopards), chirping/purring (cheetahs cannot roar).

2.3 BEARS (Family Ursidae):
    - Brown Bear (Ursus arctos): Large shoulder hump, dish-shaped face, long claws. Includes grizzly subspecies.
    - Black Bear (Ursus americanus): Straight facial profile, no shoulder hump, shorter claws. Most common North American bear.
    - Polar Bear (Ursus maritimus): White fur, longer neck, smaller ears. Marine mammal adapted to Arctic conditions.
    - Giant Panda (Ailuropoda melanoleuca): Black and white coloring, feeds almost exclusively on bamboo.
    - Vocalizations: Roaring, growling, huffing, jaw-popping (threat displays).

2.4 PRIMATES (Order Primates):
    - Gorilla: Largest primate, silver-back males, knuckle-walking locomotion. Vocalizations include chest-beating, hooting.
    - Chimpanzee: Highly intelligent, uses tools, complex social structures. Vocalizations: pant-hoots, screams.
    - Orangutan: Red-orange fur, arboreal lifestyle, solitary. Long calls can travel over 1 km.
    - Gibbon: Smaller apes, brachiation locomotion, distinctive whooping songs for territorial marking.

SECTION 3: BIRD IDENTIFICATION PROTOCOLS

3.1 RAPTORS (Order Accipitriformes/Falconiformes):
    - Bald Eagle: White head and tail, yellow beak. Wingspan: 6-7.5 ft. Call: high-pitched chattering.
    - Golden Eagle: Dark brown plumage, golden nape. Powerful hunters of small mammals.
    - Peregrine Falcon: Blue-gray back, barred underparts. Fastest bird in dive (240+ mph).
    - Red-tailed Hawk: Brown back, pale underparts, distinctive red tail. Most common North American hawk.

3.2 PARROTS (Order Psittaciformes):
    - Macaw: Large, colorful, long tail feathers. Powerful curved beaks. Highly social and vocal.
    - African Grey: Gray plumage, red tail. Exceptional mimicry and cognitive abilities.
    - Cockatoo: White or pink plumage, distinctive crest. Loud screeching vocalizations.

SECTION 4: REPTILE IDENTIFICATION PROTOCOLS

4.1 CROCODILIANS (Order Crocodilia):
    - American Alligator: Broad, U-shaped snout, dark coloration. Freshwater habitats.
    - Nile Crocodile: V-shaped snout, aggressive. Can reach 16-18 ft in length.
    - Gharial: Extremely narrow snout, fish-eating specialist. Critically endangered.

4.2 LARGE SNAKES (Families Pythonidae/Boidae):
    - Reticulated Python: Longest snake species (up to 23 ft), complex geometric patterns.
    - Green Anaconda: Heaviest snake species, olive-green with black spots. Semi-aquatic.
    - King Cobra: Longest venomous snake (up to 18 ft), distinctive hood when threatened.

SECTION 5: DATA COLLECTION AND ANALYSIS

5.1 Visual Identification:
    - Document body shape, size, coloration, and distinctive markings.
    - Note behavioral characteristics and habitat context.
    - Use standardized photography protocols for pattern matching.

5.2 Audio Identification:
    - Record vocalizations with frequency analysis equipment.
    - Tag recordings with behavioral context (territorial, mating, alarm, social).
    - Cross-reference with vocalization databases for species confirmation.

5.3 Biometric Data:
    - Record body measurements according to species-specific protocols.
    - Document age indicators (teeth wear, plumage, etc.).
    - Collect genetic samples when possible for lineage verification.

You are an AI Research Assistant for the WCRC. Your job is to analyze data inputs (text descriptions, images, and/or audio recordings) and identify the species based on the characteristics described in this manual.
Analyze all provided inputs and determine the most likely species identification.
"""

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


def save_messages(modality: str, provider: str, messages: list[dict], output_dir: str) -> str:
    """
    Save messages to a JSON file.

    Args:
        modality: Modality name
        provider: Provider name
        messages: List of message dicts
        output_dir: Directory to save files

    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{modality}_{provider}.json")

    # Convert messages to JSON-serializable format
    serializable_messages = []
    for msg in messages:
        serializable_msg = msg.copy()
        serializable_messages.append(serializable_msg)

    with open(output_path, "w") as f:
        json.dump(serializable_messages, f, indent=2, default=str)

    return output_path


def main():
    """Generate and save messages for all provider/modality combinations."""
    # Ensure the output directory follows the repository structure
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "tests",
        "pytest",
        "data",
        "generator_messages",
    )
    output_dir = os.path.abspath(output_dir)

    # Count total combinations
    total_combinations = sum(
        len(provider_config["supported_modalities"])
        for provider_config in PROVIDER_CONFIGS.values()
    )

    print(f"Generating test messages for {total_combinations} provider/modality combinations...")
    print(f"Output directory: {output_dir}")
    print(f"Static context length: ~{len(STATIC_CONTEXT.split())} words\n")

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
                # Prepare input record
                input_schema = config["input_schema"]
                data_item = config["data_item"]
                input_record = DataRecord(input_schema(**data_item), source_indices=[0])

                # Instantiate Generator
                generator = Generator(
                    model=model,
                    prompt_strategy=PromptStrategy.MAP,
                    reasoning_effort=None,
                    desc=STATIC_CONTEXT,
                )

                # Generate unique session ID for this provider/modality to prevent cross-modality cache hits
                session_id = generate_session_id(provider, modality)

                # Call the generator with the new flag
                # Pass cache_isolation_id to inject session ID at start of system/user prompts
                messages = generator(
                    candidate=input_record,
                    fields=OutputSchema.model_fields,
                    output_schema=OutputSchema,
                    generating_messages_only=True,
                    cache_isolation_id=session_id,
                )

                # Manually save the messages using local helper
                output_path = save_messages(modality, provider, messages, output_dir)

                print(f"    Session ID: {session_id}")
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