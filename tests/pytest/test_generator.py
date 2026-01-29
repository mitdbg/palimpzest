import os
import time
import uuid

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.query.generators.generators import Generator


def generate_session_id() -> str:
    """
    Generate a unique 12-character session ID.
    This ensures each test run has a unique prompt prefix, preventing cache hits from previous runs.
    """
    return uuid.uuid4().hex[:12].upper()


@pytest.fixture
def question():
    class Question(BaseModel):
        question: str = Field(description="A simple question")
    dr = DataRecord(data_item=Question(question="What color is grass? (one-word answer)"), source_indices=[0])
    return dr

@pytest.fixture
def output_schema():
    class Answer(BaseModel):
        answer: str = Field(description="The one-word answer to the question.")
    return Answer

@pytest.mark.parametrize(
    "model",
    [
        pytest.param(Model.GPT_4o_MINI, marks=pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not present")),
        pytest.param(Model.DEEPSEEK_V3, marks=pytest.mark.skipif(os.getenv("TOGETHER_API_KEY") is None, reason="TOGETHER_API_KEY not present")),
        pytest.param(Model.LLAMA3_2_3B, marks=pytest.mark.skipif(os.getenv("TOGETHER_API_KEY") is None, reason="TOGETHER_API_KEY not present")),
        pytest.param(Model.CLAUDE_3_5_HAIKU, marks=pytest.mark.skipif(os.getenv("ANTHROPIC_API_KEY") is None, reason="ANTHROPIC_API_KEY not present")),
    ]
)
def test_generator(model, question, output_schema):
    generator = Generator(model, PromptStrategy.MAP, None)
    output, _, gen_stats, _ = generator(question, output_schema.model_fields, **{"output_schema": output_schema})
    # Basic checks: generator produced output and tracked some stats
    assert gen_stats.output_text_tokens > 0, "Expected positive output tokens"
    assert output["answer"][0].lower() == "green"


# =============================================================================
# GENERATOR STATS VALIDATION TESTS
# =============================================================================
# These tests validate that the Generator correctly tracks token usage and costs
# for different provider/modality combinations.

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

# Input Schemas
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


class AnimalOutputSchema(BaseModel):
    """Output schema for animal identification."""
    animal: str = Field(description="The animal in the input")


# Expected stats from provider testing (to be filled in after running capture_provider_stats.py)
# Format: {(provider, modality): {"first_request": {...}, "second_request": {...}}}
EXPECTED_STATS = {
    # Anthropic - claude-sonnet-4-5-20250929
    # Note: Anthropic doesn't separate image tokens from text tokens in usage stats
    ("anthropic", "text-only"): {
        "first_request": {
            "input_text_tokens": 64,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 2065,
            "output_tokens": 230
        },
        "second_request": {
            "input_text_tokens": 64,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 2065,
            "cache_creation_tokens": 0,
            "output_tokens": 338,
        },
    },
    ("anthropic", "image-only"): {
        "first_request": {
            "input_text_tokens": 247,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 2205,
            "output_tokens": 472,
        },
        "second_request": {
            "input_text_tokens": 247,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 2205,
            "cache_creation_tokens": 0,
            "output_tokens": 393,
        },
    },
    # OpenAI - gpt-4o-2024-08-06
    ("openai", "text-only"): {
        "first_request": {
            "input_text_tokens": 1856,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 131,
        },
        "second_request": {
            "input_text_tokens": 832,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 1024,
            "cache_creation_tokens": 0,
            "output_tokens": 88,
        },
    },
    ("openai", "image-only"): {
        "first_request": {
            "input_text_tokens": 2220,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 85,
        },
        "second_request": {
            "input_text_tokens": 428,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 1792,
            "cache_creation_tokens": 0,
            "output_tokens": 75,
        },
    },
    # OpenAI Audio - gpt-4o-audio-preview
    ("openai-audio", "audio-only"): {
        "first_request": {
            "input_text_tokens": 1974,
            "input_image_tokens": 0,
            "input_audio_tokens": 31,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 100,
        },
        "second_request": {
            "input_text_tokens": 1974,
            "input_image_tokens": 0,
            "input_audio_tokens": 31,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 166,
        },
    },
    # Gemini - gemini-2.5-flash
    ("gemini", "text-only"): {
        "first_request": {
            "input_text_tokens": 1923,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 61,
        },
        "second_request": {
            "input_text_tokens": 913,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 1010,
            "cache_creation_tokens": 0,
            "output_tokens": 74,
        },
    },
    ("gemini", "image-only"): {
        "first_request": {
            "input_text_tokens": 2045,
            "input_image_tokens": 258,
            "input_audio_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 91,
        },
        "second_request": {
            "input_text_tokens": 247,
            "input_image_tokens": 32,
            "input_audio_tokens": 0,
            "cache_read_tokens": 2024,
            "cache_creation_tokens": 0,
            "output_tokens": 104,
        },
    },
    ("gemini", "audio-only"): {
        "first_request": {
            "input_text_tokens": 2040,
            "input_image_tokens": 0,
            "input_audio_tokens": 100,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 125,
        },
        "second_request": {
            "input_text_tokens": 117,
            "input_image_tokens": 0,
            "input_audio_tokens": 6,
            "cache_read_tokens": 2017,
            "cache_creation_tokens": 0,
            "output_tokens": 125,
        },
    },
    ("gemini", "text-image-audio"): {
        "first_request": {
            "input_text_tokens": 2262,
            "input_image_tokens": 258,
            "input_audio_tokens": 100,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 118,
        },
        "second_request": {
            "input_text_tokens": 516,
            "input_image_tokens": 59,
            "input_audio_tokens": 23,
            "cache_read_tokens": 2022,
            "cache_creation_tokens": 0,
            "output_tokens": 181,
        },
    },
    # Vertex AI - gemini-2.5-flash
    ("vertex_ai", "text-only"): {
        "first_request": {
            "input_text_tokens": None,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
        "second_request": {
            "input_text_tokens": None,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
    },
    ("vertex_ai", "image-only"): {
        "first_request": {
            "input_text_tokens": None,
            "input_image_tokens": None,
            "input_audio_tokens": 0,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
        "second_request": {
            "input_text_tokens": None,
            "input_image_tokens": None,
            "input_audio_tokens": 0,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
    },
    ("vertex_ai", "audio-only"): {
        "first_request": {
            "input_text_tokens": None,
            "input_image_tokens": 0,
            "input_audio_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
        "second_request": {
            "input_text_tokens": None,
            "input_image_tokens": 0,
            "input_audio_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
    },
    ("vertex_ai", "text-image-audio"): {
        "first_request": {
            "input_text_tokens": None,
            "input_image_tokens": None,
            "input_audio_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
        "second_request": {
            "input_text_tokens": None,
            "input_image_tokens": None,
            "input_audio_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": 0,
            "output_tokens": None,
        },
    },
}


def create_input_record(input_schema, modality: str):
    """Create an input DataRecord for the given schema and modality."""
    data_item = {}

    # Add text fields if applicable
    if "text" in modality or input_schema == TextInputSchema and hasattr(input_schema, "model_fields"):
            if "text" in input_schema.model_fields:
                data_item["text"] = "An elephant is a large gray animal with a trunk and big ears. It makes a trumpeting sound."
            if "age" in input_schema.model_fields:
                data_item["age"] = 15

    # Add image fields if applicable
    if "image" in modality or input_schema == ImageInputSchema and hasattr(input_schema, "model_fields"):
            if "image_file" in input_schema.model_fields:
                data_item["image_file"] = "tests/pytest/data/elephant.png"
            if "height" in input_schema.model_fields:
                data_item["height"] = 304.5

    # Add audio fields if applicable
    if "audio" in modality or input_schema == AudioInputSchema and hasattr(input_schema, "model_fields"):
            if "audio_file" in input_schema.model_fields:
                data_item["audio_file"] = "tests/pytest/data/elephant.wav"
            if "year" in input_schema.model_fields:
                data_item["year"] = 2020

    return DataRecord(input_schema(**data_item), source_indices=[0])


def get_model_for_provider(provider: str) -> Model:
    """Get the Model enum for a given provider."""
    if provider == "anthropic":
        return Model.CLAUDE_4_5_SONNET
    elif provider == "openai":
        return Model.GPT_4o
    elif provider == "openai-audio":
        return Model.GPT_4o_AUDIO_PREVIEW
    elif provider == "gemini":
        return Model.GOOGLE_GEMINI_2_5_FLASH
    elif provider == "vertex_ai":
        return Model.GEMINI_2_5_FLASH
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_input_schema_for_modality(modality: str):
    """Get the input schema class for a given modality."""
    schema_map = {
        "text-only": TextInputSchema,
        "image-only": ImageInputSchema,
        "audio-only": AudioInputSchema,
        "text-image": TextImageInputSchema,
        "text-audio": TextAudioInputSchema,
        "image-audio": ImageAudioInputSchema,
        "text-image-audio": TextImageAudioInputSchema,
    }
    return schema_map[modality]


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================
PROVIDER_CONFIG = {
    "anthropic": {
        "model": Model.CLAUDE_4_5_SONNET,
        "supported_modalities": ["text-only", "image-only"],
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "model": Model.GPT_4o,
        "supported_modalities": ["text-only", "image-only"],
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai-audio": {
        "model": Model.GPT_4o_AUDIO_PREVIEW,
        "supported_modalities": ["audio-only"],
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "model": Model.GOOGLE_GEMINI_2_5_FLASH,
        "supported_modalities": ["text-only", "image-only", "audio-only", "text-image-audio"],
        "api_key_env": "GOOGLE_API_KEY",
    },
    "vertex_ai": {
        "model": Model.GEMINI_2_5_FLASH,
        "supported_modalities": ["text-only", "image-only", "audio-only", "text-image-audio"],
        "api_key_env": ["GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_PROJECT"],
    },
}

ALL_MODALITIES = ["text-only", "image-only", "audio-only", "text-image-audio"]
ALL_PROVIDERS = ["anthropic", "openai", "openai-audio", "gemini", "vertex_ai"]

CACHE_WAIT_SECONDS = 10


def check_api_key(provider: str) -> bool:
    """Check if the API key for a provider is present."""
    config = PROVIDER_CONFIG[provider]
    api_key_env = config["api_key_env"]
    if isinstance(api_key_env, list):
        return any(os.getenv(key) is not None for key in api_key_env)
    return os.getenv(api_key_env) is not None


def is_modality_supported(provider: str, modality: str) -> bool:
    """Check if a modality is supported by a provider."""
    return modality in PROVIDER_CONFIG[provider]["supported_modalities"]


def within_tolerance(actual: int, expected: int, tolerance: float = 0.05) -> bool:
    """Check if actual value is within tolerance of expected value."""
    if expected == 0:
        return actual == 0
    margin = max(1, int(expected * tolerance))  # At least 1 token margin
    return abs(actual - expected) <= margin


def assert_stats_match(gen_stats, expected: dict, request_name: str, provider: str = "", tolerance: float = 0.05):
    """Assert that generation stats match expected values within tolerance.

    For OpenAI, cache hits are non-deterministic (prefix caching depends on server-side
    shard routing). So for cache_read_tokens we accept anywhere in [0, expected+5%],
    and input_text_tokens is validated as: total logical input ≈ input_text + cache_read.
    """
    is_openai = provider.startswith("openai")

    if is_openai and expected.get("cache_read_tokens") is not None and expected["cache_read_tokens"] > 0:
        # OpenAI: cache hit is non-deterministic, accept 0..expected+tolerance
        expected_cache = expected["cache_read_tokens"]
        cache_upper = expected_cache + max(1, int(expected_cache * tolerance))
        assert 0 <= gen_stats.cache_read_tokens <= cache_upper, \
            f"{request_name} cache_read_tokens out of range: got {gen_stats.cache_read_tokens}, expected 0..{cache_upper}"

        # input_text_tokens + cache_read_tokens should equal the logical total
        expected_logical_total = expected["input_text_tokens"] + expected["cache_read_tokens"]
        actual_logical_total = gen_stats.input_text_tokens + gen_stats.cache_read_tokens
        assert within_tolerance(actual_logical_total, expected_logical_total, tolerance), \
            f"{request_name} logical total input mismatch: got {actual_logical_total} (input={gen_stats.input_text_tokens} + cache={gen_stats.cache_read_tokens}), expected ~{expected_logical_total}"
    else:
        if expected.get("input_text_tokens") is not None:
            assert within_tolerance(gen_stats.input_text_tokens, expected["input_text_tokens"], tolerance), \
                f"{request_name} input_text_tokens mismatch: got {gen_stats.input_text_tokens}, expected {expected['input_text_tokens']} (±{tolerance*100}%)"

        if expected.get("cache_read_tokens") is not None:
            assert within_tolerance(gen_stats.cache_read_tokens, expected["cache_read_tokens"], tolerance), \
                f"{request_name} cache_read_tokens mismatch: got {gen_stats.cache_read_tokens}, expected {expected['cache_read_tokens']} (±{tolerance*100}%)"

    if expected.get("input_image_tokens") is not None:
        assert within_tolerance(gen_stats.input_image_tokens, expected["input_image_tokens"], tolerance), \
            f"{request_name} input_image_tokens mismatch: got {gen_stats.input_image_tokens}, expected {expected['input_image_tokens']} (±{tolerance*100}%)"

    if expected.get("input_audio_tokens") is not None:
        assert within_tolerance(gen_stats.input_audio_tokens, expected["input_audio_tokens"], tolerance), \
            f"{request_name} input_audio_tokens mismatch: got {gen_stats.input_audio_tokens}, expected {expected['input_audio_tokens']} (±{tolerance*100}%)"

    if expected.get("cache_creation_tokens") is not None:
        assert within_tolerance(gen_stats.cache_creation_tokens, expected["cache_creation_tokens"], tolerance), \
            f"{request_name} cache_creation_tokens mismatch: got {gen_stats.cache_creation_tokens}, expected {expected['cache_creation_tokens']} (±{tolerance*100}%)"

    assert gen_stats.output_text_tokens > 0, f"{request_name} output_text_tokens should be positive"
    assert gen_stats.cost_per_record > 0, f"{request_name} cost_per_record should be positive"


# =============================================================================
# COMBINED GENERATOR STATS TEST
# =============================================================================
@pytest.mark.parametrize(
    "provider,modality",
    [(p, m) for p in ALL_PROVIDERS for m in ALL_MODALITIES],
    ids=[f"{p}-{m}" for p in ALL_PROVIDERS for m in ALL_MODALITIES],
)
def test_generator_stats(provider, modality):
    """Test Generator stats tracking for all provider/modality combinations.

    Makes two requests:
    1. First request (no cache) - should show cache_creation_tokens for providers that support it
    2. Second request (with cache) - should show cache_read_tokens after waiting for cache availability
    """
    # Skip if modality not supported by provider
    if not is_modality_supported(provider, modality):
        pytest.skip(f"Modality {modality} not supported by {provider}")

    # Skip if API key not present
    if not check_api_key(provider):
        config = PROVIDER_CONFIG[provider]
        pytest.skip(f"API key not present: {config['api_key_env']}")

    # Get model and create input
    model = PROVIDER_CONFIG[provider]["model"]
    input_schema = get_input_schema_for_modality(modality)
    input_record = create_input_record(input_schema, modality)
    session_id = generate_session_id()

    # Create generator
    generator = Generator(model, PromptStrategy.MAP, None, desc=STATIC_CONTEXT)

    # Get expected stats
    expected = EXPECTED_STATS.get((provider, modality), {})

    # First request (no cache)
    output1, _, gen_stats1, _ = generator(
        input_record,
        AnimalOutputSchema.model_fields,
        output_schema=AnimalOutputSchema,
        cache_isolation_id=session_id,
    )

    # Verify first request output
    assert output1 is not None
    assert "animal" in output1

    # Assert first request stats
    if "first_request" in expected:
        assert_stats_match(gen_stats1, expected["first_request"], "first_request", provider=provider)

    # Wait for cache to be available
    time.sleep(CACHE_WAIT_SECONDS)

    # Second request (should use cache)
    output2, _, gen_stats2, _ = generator(
        input_record,
        AnimalOutputSchema.model_fields,
        output_schema=AnimalOutputSchema,
        cache_isolation_id=session_id,
    )

    # Verify second request output
    assert output2 is not None
    assert "animal" in output2

    # Assert second request stats
    if "second_request" in expected:
        assert_stats_match(gen_stats2, expected["second_request"], "second_request", provider=provider)
