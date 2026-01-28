import hashlib
import os

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.query.generators.generators import Generator


def generate_session_id(provider: str, modality: str) -> str:
    """
    Generate a unique 12-character session ID for a provider/modality combination.
    This ensures each modality test has a unique prompt prefix, preventing cross-modality cache hits.
    The ID is deterministic based on provider+modality so repeated runs produce consistent results.
    """
    hash_input = f"{provider}_{modality}"
    hash_hex = hashlib.md5(hash_input.encode()).hexdigest()
    return hash_hex[:12].upper()


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
    assert (gen_stats.input_text_tokens + gen_stats.cache_read_tokens + gen_stats.cache_creation_tokens) > 0
    assert gen_stats.output_text_tokens > 0
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
# Format: {(provider, modality): {"input_text_tokens": X, "input_image_tokens": X, "input_audio_tokens": X, ...}}
EXPECTED_STATS = {
    # Anthropic - claude-sonnet-4-5-20250929
    ("anthropic", "text-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": 0,
    },
    ("anthropic", "image-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    ("anthropic", "text-image"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    # OpenAI - gpt-4o-2024-08-06
    ("openai", "text-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": 0,
    },
    ("openai", "image-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    ("openai", "text-image"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    # OpenAI Audio - gpt-4o-audio-preview
    ("openai-audio", "audio-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    ("openai-audio", "text-audio"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    # Gemini - gemini-2.5-flash (all seven modalities)
    ("gemini", "text-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": 0,
    },
    ("gemini", "image-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    ("gemini", "audio-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    ("gemini", "text-image"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    ("gemini", "text-audio"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    ("gemini", "image-audio"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    ("gemini", "text-image-audio"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    # Vertex AI - gemini-2.5-flash (all seven modalities)
    ("vertex_ai", "text-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": 0,
    },
    ("vertex_ai", "image-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    ("vertex_ai", "audio-only"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    ("vertex_ai", "text-image"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": 0,
    },
    ("vertex_ai", "text-audio"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": 0,
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    ("vertex_ai", "image-audio"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
    ("vertex_ai", "text-image-audio"): {
        "input_text_tokens": None,  # TODO: Fill in from provider stats
        "input_image_tokens": None,  # TODO: Fill in from provider stats
        "input_audio_tokens": None,  # TODO: Fill in from provider stats
    },
}


def create_input_record(input_schema, modality: str):
    """Create an input DataRecord for the given schema and modality."""
    data_item = {}

    # Add text fields if applicable
    if "text" in modality or input_schema == TextInputSchema:
        if hasattr(input_schema, "model_fields"):
            if "text" in input_schema.model_fields:
                data_item["text"] = "An elephant is a large gray animal with a trunk and big ears. It makes a trumpeting sound."
            if "age" in input_schema.model_fields:
                data_item["age"] = 15

    # Add image fields if applicable
    if "image" in modality or input_schema == ImageInputSchema:
        if hasattr(input_schema, "model_fields"):
            if "image_file" in input_schema.model_fields:
                data_item["image_file"] = "tests/pytest/data/elephant.png"
            if "height" in input_schema.model_fields:
                data_item["height"] = 304.5

    # Add audio fields if applicable
    if "audio" in modality or input_schema == AudioInputSchema:
        if hasattr(input_schema, "model_fields"):
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
# ANTHROPIC TESTS (text, image, text+image only - no audio support)
# =============================================================================
@pytest.mark.parametrize(
    "modality",
    ["text-only", "image-only", "text-image"],
    ids=["text-only", "image-only", "text-image"],
)
@pytest.mark.skipif(os.getenv("ANTHROPIC_API_KEY") is None, reason="ANTHROPIC_API_KEY not present")
def test_generator_stats_anthropic(modality):
    """Test Generator stats tracking for Anthropic models."""
    model = Model.CLAUDE_4_5_SONNET
    input_schema = get_input_schema_for_modality(modality)
    input_record = create_input_record(input_schema, modality)
    session_id = generate_session_id("anthropic", modality)

    generator = Generator(model, PromptStrategy.MAP, None, desc=STATIC_CONTEXT)
    output, _, gen_stats, _ = generator(
        input_record,
        AnimalOutputSchema.model_fields,
        output_schema=AnimalOutputSchema,
        cache_isolation_id=session_id,
    )

    # Verify output
    assert output is not None
    assert "animal" in output

    # Get expected stats
    expected = EXPECTED_STATS.get(("anthropic", modality), {})

    # Assert input token counts match expected values
    if expected.get("input_text_tokens") is not None:
        assert gen_stats.input_text_tokens == expected["input_text_tokens"], \
            f"input_text_tokens mismatch: got {gen_stats.input_text_tokens}, expected {expected['input_text_tokens']}"

    if expected.get("input_image_tokens") is not None:
        assert gen_stats.input_image_tokens == expected["input_image_tokens"], \
            f"input_image_tokens mismatch: got {gen_stats.input_image_tokens}, expected {expected['input_image_tokens']}"

    assert gen_stats.input_audio_tokens == 0, "Anthropic should have 0 audio tokens"

    # Assert output tokens are positive
    assert gen_stats.output_text_tokens > 0, "output_text_tokens should be positive"

    # Assert cost_per_record is calculated correctly
    # cost = input_cost + output_cost + cache_costs
    assert gen_stats.cost_per_record > 0, "cost_per_record should be positive"


# =============================================================================
# OPENAI TESTS (text, image, text+image)
# =============================================================================
@pytest.mark.parametrize(
    "modality",
    ["text-only", "image-only", "text-image"],
    ids=["text-only", "image-only", "text-image"],
)
@pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not present")
def test_generator_stats_openai(modality):
    """Test Generator stats tracking for OpenAI models (non-audio)."""
    model = Model.GPT_4o
    input_schema = get_input_schema_for_modality(modality)
    input_record = create_input_record(input_schema, modality)
    session_id = generate_session_id("openai", modality)

    generator = Generator(model, PromptStrategy.MAP, None, desc=STATIC_CONTEXT)
    output, _, gen_stats, _ = generator(
        input_record,
        AnimalOutputSchema.model_fields,
        output_schema=AnimalOutputSchema,
        cache_isolation_id=session_id,
    )

    # Verify output
    assert output is not None
    assert "animal" in output

    # Get expected stats
    expected = EXPECTED_STATS.get(("openai", modality), {})

    # Assert input token counts match expected values
    if expected.get("input_text_tokens") is not None:
        assert gen_stats.input_text_tokens == expected["input_text_tokens"], \
            f"input_text_tokens mismatch: got {gen_stats.input_text_tokens}, expected {expected['input_text_tokens']}"

    if expected.get("input_image_tokens") is not None:
        assert gen_stats.input_image_tokens == expected["input_image_tokens"], \
            f"input_image_tokens mismatch: got {gen_stats.input_image_tokens}, expected {expected['input_image_tokens']}"

    assert gen_stats.input_audio_tokens == 0, "Non-audio OpenAI model should have 0 audio tokens"

    # Assert output tokens are positive
    assert gen_stats.output_text_tokens > 0, "output_text_tokens should be positive"

    # Assert cost_per_record is calculated correctly
    assert gen_stats.cost_per_record > 0, "cost_per_record should be positive"


# =============================================================================
# OPENAI AUDIO TESTS (audio-only, text+audio)
# =============================================================================
@pytest.mark.parametrize(
    "modality",
    ["audio-only", "text-audio"],
    ids=["audio-only", "text-audio"],
)
@pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not present")
def test_generator_stats_openai_audio(modality):
    """Test Generator stats tracking for OpenAI audio models."""
    model = Model.GPT_4o_AUDIO_PREVIEW
    input_schema = get_input_schema_for_modality(modality)
    input_record = create_input_record(input_schema, modality)
    session_id = generate_session_id("openai-audio", modality)

    generator = Generator(model, PromptStrategy.MAP, None, desc=STATIC_CONTEXT)
    output, _, gen_stats, _ = generator(
        input_record,
        AnimalOutputSchema.model_fields,
        output_schema=AnimalOutputSchema,
        cache_isolation_id=session_id,
    )

    # Verify output
    assert output is not None
    assert "animal" in output

    # Get expected stats
    expected = EXPECTED_STATS.get(("openai-audio", modality), {})

    # Assert input token counts match expected values
    if expected.get("input_text_tokens") is not None:
        assert gen_stats.input_text_tokens == expected["input_text_tokens"], \
            f"input_text_tokens mismatch: got {gen_stats.input_text_tokens}, expected {expected['input_text_tokens']}"

    if expected.get("input_audio_tokens") is not None:
        assert gen_stats.input_audio_tokens == expected["input_audio_tokens"], \
            f"input_audio_tokens mismatch: got {gen_stats.input_audio_tokens}, expected {expected['input_audio_tokens']}"

    assert gen_stats.input_image_tokens == 0, "Audio model should have 0 image tokens"

    # Assert output tokens are positive
    assert gen_stats.output_text_tokens > 0, "output_text_tokens should be positive"

    # Assert cost_per_record is calculated correctly
    assert gen_stats.cost_per_record > 0, "cost_per_record should be positive"


# =============================================================================
# GEMINI TESTS (all seven modality combinations)
# =============================================================================
@pytest.mark.parametrize(
    "modality",
    ["text-only", "image-only", "audio-only", "text-image", "text-audio", "image-audio", "text-image-audio"],
    ids=["text-only", "image-only", "audio-only", "text-image", "text-audio", "image-audio", "text-image-audio"],
)
@pytest.mark.skipif(os.getenv("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not present")
def test_generator_stats_gemini(modality):
    """Test Generator stats tracking for Gemini models (Google AI Studio)."""
    model = Model.GOOGLE_GEMINI_2_5_FLASH
    input_schema = get_input_schema_for_modality(modality)
    input_record = create_input_record(input_schema, modality)
    session_id = generate_session_id("gemini", modality)

    generator = Generator(model, PromptStrategy.MAP, None, desc=STATIC_CONTEXT)
    output, _, gen_stats, _ = generator(
        input_record,
        AnimalOutputSchema.model_fields,
        output_schema=AnimalOutputSchema,
        cache_isolation_id=session_id,
    )

    # Verify output
    assert output is not None
    assert "animal" in output

    # Get expected stats
    expected = EXPECTED_STATS.get(("gemini", modality), {})

    # Assert input token counts match expected values
    if expected.get("input_text_tokens") is not None:
        assert gen_stats.input_text_tokens == expected["input_text_tokens"], \
            f"input_text_tokens mismatch: got {gen_stats.input_text_tokens}, expected {expected['input_text_tokens']}"

    if expected.get("input_image_tokens") is not None:
        assert gen_stats.input_image_tokens == expected["input_image_tokens"], \
            f"input_image_tokens mismatch: got {gen_stats.input_image_tokens}, expected {expected['input_image_tokens']}"

    if expected.get("input_audio_tokens") is not None:
        assert gen_stats.input_audio_tokens == expected["input_audio_tokens"], \
            f"input_audio_tokens mismatch: got {gen_stats.input_audio_tokens}, expected {expected['input_audio_tokens']}"

    # Assert output tokens are positive
    assert gen_stats.output_text_tokens > 0, "output_text_tokens should be positive"

    # Assert cost_per_record is calculated correctly
    assert gen_stats.cost_per_record > 0, "cost_per_record should be positive"


# =============================================================================
# VERTEX AI TESTS (all seven modality combinations)
# =============================================================================
@pytest.mark.parametrize(
    "modality",
    ["text-only", "image-only", "audio-only", "text-image", "text-audio", "image-audio", "text-image-audio"],
    ids=["text-only", "image-only", "audio-only", "text-image", "text-audio", "image-audio", "text-image-audio"],
)
@pytest.mark.skipif(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None and os.getenv("VERTEX_PROJECT") is None,
    reason="Vertex AI credentials not present"
)
def test_generator_stats_vertex_ai(modality):
    """Test Generator stats tracking for Vertex AI Gemini models."""
    model = Model.GEMINI_2_5_FLASH
    input_schema = get_input_schema_for_modality(modality)
    input_record = create_input_record(input_schema, modality)
    session_id = generate_session_id("vertex_ai", modality)

    generator = Generator(model, PromptStrategy.MAP, None, desc=STATIC_CONTEXT)
    output, _, gen_stats, _ = generator(
        input_record,
        AnimalOutputSchema.model_fields,
        output_schema=AnimalOutputSchema,
        cache_isolation_id=session_id,
    )

    # Verify output
    assert output is not None
    assert "animal" in output

    # Get expected stats
    expected = EXPECTED_STATS.get(("vertex_ai", modality), {})

    # Assert input token counts match expected values
    if expected.get("input_text_tokens") is not None:
        assert gen_stats.input_text_tokens == expected["input_text_tokens"], \
            f"input_text_tokens mismatch: got {gen_stats.input_text_tokens}, expected {expected['input_text_tokens']}"

    if expected.get("input_image_tokens") is not None:
        assert gen_stats.input_image_tokens == expected["input_image_tokens"], \
            f"input_image_tokens mismatch: got {gen_stats.input_image_tokens}, expected {expected['input_image_tokens']}"

    if expected.get("input_audio_tokens") is not None:
        assert gen_stats.input_audio_tokens == expected["input_audio_tokens"], \
            f"input_audio_tokens mismatch: got {gen_stats.input_audio_tokens}, expected {expected['input_audio_tokens']}"

    # Assert output tokens are positive
    assert gen_stats.output_text_tokens > 0, "output_text_tokens should be positive"

    # Assert cost_per_record is calculated correctly
    assert gen_stats.cost_per_record > 0, "cost_per_record should be positive"
