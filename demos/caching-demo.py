#!/usr/bin/env python3
"""
Realistic Demo showcasing prompt caching capabilities in Palimpzest.

This demo processes multiple employee travel requests against a comprehensive 
Corporate Travel Policy. The policy text (~2000 tokens) is included in the 
system prompt, creating a realistic scenario for prompt caching where a large 
static context is reused across multiple dynamic inputs.

Workload:
- Context: A lengthy 10-page Corporate Travel & Expense Policy.
- Input: Short email requests from employees.
- Task: Analyze each request for policy compliance, identifying violations and reimbursable amounts.

Supported caching providers:
- OpenAI (GPT-4o, GPT-4o-mini): Automatic prefix caching
- Anthropic (Claude 3.5 Sonnet/Haiku): Explicit cache_control markers
- Gemini: Implicit caching
- DeepSeek: Context caching
"""
import argparse
import os
import time
from typing import List

from dotenv import load_dotenv

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.core.lib.schemas import TextFile

load_dotenv()

# =============================================================================
# MOCK DATA: CORPORATE TRAVEL POLICY (Static Context > 1024 tokens)
# =============================================================================
CORPORATE_TRAVEL_POLICY = """
GLOBAL CORP TRAVEL & EXPENSE POLICY (v2024.1)

SECTION 1: OVERVIEW AND PHILOSOPHY
Global Corp expects employees to act responsibly and professionally when incurring and submitting costs. 
The company will reimburse employees for reasonable and necessary expenses incurred during approved business travel. 
This policy applies to all employees, contractors, and consultants.

SECTION 2: AIR TRAVEL
2.1 Booking Window: All domestic flights must be booked at least 14 days in advance. International flights must be booked 21 days in advance.
2.2 Class of Service:
    - Economy Class: Required for all domestic flights under 6 hours.
    - Premium Economy: Allowed for domestic flights over 6 hours or international flights under 8 hours.
    - Business Class: Allowed for international flights exceeding 8 hours duration.
    - First Class: Strictly prohibited unless approved by the CEO.
2.3 Ancillary Fees:
    - Checked Bags: Up to two bags reimbursed for trips > 3 days. One bag for trips <= 3 days.
    - Wi-Fi: Reimbursed only if business justification is provided (e.g., "urgent client deadline").
    - Seat Selection: Fees > $50 require VP approval.

SECTION 3: LODGING
3.1 Hotel Caps (Nightly Rates excluding taxes):
    - Tier 1 Cities (NY, London, Tokyo, SF, Zurich): $350 USD
    - Tier 2 Cities (Chicago, Paris, Berlin, Austin): $250 USD
    - All Other Locations: $175 USD
3.2 Room Type: Standard single rooms only. Suites are prohibited.
3.3 Laundry: Reasonable laundry expenses reimbursed for trips exceeding 5 consecutive nights.

SECTION 4: MEALS AND ENTERTAINMENT
4.1 Daily Meal Allowance (Per Diem):
    - Tier 1 Cities: $100/day
    - Tier 2 Cities: $75/day
    - Others: $60/day
4.2 Client Entertainment:
    - Must include at least one current or prospective client.
    - Cap is $150 per person (including employees).
    - Names and affiliations of all attendees must be documented.
4.3 Alcohol:
    - Reimbursable only with dinner.
    - Moderate consumption allowed (max 2 drinks per person).
    - "Top Shelf" liquors prohibited.

SECTION 5: GROUND TRANSPORTATION
5.1 Ride Share/Taxi: Preferred mode for travel between airport and hotel.
5.2 Car Rentals:
    - Class: Intermediate/Mid-size or smaller.
    - Insurance: Decline CDW/LDW (covered by corporate policy).
    - Fuel: Pre-paid fuel options are prohibited; cars must be returned full.
5.3 Rail: Economy/Standard class only. Acela Business Class permitted for Northeast Corridor travel.

SECTION 6: MISCELLANEOUS
6.1 Tipping:
    - Meals: 15-20%
    - Taxis: 10-15%
    - Bellhop: $1-2 per bag
6.2 Non-Reimbursable Items:
    - Personal grooming/toiletries.
    - Fines (parking, speeding).
    - Airline club memberships.
    - In-room movies.
    - Lost luggage/property.

SECTION 7: SUBMISSION PROCESS
Expenses must be submitted within 30 days of trip completion. Receipts required for all expenses > $25.
"""

# =============================================================================
# MOCK DATA: EMPLOYEE REQUESTS (Dynamic Inputs)
# =============================================================================
EMPLOYEE_REQUESTS = [
    # Request 1: Compliant
    """Subject: Trip to London
    I booked a flight to London (8.5 hours) in Business Class for the client summit. 
    Hotel is $320/night. Meal expenses were about $90/day. 
    Receipts attached.""",

    # Request 2: Violation (Booking window & First Class)
    """Subject: Urgent NY Trip
    I need to fly to New York tomorrow. Booked First Class because it was the only seat left.
    Hotel is the Ritz at $500/night. 
    Also expensed $40 for in-flight Wi-Fi to finish the Q3 report.""",

    # Request 3: Violation (Car Rental & Alcohol)
    """Subject: Austin Conference
    Rented a luxury SUV for the team in Austin. 
    Dinner with the team (no clients) came to $800 ($200/person) including 3 bottles of wine.
    Hotel was $240/night.""",

    # Request 4: Compliant (Tier 2 City)
    """Subject: Berlin Site Visit
    Flew Economy to Berlin. Hotel was $220/night.
    Took a taxi from TXL ($45 + $5 tip).
    Daily meals averaged $70.""",
    
    # Request 5: Violation (Misc items)
    """Subject: Tokyo Tech Symposium
    Trip duration: 4 days. 
    Expensed:
    - Flight (Premium Econ, 11 hours)
    - Hotel ($340/night)
    - Laundry service ($60)
    - Forgotten toothbrush replacement ($15)
    - Parking ticket ($50)
    """
]

# Output Schema
OUTPUT_SCHEMA = [
    {"name": "status", "type": str, "desc": "One of: 'COMPLIANT', 'PARTIAL_VIOLATION', 'MAJOR_VIOLATION'"},
    {"name": "violations", "type": str, "desc": "A list of specific policy violations found, referencing the specific section numbers (e.g., 'Violation of Section 2.2'). If compliant, return 'None'."},
    {"name": "reimbursable_summary", "type": str, "desc": "A concise summary of what should be reimbursed vs rejected based on the policy text."},
    {"name": "flag_for_review", "type": bool, "desc": "True if the request requires manual review by a manager (e.g. for high amounts or ambiguous justifications)."},
]

TASK_DESC = f"""
You are an AI auditor for Global Corp. Your job is to review employee travel expense descriptions against the Corporate Travel Policy.
The full policy text is provided below. 

{CORPORATE_TRAVEL_POLICY}

Analyze the input email and determine if the expenses adhere to the policy.
"""

class TravelRequestDataset(pz.IterDataset):
    """Custom dataset that provides travel requests as text records."""
    def __init__(self, requests: List[str]):
        super().__init__(id="travel_requests", schema=TextFile)
        self.requests = requests

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx: int):
        return {
            "filename": f"request_{idx + 1}.txt",
            "contents": self.requests[idx],
        }

# Model mapping (Same as original)
MODEL_MAPPING = {
    "gpt-4o": Model.GPT_4o,
    "gpt-4o-mini": Model.GPT_4o_MINI,
    "claude-4-0-sonnet": Model.CLAUDE_4_0_SONNET,
    # "claude-3-7-sonnet": Model.CLAUDE_3_7_SONNET, # deprecated model testing
    "claude-3-5-haiku": Model.CLAUDE_3_5_HAIKU,
    "gemini-2.5-flash": Model.GOOGLE_GEMINI_2_5_FLASH,
    # "deepseek-v3": Model.DEEPSEEK_V3, # TODO: test with changes from 265
}

def get_model_from_string(model_str: str) -> Model:
    if model_str.lower() in MODEL_MAPPING:
        return MODEL_MAPPING[model_str.lower()]
    for model in Model:
        if model.value.lower() == model_str.lower():
            return model
    raise ValueError(f"Unknown model: {model_str}")

def print_cache_stats(execution_stats):
    """Print cache-related statistics from execution."""
    print("\n" + "=" * 60)
    print(" CACHE STATISTICS & COST ANALYSIS")
    print("=" * 60)

    # Token counts are now disjoint:
    # - total_input_tokens: regular (non-cached) input tokens
    # - total_cache_read_tokens: tokens read from cache (hits)
    # - total_cache_creation_tokens: tokens written to cache
    regular_input = execution_stats.total_input_tokens
    cache_read = execution_stats.total_cache_read_tokens
    cache_creation = execution_stats.total_cache_creation_tokens
    total_output = execution_stats.total_output_tokens
    total_embedding = execution_stats.total_embedding_input_tokens

    # Logical total = regular + cache read + cache creation
    logical_total_input = regular_input + cache_read + cache_creation

    print(f"{'Metric':<35} | {'Count':<15}")
    print("-" * 55)
    print(f"{'Logical Total Input Tokens':<35} | {logical_total_input:,}")
    print(f"{'  - Regular Input (full rate)':<35} | {regular_input:,}")
    print(f"{'  - Cache Read (discounted)':<35} | {cache_read:,}")
    print(f"{'  - Cache Creation':<35} | {cache_creation:,}")
    print("-" * 55)
    print(f"{'Total Output Tokens':<35} | {total_output:,}")
    if total_embedding > 0:
        print(f"{'Total Embedding Input Tokens':<35} | {total_embedding:,}")
    print("-" * 55)
    print(f"{'Total Execution Cost':<35} | ${execution_stats.total_execution_cost:.6f}")

    # Calculate and display cache hit rate
    # Hit rate = cache_read / (regular_input + cache_read)
    total_cacheable = regular_input + cache_read
    if total_cacheable > 0:
        hit_rate = (cache_read / total_cacheable) * 100
        print(f"\nCache Hit Rate: {hit_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Realistic Demo showcasing prompt caching in Palimpzest")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--num-records", type=int, default=5, help="Number of requests to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--profile", action="store_true", help="Save profiling data")
    
    args = parser.parse_args()
    model = get_model_from_string(args.model)

    # Validate env vars (Simplified for brevity)
    if model.is_openai_model() and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return
    if model.is_anthropic_model() and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    if (model.is_google_ai_studio_model() or model.is_vertex_model()) and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set")
        return

    print("=" * 60)
    print(" PZ CACHING DEMO: CORPORATE AUDIT")
    print("=" * 60)
    print(f"Model: {model.value}")
    print(f"Policy Context Size: ~{len(CORPORATE_TRAVEL_POLICY.split())} words (~{int(len(CORPORATE_TRAVEL_POLICY.split()) * 1.3)} tokens)")
    
    # Repeat the request list if user wants more records than we have mocks
    base_requests = EMPLOYEE_REQUESTS
    requests = []
    while len(requests) < args.num_records:
        requests.extend(base_requests)
    requests = requests[:args.num_records]
    
    print(f"Processing {len(requests)} travel requests...")

    # Build Plan
    dataset = TravelRequestDataset(requests)
    
    # The 'desc' field incorporates the huge CORPORATE_TRAVEL_POLICY string.
    # This ensures the System Prompt is large (>1024 tokens) and identical for all records.
    plan = dataset.sem_map(OUTPUT_SCHEMA, desc=TASK_DESC)

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        verbose=args.verbose,
        execution_strategy="sequential", # Sequential often easier to debug caching behavior initially
        available_models=[model],
    )

    start_time = time.time()
    result = plan.run(config)
    end_time = time.time()

    # Output Results
    print("\n" + "=" * 60)
    print(" AUDIT RESULTS")
    print("=" * 60)
    for i, record in enumerate(result.data_records):
        print(f"\n[Request {i+1}]")
        print(f"Status: {record.status}")
        print(f"Violations: {record.violations}")
        print(f"Summary: {record.reimbursable_summary}")

    print_cache_stats(result.execution_stats)
    print(f"\nWall Clock Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()