"""
Task 1.1 — AI Copywriting API (OpenAI)
Generates 3 variations of advertising copy from a product brief.
Each variation includes: headline, tagline, body copy, and CTA.

Temperature: 0.8 — chosen because advertising copy needs creativity and variety,
but not so high (e.g. 1.5) that outputs become incoherent. 0.8 gives a good
balance between creative language and structured, on-brand messaging.
"""

import os
import json
import time
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",  # Groq API — OpenAI-compatible endpoint
)

# --- Configuration ---
MODEL = "llama-3.3-70b-versatile"  # Groq's fast and capable model
TEMPERATURE = 0.8       # Creative but coherent — ideal for ad copy
MAX_RETRIES = 3          # Retry up to 3 times on rate limit / transient errors
RETRY_DELAY = 2          # Base delay in seconds (doubles each retry)


def call_openai_with_retry(system_prompt: str, user_prompt: str) -> str:
    """
    Calls OpenAI API with exponential backoff retry logic.
    Handles rate limits, API errors, and connection issues gracefully.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},  # Ensures valid JSON output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"  [Retry {attempt}/{MAX_RETRIES}] Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Rate limit exceeded after {MAX_RETRIES} retries: {e}")

        except APIConnectionError as e:
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"  [Retry {attempt}/{MAX_RETRIES}] Connection error. Waiting {wait_time}s...")
            time.sleep(wait_time)
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Connection failed after {MAX_RETRIES} retries: {e}")

        except APIError as e:
            if e.status_code and e.status_code >= 500:
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Server error ({e.status_code}). Waiting {wait_time}s...")
                time.sleep(wait_time)
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Server error after {MAX_RETRIES} retries: {e}")
            else:
                raise  # Client errors (4xx) should not be retried


SYSTEM_PROMPT = """You are an elite advertising copywriter at a top-tier creative agency.
Your job is to generate compelling, on-brand advertising copy based on product briefs.

RULES:
- Generate exactly 3 distinct creative variations
- Each variation must have a completely different creative angle/approach
- Keep headlines punchy (under 10 words)
- Taglines should be memorable and concise (under 15 words)
- Body copy should be 2-3 sentences, persuasive and targeted to the audience
- CTAs should be action-oriented and create urgency
- Match the tone to the target audience and product positioning

OUTPUT FORMAT — respond with ONLY valid JSON in this exact structure:
{
  "variation_1": {
    "headline": "...",
    "tagline": "...",
    "body": "...",
    "cta": "..."
  },
  "variation_2": {
    "headline": "...",
    "tagline": "...",
    "body": "...",
    "cta": "..."
  },
  "variation_3": {
    "headline": "...",
    "tagline": "...",
    "body": "...",
    "cta": "..."
  }
}"""


def generate_ad_copy(brief: str) -> dict:
    """
    Takes a product brief string and returns 3 variations of ad copy.
    Returns parsed JSON dict with variation_1, variation_2, variation_3.
    """
    print(f"\n{'='*60}")
    print("AI COPYWRITING GENERATOR")
    print(f"{'='*60}")
    print(f"\nProduct Brief: {brief}")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"\nGenerating 3 ad copy variations...")

    user_prompt = f"""Generate 3 creative advertising copy variations for the following product brief:

PRODUCT BRIEF:
{brief}

Remember: each variation must take a completely different creative angle. Return ONLY valid JSON."""

    raw_response = call_openai_with_retry(SYSTEM_PROMPT, user_prompt)

    # Parse and validate JSON
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"API returned invalid JSON: {e}\nRaw response: {raw_response}")

    # Validate structure
    required_keys = {"headline", "tagline", "body", "cta"}
    for var_key in ["variation_1", "variation_2", "variation_3"]:
        if var_key not in result:
            raise ValueError(f"Missing key '{var_key}' in response")
        missing = required_keys - set(result[var_key].keys())
        if missing:
            raise ValueError(f"Missing fields in {var_key}: {missing}")

    return result


def display_results(result: dict):
    """Pretty-prints the generated ad copy variations."""
    for i in range(1, 4):
        var = result[f"variation_{i}"]
        print(f"\n{'─'*60}")
        print(f"  VARIATION {i}")
        print(f"{'─'*60}")
        print(f"  Headline : {var['headline']}")
        print(f"  Tagline  : {var['tagline']}")
        print(f"  Body     : {var['body']}")
        print(f"  CTA      : {var['cta']}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Run with the required example brief from the assessment
    example_brief = (
        "New luxury perfume for men, brand name: Noir, "
        "target: 30-45 year old professionals"
    )

    result = generate_ad_copy(example_brief)
    display_results(result)

    # Save output as valid JSON file
    output_path = "copy_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Output saved to: {output_path}")
