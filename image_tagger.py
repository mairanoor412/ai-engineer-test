"""
Task 2.2 — AI Image Description & Auto-Tagging System
Processes a folder of advertising images using a vision-capable LLM.
Generates: alt text, content tags, brand safety score (1-10), and use cases.

Uses Groq's LLaMA 3.2 Vision model for image analysis.
Run: python image_tagger.py
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq's latest vision-capable model
MAX_RETRIES = 3
RETRY_DELAY = 2
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
IMAGE_FOLDER = "test_images"
OUTPUT_FILE = "tags_output.json"


def encode_image_to_base64(image_path: str) -> str:
    """Reads an image file and returns its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(extension: str) -> str:
    """Returns the MIME type for a given image file extension."""
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(extension.lower(), "image/jpeg")


SYSTEM_PROMPT = """You are an expert advertising image analyst and brand safety reviewer.

Analyze the provided image and return a JSON object with these exact fields:

{
  "alt_text": "A concise, descriptive alt text for accessibility (1-2 sentences)",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "brand_safety_score": 8,
  "use_cases": ["use case 1", "use case 2", "use case 3"]
}

RULES:
- alt_text: Describe what is visually in the image — objects, people, colors, setting, mood
- tags: 5-10 relevant content tags for categorization (e.g. "outdoor", "lifestyle", "luxury")
- brand_safety_score: Rate from 1-10 where:
  1-3 = Unsafe (violence, explicit content, controversial)
  4-6 = Moderate risk (alcohol, mild controversy, sensitive topics)
  7-9 = Safe (professional, family-friendly, positive)
  10 = Perfect (universally safe, no risk whatsoever)
- use_cases: 2-4 specific advertising campaign types this image would be good for
- Return ONLY valid JSON, no extra text"""


def analyze_image(image_path: str) -> dict:
    """
    Sends an image to the vision LLM and returns structured analysis.
    Handles base64 encoding and retry logic.
    """
    ext = Path(image_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        return {
            "filename": Path(image_path).name,
            "error": f"Unsupported format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}",
        }

    base64_image = encode_image_to_base64(image_path)
    mime_type = get_mime_type(ext)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.3,  # Low temperature for consistent, accurate analysis
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this advertising image and return structured JSON with alt_text, tags, brand_safety_score, and use_cases.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            },
                        ],
                    },
                ],
            )
            raw = response.choices[0].message.content

            # Try to extract JSON from response (handle markdown code blocks)
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            result = json.loads(raw)

            # Validate required fields
            required = {"alt_text", "tags", "brand_safety_score", "use_cases"}
            missing = required - set(result.keys())
            if missing:
                raise ValueError(f"Missing fields: {missing}")

            return result

        except (RateLimitError, APIConnectionError) as e:
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"    [Retry {attempt}/{MAX_RETRIES}] Waiting {wait_time}s...")
            time.sleep(wait_time)
            if attempt == MAX_RETRIES:
                return {"error": f"API failed after {MAX_RETRIES} retries: {str(e)}"}

        except APIError as e:
            if e.status_code and e.status_code >= 500:
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"    [Retry {attempt}/{MAX_RETRIES}] Server error. Waiting {wait_time}s...")
                time.sleep(wait_time)
                if attempt == MAX_RETRIES:
                    return {"error": f"Server error after {MAX_RETRIES} retries: {str(e)}"}
            else:
                return {"error": f"API error: {str(e)}"}

        except (json.JSONDecodeError, ValueError) as e:
            if attempt == MAX_RETRIES:
                return {"error": f"Invalid response format: {str(e)}"}
            time.sleep(RETRY_DELAY)


def process_folder(folder_path: str) -> list:
    """
    Batch processes all supported images in a folder.
    Returns list of analysis results with filenames.
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' not found.")
        sys.exit(1)

    # Get all supported image files
    images = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not images:
        print(f"Error: No supported images found in '{folder_path}'.")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("AI IMAGE AUTO-TAGGER")
    print(f"{'='*60}")
    print(f"\nFolder: {folder_path}")
    print(f"Images found: {len(images)}")
    print(f"Model: {MODEL}")

    results = []

    for i, image_path in enumerate(images, 1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(images)}] Processing: {image_path.name}")
        print(f"{'─'*60}")

        analysis = analyze_image(str(image_path))
        analysis["filename"] = image_path.name

        # Display results
        if "error" in analysis:
            print(f"  ERROR: {analysis['error']}")
        else:
            print(f"  Alt Text : {analysis['alt_text']}")
            print(f"  Tags     : {', '.join(analysis['tags'])}")
            print(f"  Safety   : {analysis['brand_safety_score']}/10")
            print(f"  Use Cases: {', '.join(analysis['use_cases'])}")

        results.append(analysis)

        # Small delay between API calls to avoid rate limiting
        if i < len(images):
            time.sleep(1)

    return results


if __name__ == "__main__":
    # Process all images in the test_images folder
    results = process_folder(IMAGE_FOLDER)

    # Save output as JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(results)} images.")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")
