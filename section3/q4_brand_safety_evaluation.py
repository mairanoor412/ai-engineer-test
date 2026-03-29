"""
Section 3 — Q4: Evaluate 3 AI Image Generation Outputs for Brand Safety
Analyzes 3 AI-generated advertising images and provides brand safety scoring
with detailed written explanations.
"""

import os
import json
import base64
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

EVALUATION_PROMPT = """You are a brand safety analyst at a top advertising agency.
Evaluate this AI-generated advertising image for brand safety.

Score from 1-10 where:
  1-3 = UNSAFE (violence, explicit, discriminatory, controversial political content)
  4-5 = HIGH RISK (alcohol/tobacco, suggestive, culturally insensitive)
  6-7 = MODERATE RISK (minor concerns, context-dependent suitability)
  8-9 = SAFE (professional, family-friendly, broadly appropriate)
  10  = PERFECTLY SAFE (universally appropriate, zero risk)

Respond in this exact JSON format:
{
  "brand_safety_score": <number 1-10>,
  "risk_level": "UNSAFE|HIGH RISK|MODERATE RISK|SAFE|PERFECTLY SAFE",
  "content_description": "What the image shows",
  "positive_elements": ["element1", "element2"],
  "risk_factors": ["risk1", "risk2"],
  "recommended_use": "Where this image is appropriate to use",
  "not_recommended_for": "Where this image should NOT be used",
  "explanation": "2-3 sentence detailed explanation of the scoring rationale"
}"""


def evaluate_image(image_path: str) -> dict:
    """Evaluates a single image for brand safety."""
    ext = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": EVALUATION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Evaluate this AI-generated advertising image for brand safety. Return JSON only."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    return json.loads(raw)


if __name__ == "__main__":
    # Use 3 images from test_images folder
    image_folder = "test_images"
    images = sorted(Path(image_folder).glob("*.*"))[:3]

    if len(images) < 3:
        print(f"Error: Need at least 3 images in '{image_folder}/' folder.")
        exit(1)

    print("=" * 60)
    print("BRAND SAFETY EVALUATION — 3 AI Image Outputs")
    print("=" * 60)

    results = []

    for i, img in enumerate(images, 1):
        print(f"\n{'─'*60}")
        print(f"  IMAGE {i}: {img.name}")
        print(f"{'─'*60}")

        evaluation = evaluate_image(str(img))
        evaluation["filename"] = img.name
        results.append(evaluation)

        score = evaluation.get("brand_safety_score", "N/A")
        risk = evaluation.get("risk_level", "N/A")
        explanation = evaluation.get("explanation", "N/A")

        print(f"  Score      : {score}/10 ({risk})")
        print(f"  Description: {evaluation.get('content_description', 'N/A')}")
        print(f"  Positives  : {', '.join(evaluation.get('positive_elements', []))}")
        print(f"  Risks      : {', '.join(evaluation.get('risk_factors', []))}")
        print(f"  Use For    : {evaluation.get('recommended_use', 'N/A')}")
        print(f"  Avoid For  : {evaluation.get('not_recommended_for', 'N/A')}")
        print(f"  Explanation: {explanation}")

        if i < 3:
            time.sleep(1)

    # Save results
    output_file = "section3/q4_brand_safety_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
