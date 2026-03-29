"""
Task 1.2 — Advanced Prompt Engineering Challenge
Rewrites 3 poorly-performing prompts using advanced techniques.
Includes before/after comparison of actual AI outputs for Prompt 1.
"""

import os
import json
import time
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY = 2


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Calls LLM with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        except (RateLimitError, APIConnectionError) as e:
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"  [Retry {attempt}/{MAX_RETRIES}] Waiting {wait_time}s...")
            time.sleep(wait_time)
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {e}")

        except APIError as e:
            if e.status_code and e.status_code >= 500:
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Server error. Waiting {wait_time}s...")
                time.sleep(wait_time)
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Server error after {MAX_RETRIES} retries: {e}")
            else:
                raise


# =============================================================================
# PROMPT 1 — Social Media Post for Shoe Brand
# =============================================================================

PROMPT_1_WEAK = "Write a social media post for our new shoe brand."

PROMPT_1_REWRITTEN = """You are a senior social media strategist at a leading fashion marketing agency.

TASK: Write an Instagram post for the launch of "StrideX" — a premium athletic shoe brand
targeting Gen-Z fitness enthusiasts (ages 18-25).

CONTEXT:
- Brand voice: Bold, energetic, inclusive
- Product: StrideX Aero — lightweight running shoe with recycled ocean plastic sole
- Campaign theme: "Every Step Counts" (sustainability + performance)
- Launch date: Spring 2026

REQUIREMENTS:
1. Hook: Start with an attention-grabbing first line (under 10 words)
2. Body: 2-3 sentences highlighting the product's unique value proposition
3. Include exactly 2 relevant emojis (not more)
4. End with a clear CTA (follow, shop, or engage)
5. Add 5 relevant hashtags at the bottom
6. Tone: Confident and aspirational, not salesy

FORMAT:
[Hook]
[Body - 2-3 sentences]
[CTA]
[Hashtags]"""

PROMPT_1_EXPLANATION = """
TECHNIQUES USED FOR PROMPT 1:

1. **Role Assignment**: Assigned "senior social media strategist at a fashion marketing agency"
   — this primes the LLM to think like a domain expert, producing more professional output.

2. **Specificity & Context**: Added brand name (StrideX), target audience (Gen-Z, 18-25),
   product details (recycled ocean plastic), and campaign theme ("Every Step Counts").
   The weak prompt had zero context, so the AI was guessing everything.

3. **Output Constraints**: Defined exact structure (hook, body, CTA, hashtags), word limits,
   emoji count, and tone. This eliminates vague, generic responses.

4. **Structured Format Template**: Provided a clear output format so the AI knows exactly
   what sections to produce and in what order.

5. **Negative Constraint**: "Confident and aspirational, not salesy" — telling the model
   what NOT to do is just as important as telling it what to do.
"""


# =============================================================================
# PROMPT 2 — Make Ad Copy More Creative
# =============================================================================

PROMPT_2_WEAK = "Make our ad copy more creative."

PROMPT_2_REWRITTEN = """You are a creative director reviewing ad copy for a client presentation.

TASK: Rewrite the following ad copy using advanced creative techniques to make it more
compelling, memorable, and emotionally resonant.

ORIGINAL COPY:
\"\"\"
{original_copy}
\"\"\"

APPLY THESE CREATIVE TECHNIQUES (use at least 3):
- Metaphor or analogy that connects to the audience's daily life
- Sensory language (sight, sound, touch, taste, smell)
- Power words that trigger emotion (discover, unleash, transform, exclusive)
- Rhythmic structure or alliteration for memorability
- Story micro-arc: setup → tension → resolution in 2-3 sentences

CONSTRAINTS:
- Keep the same core message and CTA
- Stay within +/- 20% of the original word count
- Maintain brand-appropriate tone (professional yet exciting)

OUTPUT FORMAT:
1. REWRITTEN COPY: [your improved version]
2. TECHNIQUES USED: [list which 3+ techniques you applied and where]
3. WHY IT'S BETTER: [1-2 sentences explaining the improvement]"""

PROMPT_2_EXPLANATION = """
TECHNIQUES USED FOR PROMPT 2:

1. **Role Assignment**: "Creative director reviewing for a client presentation" — this sets
   a high quality bar, as the LLM now thinks it's preparing work for stakeholder review.

2. **Chain-of-Thought via Technique Checklist**: Instead of vaguely asking for "more creative",
   we provide a specific menu of creative techniques (metaphor, sensory language, power words,
   rhythm, micro-arc). This gives the LLM concrete strategies to apply.

3. **Input Placeholder**: Uses {original_copy} so the prompt is reusable with any ad copy.
   The weak prompt didn't even provide the copy to improve.

4. **Guardrails**: "Keep same core message", "stay within +/- 20% word count" — prevents
   the model from completely rewriting or going off-topic.

5. **Self-Reflection Output**: Asking the model to explain "TECHNIQUES USED" and "WHY IT'S
   BETTER" forces it to reason about its own creative choices, improving output quality.
"""


# =============================================================================
# PROMPT 3 — Summarize Campaign Brief
# =============================================================================

PROMPT_3_WEAK = "Summarize this campaign brief."

PROMPT_3_REWRITTEN = """You are a senior account strategist at an advertising agency. Your job is to distill
campaign briefs into clear, actionable summaries for the creative team.

TASK: Analyze the following campaign brief and produce a structured executive summary.

CAMPAIGN BRIEF:
\"\"\"
{brief_text}
\"\"\"

OUTPUT — provide ALL of the following sections:

1. **OBJECTIVE** (1 sentence): What is the campaign trying to achieve?
2. **TARGET AUDIENCE** (2-3 bullet points): Demographics, psychographics, behaviors
3. **KEY MESSAGE** (1 sentence): The single most important takeaway for the audience
4. **TONE & STYLE** (3-5 adjectives): How should the campaign feel?
5. **CHANNELS** (bulleted list): Recommended media channels with brief rationale
6. **BUDGET IMPLICATION** (1 sentence): Any noted budget constraints or priorities
7. **TIMELINE** (key dates): Critical milestones or deadlines
8. **RED FLAGS** (if any): Risks, ambiguities, or missing information in the brief

RULES:
- Be concise — each section should be 1-3 lines maximum
- Flag anything that is unclear or missing from the brief under RED FLAGS
- Do not add information that is not in the brief — only summarize what is provided
- Use bullet points, not paragraphs"""

PROMPT_3_EXPLANATION = """
TECHNIQUES USED FOR PROMPT 3:

1. **Role Assignment**: "Senior account strategist" — positions the LLM as someone who
   regularly processes briefs and knows what information matters for creative teams.

2. **Structured Output Template**: Instead of a vague "summarize", we define 8 specific
   sections (objective, audience, key message, tone, channels, budget, timeline, red flags).
   This ensures nothing important is missed.

3. **Grounding Constraint**: "Do not add information not in the brief" — prevents the LLM
   from hallucinating details. This is critical for business documents.

4. **Critical Thinking via Red Flags**: Asking the model to identify "risks, ambiguities,
   or missing information" turns a passive summary into an active analysis. This adds
   strategic value beyond simple summarization.

5. **Conciseness Rules**: "1-3 lines maximum per section", "use bullet points not paragraphs"
   — ensures the output is scannable and practical for busy agency teams.
"""


# =============================================================================
# BEFORE/AFTER COMPARISON — Run Prompt 1 (Weak vs Rewritten)
# =============================================================================

def run_before_after_comparison():
    """Runs both weak and rewritten Prompt 1 against the LLM and shows comparison."""

    print("=" * 70)
    print("TASK 1.2 — PROMPT ENGINEERING: BEFORE/AFTER COMPARISON (PROMPT 1)")
    print("=" * 70)

    # --- BEFORE: Weak Prompt ---
    print("\n" + "─" * 70)
    print("  BEFORE — Weak Prompt")
    print("─" * 70)
    print(f"  Prompt: \"{PROMPT_1_WEAK}\"")
    print("\n  Generating output...\n")

    before_output = call_llm(
        system_prompt="You are a helpful assistant.",
        user_prompt=PROMPT_1_WEAK,
        temperature=0.7,
    )
    print(f"  OUTPUT:\n{before_output}")

    # --- AFTER: Rewritten Prompt ---
    print("\n" + "─" * 70)
    print("  AFTER — Rewritten Prompt (Advanced Techniques)")
    print("─" * 70)
    print(f"  Prompt: [See prompt_engineering.py — PROMPT_1_REWRITTEN]")
    print("\n  Generating output...\n")

    after_output = call_llm(
        system_prompt="You are a senior social media strategist at a leading fashion marketing agency.",
        user_prompt=PROMPT_1_REWRITTEN,
        temperature=0.7,
    )
    print(f"  OUTPUT:\n{after_output}")

    return before_output, after_output


def save_deliverables(before_output: str, after_output: str):
    """Saves all prompt engineering deliverables to a JSON file."""

    deliverables = {
        "prompt_1": {
            "weak_version": PROMPT_1_WEAK,
            "rewritten_version": PROMPT_1_REWRITTEN,
            "explanation": PROMPT_1_EXPLANATION.strip(),
            "before_output": before_output,
            "after_output": after_output,
        },
        "prompt_2": {
            "weak_version": PROMPT_2_WEAK,
            "rewritten_version": PROMPT_2_REWRITTEN,
            "explanation": PROMPT_2_EXPLANATION.strip(),
        },
        "prompt_3": {
            "weak_version": PROMPT_3_WEAK,
            "rewritten_version": PROMPT_3_REWRITTEN,
            "explanation": PROMPT_3_EXPLANATION.strip(),
        },
    }

    output_path = "prompt_engineering_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deliverables, f, indent=2, ensure_ascii=False)

    print(f"\nAll deliverables saved to: {output_path}")


if __name__ == "__main__":
    # Print all 3 rewritten prompts with explanations
    prompts = [
        ("PROMPT 1 — Social Media Post", PROMPT_1_WEAK, PROMPT_1_REWRITTEN, PROMPT_1_EXPLANATION),
        ("PROMPT 2 — Creative Ad Copy", PROMPT_2_WEAK, PROMPT_2_REWRITTEN, PROMPT_2_EXPLANATION),
        ("PROMPT 3 — Campaign Brief Summary", PROMPT_3_WEAK, PROMPT_3_REWRITTEN, PROMPT_3_EXPLANATION),
    ]

    for title, weak, rewritten, explanation in prompts:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        print(f"\n  WEAK VERSION:\n  \"{weak}\"")
        print(f"\n  REWRITTEN VERSION:\n{rewritten}")
        print(f"\n  EXPLANATION:{explanation}")

    # Run before/after comparison for Prompt 1
    print("\n\n")
    before, after = run_before_after_comparison()

    # Save everything
    save_deliverables(before, after)
