# AI Engineer — Advertising Agency Skills Assessment

AI-powered tools built for an advertising agency context using Python + Groq (LLaMA 3.3 70B).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-groq-api-key-here
```

## Tasks

### Task 1.1 — AI Copywriting API

Generates 3 variations of advertising copy (headline, tagline, body, CTA) from a product brief.

**File:** `copy_generator.py`

**Features:**
- System + User prompt structure for role-based generation
- Temperature tuning (0.8) for creative but coherent output
- Retry logic with exponential backoff (3 retries)
- Handles rate limits, connection errors, and server errors
- Validated JSON output with structure checking
- Saves output to `copy_output.json`

**Run:**

```bash
python copy_generator.py
```

**Example Output:** See `copy_output.json` after running.

### Task 1.2 — Advanced Prompt Engineering

Rewrites 3 weak prompts using advanced techniques and demonstrates before/after output comparison.

**File:** `prompt_engineering.py`

**Prompts Rewritten:**
1. Social media post for shoe brand (with before/after AI output comparison)
2. Making ad copy more creative
3. Summarizing a campaign brief

**Techniques Used:**
- Role assignment
- Chain-of-thought via technique checklists
- Structured output templates
- Specificity & context injection
- Negative constraints & grounding rules
- Self-reflection output

**Run:**

```bash
python prompt_engineering.py
```

**Output:** See `prompt_engineering_output.json` after running.

### Task 2.1 — AI Campaign Brief Analyzer

FastAPI web API that accepts a campaign brief and returns structured analysis using AI.

**File:** `brief_analyzer.py`

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/analyze-brief` | Analyze plain text brief |
| POST | `/analyze-brief/pdf` | Analyze PDF upload (Bonus) |
| POST | `/analyze-brief/stream` | SSE streaming response (Bonus) |

**Response Fields:** `audience`, `key_messages[]`, `tone`, `channels[]`, `risks[]`

**Run:**

```bash
python brief_analyzer.py
# or
uvicorn brief_analyzer:app --reload
```

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.

**Test with curl:**

```bash
curl -X POST http://127.0.0.1:8000/analyze-brief \
  -H "Content-Type: application/json" \
  -d '{"brief_text": "Launch campaign for EcoBlend sustainable coffee..."}'
```

## Tech Stack

- Python 3.12
- Groq API (OpenAI-compatible) with LLaMA 3.3 70B
- python-dotenv for environment management
- FastAPI + Uvicorn for web API
- PyMuPDF for PDF text extraction
