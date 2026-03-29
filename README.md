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

**Docker:**

```bash
docker build -t brief-analyzer .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-groq-key brief-analyzer
```

**Test with curl:**

```bash
curl -X POST http://127.0.0.1:8000/analyze-brief \
  -H "Content-Type: application/json" \
  -d '{"brief_text": "Launch campaign for EcoBlend sustainable coffee..."}'
```

### Task 2.2 — AI Image Auto-Tagger

Batch processes advertising images using a vision LLM to generate alt text, tags, brand safety scores, and use cases.

**File:** `image_tagger.py`

**Features:**
- Batch processes all images in `test_images/` folder — no manual input per image
- Base64 encoding for vision API calls
- Handles unsupported formats gracefully
- Outputs structured JSON to `tags_output.json`

**Each image entry includes:** `filename`, `alt_text`, `tags[]`, `brand_safety_score` (1-10), `use_cases[]`

**Run:**

```bash
python image_tagger.py
```

**Output:** See `tags_output.json` after running.

### Task 2.3 — RAG Campaign Knowledge Bot

Chatbot that answers questions ONLY from provided agency documents (case studies + brand guidelines). Uses RAG (Retrieval-Augmented Generation) with source attribution.

**File:** `rag_chatbot.py`

**Documents (in `documents/` folder):**
- `case_study_nike.txt` — Nike "Just Do It" 35th Anniversary Campaign
- `case_study_cocacola.txt` — Coca-Cola "Share a Coke" Digital Revival
- `brand_guidelines.txt` — Apex Digital Agency Brand Guidelines

**Features:**
- Retrieves answers only from provided documents (no general knowledge)
- Source document citation + relevant quote in every answer
- Refuses to answer questions outside the documents
- Streamlit web UI with chat interface
- CLI fallback mode if Streamlit not available
- FAISS vector store for fast similarity search
- HuggingFace embeddings (free, no API key needed)

**Run:**

```bash
streamlit run rag_chatbot.py
```

Or CLI mode:

```bash
python rag_chatbot.py
```

## Section 3 — Speed & Practical Tasks

All files in `section3/` folder.

### Q1 — Anthropic API Retry Function
Python function with exponential backoff retry (up to 3 times) on rate limit errors.

**File:** `section3/q1_anthropic_retry.py`

### Q2 — Debug Broken RAG Pipeline
Fixed 3 bugs in a LangChain RAG pipeline with detailed explanations.

**File:** `section3/q2_debug_rag_pipeline.py`

**Bugs Found:**
1. Wrong file path in TextLoader
2. chunk_overlap > chunk_size (ValueError)
3. Document objects passed to LLM instead of `.page_content` text

### Q3 — Brand Tone Enforcer System Prompt
System prompt that rewrites off-brand copy to match Apex Digital Agency guidelines.

**File:** `section3/q3_brand_tone_enforcer.py`

**Run:** `python section3/q3_brand_tone_enforcer.py`

### Q4 — Brand Safety Evaluation
Evaluates 3 AI-generated images for brand safety with scoring and written explanations.

**File:** `section3/q4_brand_safety_evaluation.py`

**Run:** `python section3/q4_brand_safety_evaluation.py`

### Q5 — System Architecture Diagram
Text-based architecture for an AI-powered ad personalization engine for e-commerce.

**File:** `section3/q5_architecture_diagram.py`

**Run:** `python section3/q5_architecture_diagram.py`

## Tech Stack

- Python 3.12
- Groq API (OpenAI-compatible) with LLaMA 3.3 70B
- python-dotenv for environment management
- FastAPI + Uvicorn for web API
- PyMuPDF for PDF text extraction
- LLaMA 3.2 90B Vision (Groq) for image analysis
- LangChain + FAISS for RAG retrieval
- HuggingFace sentence-transformers for embeddings
- Streamlit for chatbot web UI
