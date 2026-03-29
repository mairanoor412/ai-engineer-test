"""
Task 2.1 — AI Campaign Brief Analyzer
FastAPI web API that accepts a campaign brief (text or PDF) and returns
structured analysis: audience, key messages, tone, channels, and risk flags.

Run: uvicorn brief_analyzer:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
import json
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

# --- App Setup ---
app = FastAPI(
    title="AI Campaign Brief Analyzer",
    description="Analyzes advertising campaign briefs using AI and returns structured insights.",
    version="1.0.0",
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY = 2


# --- Pydantic Models ---
class BriefRequest(BaseModel):
    brief_text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "brief_text": "Launch campaign for EcoBlend, a new sustainable coffee brand targeting environmentally-conscious millennials aged 25-35. Budget: $50,000. Timeline: 3 months starting June 2026. The brand uses 100% compostable packaging and sources beans from fair-trade farms in Colombia. Key differentiator: carbon-neutral delivery. Competitors include Blue Bottle and Stumptown. We want to drive online DTC sales through social media and influencer partnerships. Tone should be warm, authentic, and educational — not preachy. Risk: audience fatigue with greenwashing claims."
                }
            ]
        }
    }


class BriefAnalysis(BaseModel):
    audience: str
    key_messages: list[str]
    tone: str
    channels: list[str]
    risks: list[str]


# --- LLM Call with Retry ---
def call_llm(system_prompt: str, user_prompt: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.4,  # Lower temperature for analytical accuracy
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        except (RateLimitError, APIConnectionError) as e:
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            time.sleep(wait_time)
            if attempt == MAX_RETRIES:
                raise HTTPException(status_code=503, detail=f"LLM service unavailable after {MAX_RETRIES} retries: {str(e)}")

        except APIError as e:
            if e.status_code and e.status_code >= 500:
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                time.sleep(wait_time)
                if attempt == MAX_RETRIES:
                    raise HTTPException(status_code=503, detail=f"LLM server error: {str(e)}")
            else:
                raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")


SYSTEM_PROMPT = """You are a senior advertising strategist. Your job is to analyze campaign briefs
and extract structured insights for the creative and media teams.

Analyze the provided campaign brief and return a JSON object with these fields:

{
  "audience": "Detailed target audience description — demographics, psychographics, behaviors",
  "key_messages": ["Message 1", "Message 2", "Message 3"],
  "tone": "Recommended tone of voice for the campaign (3-5 descriptive words)",
  "channels": ["Channel 1 — brief rationale", "Channel 2 — brief rationale"],
  "risks": ["Risk 1", "Risk 2"]
}

RULES:
- Extract information ONLY from the provided brief
- key_messages should have 2-5 items — the core messages the campaign should communicate
- channels should include specific platforms/media with a brief reason for each
- risks should flag any potential issues, missing info, or strategic concerns
- Be concise and actionable — this will be used by a creative team
- Return ONLY valid JSON"""


# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "AI Campaign Brief Analyzer API", "docs": "/docs"}


@app.post("/analyze-brief", response_model=BriefAnalysis)
async def analyze_brief(request: BriefRequest):
    """
    Accepts a campaign brief as text and returns structured AI analysis.
    Returns: audience, key_messages[], tone, channels[], risks[]
    """
    if not request.brief_text.strip():
        raise HTTPException(status_code=400, detail="brief_text cannot be empty")

    user_prompt = f"""Analyze the following campaign brief and provide structured insights:

CAMPAIGN BRIEF:
{request.brief_text}"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON")

    # Validate required fields
    required = {"audience", "key_messages", "tone", "channels", "risks"}
    missing = required - set(result.keys())
    if missing:
        raise HTTPException(status_code=500, detail=f"LLM response missing fields: {missing}")

    return result


# --- BONUS: PDF Upload Support ---

@app.post("/analyze-brief/pdf", response_model=BriefAnalysis)
async def analyze_brief_pdf(file: UploadFile = File(...)):
    """
    BONUS: Accepts a PDF file upload and returns structured AI analysis.
    Extracts text from PDF using PyMuPDF, then analyzes it.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(status_code=500, detail="PyMuPDF not installed. Run: pip install PyMuPDF")

    # Read and extract text from PDF
    content = await file.read()
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF contains no extractable text")

    user_prompt = f"""Analyze the following campaign brief and provide structured insights:

CAMPAIGN BRIEF:
{text}"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON")

    required = {"audience", "key_messages", "tone", "channels", "risks"}
    missing = required - set(result.keys())
    if missing:
        raise HTTPException(status_code=500, detail=f"LLM response missing fields: {missing}")

    return result


# --- BONUS: SSE Streaming Endpoint ---

@app.post("/analyze-brief/stream")
async def analyze_brief_stream(request: BriefRequest):
    """
    BONUS: Streams the analysis response using Server-Sent Events (SSE).
    """
    if not request.brief_text.strip():
        raise HTTPException(status_code=400, detail="brief_text cannot be empty")

    user_prompt = f"""Analyze the following campaign brief and provide structured insights:

CAMPAIGN BRIEF:
{request.brief_text}"""

    def event_stream():
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    data = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': data})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("brief_analyzer:app", host="127.0.0.1", port=8000, reload=True)
