
ADVERTISING AGENCY
AI ENGINEER

CANDIDATE SKILLS ASSESSMENT

Duration: 4 Hours	Total Score: 100 pts	Pass Mark: 70 pts

Candidate Name: _________________________	Date: _________________________________
GitHub / Portfolio URL: __________________	Assessor: _____________________________

This test evaluates your ability to design, build, and deploy AI-powered systems within an advertising agency context. You will be assessed on LLM integration, prompt engineering, API development, and real-world AI tool deployment.




SECTION 1 — LLM Integration & Prompt Engineering
Demonstrate your ability to work with large language models via APIs to build real advertising agency tools.

Task 1.1 — AI Copywriting API (OpenAI / Anthropic)
BRIEF
Build a Python script (or Node.js) that calls the OpenAI or Anthropic API to generate advertising copy. The tool must accept a product brief as input and return 3 variations of: headline, tagline, body copy, and CTA.

Deliverables required:
Working script: copy_generator.py or copy_generator.js
Example output: run with the brief 'New luxury perfume for men, brand name: Noir, target: 30-45 year old professionals'
3 structured JSON outputs, each with headline / tagline / body / cta fields

Technical Requirements:
Use system prompt + user prompt structure properly
Implement temperature tuning (explain your choice in comments)
Handle API rate limits and errors gracefully with retry logic
Output must be valid JSON — parseable without cleanup

# Expected output format:
{
  "variation_1": {
    "headline": "...",
    "tagline": "...",
    "body": "...",
    "cta": "..."
  },
  ...
}

Task 1.2 — Advanced Prompt Engineering Challenge
BRIEF
You are given 3 poorly-performing prompts. Your task is to rewrite each one using advanced prompt engineering techniques to dramatically improve output quality. Document your reasoning for each change.

Prompt 1 — Weak Version (rewrite this):
"Write a social media post for our new shoe brand."

Prompt 2 — Weak Version (rewrite this):
"Make our ad copy more creative."

Prompt 3 — Weak Version (rewrite this):
"Summarize this campaign brief."

Deliverables required:
3 rewritten prompts — each in a code block
For each: a 3-5 sentence explanation of techniques used (chain-of-thought, role assignment, output constraints, few-shot, etc.)
Before/after comparison of actual AI outputs for Prompt 1




SECTION 2 — AI System Design & API Development
Build real, functional AI systems that an advertising agency could deploy in production.

Task 2.1 — AI Campaign Brief Analyzer
BRIEF
Build a web API endpoint (FastAPI or Express.js) that accepts a campaign brief (PDF or plain text) and returns a structured analysis: target audience, key messages, tone of voice, suggested channels, and risk flags.

Deliverables required:
POST /analyze-brief endpoint — accepts JSON body with 'brief_text' field
Returns structured JSON with: audience, key_messages[], tone, channels[], risks[]
Demo: run against a provided 500-word campaign brief sample
Dockerfile OR setup instructions to run locally

Technical Stack (choose one):
Python: FastAPI + OpenAI/Anthropic SDK
Node.js: Express + OpenAI/Anthropic SDK

Bonus (extra 5 points):
Add PDF upload support using PyMuPDF or pdf-parse
Stream the response using SSE (Server-Sent Events)

Task 2.2 — AI Image Description & Auto-Tagging System
BRIEF
Build a script that takes a folder of advertising images and uses a vision-capable LLM (GPT-4o Vision or Claude 3) to automatically generate: alt text, content tags, brand safety score (1-10), and suggested campaign use cases.

Deliverables required:
Script: image_tagger.py or image_tagger.js
Processes a folder of 5 provided test images
Outputs a single JSON file: tags_output.json with all results
Each image entry must include: filename, alt_text, tags[], brand_safety_score, use_cases[]

Technical Requirements:
Batch process all images in one run — no manual input per image
Handle image encoding (base64) correctly for vision API calls
Include error handling for unsupported formats

Task 2.3 — RAG-Based Campaign Knowledge Bot
BRIEF
Build a simple RAG (Retrieval-Augmented Generation) chatbot that can answer questions about a provided set of agency case studies and brand guidelines (PDFs provided). The bot must only answer from the provided documents, not from general knowledge.

Deliverables required:
Working chatbot: can be CLI, Streamlit app, or simple web page
Correctly retrieves from 3 provided documents
Responds with: answer + source document name + relevant quote
Refuses to answer questions outside the provided documents

Recommended Stack:
LangChain / LlamaIndex + any embedding model + OpenAI / Anthropic for generation
ChromaDB or FAISS for vector storage




SECTION 3 — Speed & Practical Tasks
Real-world technical tasks evaluated under time pressure.

#	Task	Stack / Tool	Time	Score /10
Q1	Write a Python function that calls the Anthropic API and retries up to 3 times on rate limit errors	Python + Anthropic SDK	15 min	_____ / 10
Q2	Debug a provided broken LangChain RAG pipeline — 3 bugs hidden in the code	LangChain / Python	20 min	_____ / 10
Q3	Write a system prompt for an AI brand tone enforcer — it must rewrite off-brand copy to match provided guidelines	Prompt Engineering	15 min	_____ / 10
Q4	Evaluate 3 AI image generation outputs for brand safety and explain your scoring in writing	GPT-4o Vision / Analysis	10 min	_____ / 10
Q5	Sketch a system architecture diagram for an AI-powered ad personalization engine for a major e-commerce client	System Design	20 min	_____ / 10


EVALUATOR SCORING SHEET

Evaluation Criteria	Max Points	Score
1.1 Copywriting API — Code Quality, JSON Structure, Error Handling	15	______
1.1 Copywriting API — Output Quality of Generated Copy	5	______
1.2 Prompt Engineering — Technique Variety & Explanation Depth	15	______
1.2 Prompt Engineering — Quality Improvement of Rewritten Prompts	5	______
2.1 Campaign Brief Analyzer — API Design & Working Endpoint	10	______
2.1 Campaign Brief Analyzer — Output Accuracy & Structure	5	______
2.2 Image Tagging System — Accuracy, Batch Processing, JSON Output	10	______
2.3 RAG Chatbot — Retrieval Accuracy, Source Attribution, Refusal Logic	15	______
Section 3 — Speed Tasks (5 x 4 points each)	20	______

TOTAL SCORE: _______ / 100	RECOMMENDATION: HIRE / HOLD / REJECT


Evaluator Notes / Comments:

______________________________________________________________________________
______________________________________________________________________________
______________________________________________________________________________
______________________________________________________________________________

