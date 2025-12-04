from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="Job Helper API",
    description="Provides /score-resume and /generate-cover-letter endpoints for your job workflow",
    version="0.1.0",
)

# ---------- Pydantic models for requests and responses ----------

class JobInfo(BaseModel):
    company: str
    role_title: str
    location: Optional[str] = None
    job_description: str
    job_url: Optional[str] = None


class ResumeInfo(BaseModel):
    name: str
    text: str


class ExtrasInfo(BaseModel):
    tone: Optional[str] = "professional but conversational"
    length: Optional[str] = "3 paragraphs"
    focus_points: Optional[List[str]] = None


class ScoreResumeRequest(BaseModel):
    job: JobInfo
    resume: ResumeInfo


class ScoreResumeResponse(BaseModel):
    match_score: int
    top_keywords: List[str]
    missing_or_weak_keywords: List[str]
    suggested_resume_edits: str


class GenerateCLRequest(BaseModel):
    job: JobInfo
    resume: ResumeInfo
    extras: Optional[ExtrasInfo] = None


class GenerateCLResponse(BaseModel):
    cover_letter_text: str
    summary: Optional[str] = None


# ---------- Helper function to call OpenAI and safely parse JSON ----------

def call_openai_for_json(prompt: str) -> dict:
    """
    Calls OpenAI with a prompt that asks for JSON and returns a parsed dict.
    Raises HTTPException if parsing fails.
    """
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        # Extract text output
        try:
            text_output = response.output[0].content[0].text
        except Exception:
            text_output = str(response)

        # Attempt to parse JSON
        data = json.loads(text_output)
        return data

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="OpenAI did not return valid JSON. Check the prompt or response."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while calling OpenAI: {str(e)}"
        )


# ---------- Endpoint: /score-resume ----------

@app.post("/score-resume", response_model=ScoreResumeResponse)
def score_resume(payload: ScoreResumeRequest):
    job = payload.job
    resume = payload.resume

    prompt = f"""
You are acting as an ATS style resume evaluator.

Job description:
{job.job_description}

Candidate resume:
{resume.text}

Tasks:
1. Give an overall match score from 0 to 100, where 100 means the resume is an excellent match.
2. List the 10 to 15 most important skills, keywords, or phrases from the job description that a hiring manager would expect in a strong resume for this role.
3. Identify which of those keywords or skills are missing or only weakly represented in the resume.
4. Suggest specific edits to the resume that would improve the match, including examples of bullet point changes or additions.

Respond in valid JSON only, with this structure:

{{
  "match_score": number,
  "top_keywords": [ "keyword1", "keyword2", ... ],
  "missing_or_weak_keywords": [ "kwA", "kwB", ... ],
  "suggested_resume_edits": "string with concrete suggestions"
}}
"""

    data = call_openai_for_json(prompt)

    required_keys = [
        "match_score",
        "top_keywords",
        "missing_or_weak_keywords",
        "suggested_resume_edits",
    ]
    for key in required_keys:
        if key not in data:
            raise HTTPException(
                status_code=500,
                detail=f"Missing key in OpenAI response: {key}"
            )

    match_score = int(round(float(data["match_score"])))

    return ScoreResumeResponse(
        match_score=match_score,
        top_keywords=data.get("top_keywords", []),
        missing_or_weak_keywords=data.get("missing_or_weak_keywords", []),
        suggested_resume_edits=data.get("suggested_resume_edits", ""),
    )


# ---------- Endpoint: /generate-cover-letter ----------

@app.post("/generate-cover-letter", response_model=GenerateCLResponse)
def generate_cover_letter(payload: GenerateCLRequest):
    job = payload.job
    resume = payload.resume
    extras = payload.extras or ExtrasInfo()

    focus_points_text = ""
    if extras.focus_points:
        focus_points_text = "Focus on these points: " + ", ".join(extras.focus_points)

    prompt = f"""
You are a job application assistant that writes tailored cover letters.

Job description:
{job.job_description}

Company: {job.company}
Role title: {job.role_title}
Location: {job.location or "N/A"}

Candidate resume:
{resume.text}

Write a cover letter with the following guidelines:
- Tone: {extras.tone}
- Length: {extras.length}
- {focus_points_text}

The cover letter should highlight the candidate's most relevant experience and skills for this specific role and company. It should sound natural and professional.

Respond in valid JSON only, with this structure:

{{
  "cover_letter_text": "full cover letter text here",
  "summary": "one sentence summary of why this candidate is a strong fit"
}}
"""

    data = call_openai_for_json(prompt)

    if "cover_letter_text" not in data:
        raise HTTPException(
            status_code=500,
            detail="Missing 'cover_letter_text' in OpenAI response"
        )

    return GenerateCLResponse(
        cover_letter_text=data.get("cover_letter_text", ""),
        summary=data.get("summary"),
    )


# ---------- Root endpoint for quick health check ----------

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Job Helper API is running"}