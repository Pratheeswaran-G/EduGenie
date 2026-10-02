"""EduGenie - FastAPI backend.

Run with:  uvicorn main:app --reload
"""
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from explanation_module import explain_topic
from gemini_client import MODEL_NAME, GeminiError
from learning_path import get_learning_recommendations
from qna import answer_question_with_gemini
from quiz_module import generate_quiz
from summary_module import summarize_text

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="EduGenie", description="Gemini powered learning assistant")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TopicIn(BaseModel):
    topic: str = ""


class TextIn(BaseModel):
    text: str = ""


def error(message: str, status: int = 500) -> JSONResponse:
    return JSONResponse(content={"error": message}, status_code=status)


@app.exception_handler(GeminiError)
async def gemini_error_handler(_: Request, exc: GeminiError):
    return error(str(exc), 502)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "gemini_model": MODEL_NAME}


# Q&A - GET using Gemini
@app.get("/qa")
def qa(question: str = Query(..., min_length=1)):
    if not question.strip():
        return error("Please provide a question.", 400)
    return {"answer": answer_question_with_gemini(question.strip())}


# Explanation - POST using the local LaMini-Flan-T5 model
@app.post("/explain")
def explain(body: TopicIn):
    topic = body.topic.strip()
    if not topic:
        return error("Please provide a topic.", 400)
    try:
        return {"topic": topic, "explanation": explain_topic(topic)}
    except GeminiError:
        raise
    except Exception as exc:
        return error(f"Explanation failed: {exc}", 500)


# Summarization - POST
@app.post("/summarize")
def summarize(body: TextIn):
    text = body.text.strip()
    if not text:
        return error("Please provide text to summarize.", 400)
    return {"summary": summarize_text(text)}


# Quiz generation - POST
@app.post("/quiz")
def quiz(body: TextIn):
    text = body.text.strip()
    if not text:
        return error("Please provide a topic or text for the quiz.", 400)
    return {"quiz": generate_quiz(text)}


# Learning recommendations - GET
@app.get("/learn/recommendations")
def learning_recommendations(topic: str = Query(..., min_length=1)):
    if not topic.strip():
        return error("Please provide a topic.", 400)
    return {"topic": topic.strip(), "recommendation": get_learning_recommendations(topic.strip())}
