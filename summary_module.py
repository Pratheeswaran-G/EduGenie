"""Summarization with Gemini."""
from gemini_client import generate_text


def summarize_text(text: str) -> str:
    prompt = (
        "Summarize the following text in simple language for a student. "
        "Keep every key point, remove redundancy, and keep it short.\n\n"
        f"{text}"
    )
    return generate_text(prompt)
