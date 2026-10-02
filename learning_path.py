"""Personalized learning path with Gemini."""
from gemini_client import generate_text


def get_learning_recommendations(topic: str) -> str:
    prompt = f"""You are an AI tutor. The student wants to learn about: {topic}.
Suggest a structured and adaptive learning path including key topics, order of learning,
estimated timelines, and resources (videos, articles, books).
Include beginner, intermediate, and advanced levels if needed.
Use markdown with headings, bullet lists and **bold** for emphasis."""
    return generate_text(prompt)
