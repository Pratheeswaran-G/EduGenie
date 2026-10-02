"""Shared Gemini helper used by qna, quiz, summary and learning-path modules."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_client = None


class GeminiError(Exception):
    """Raised for any problem talking to Gemini (missing key, API error, empty reply)."""


def _get_client() -> "genai.Client":
    global _client
    if not API_KEY or API_KEY == "your_api_key_here":
        raise GeminiError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey, then restart the server."
        )
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client


def generate_text(prompt: str, json_mode: bool = False) -> str:
    """Send a prompt to Gemini and return the reply text.

    json_mode=True asks Gemini to return raw JSON (no markdown fences).
    """
    client = _get_client()
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
    except Exception as exc:  # network, quota, invalid key, unknown model...
        raise GeminiError(f"Gemini request failed: {exc}") from exc

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise GeminiError("Gemini returned an empty response (it may have been blocked by safety filters).")
    return text
