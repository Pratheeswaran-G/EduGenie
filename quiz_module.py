"""Quiz generation with Gemini. Returns a validated list of 3 MCQs."""
import json
import re

from gemini_client import GeminiError, generate_text


def clean_json_block(text: str) -> str:
    """Remove Markdown ```json code fences if the model added them."""
    return re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", text, flags=re.DOTALL).strip()


def _normalize(items) -> list:
    """Validate the structure and make sure 'answer' is the exact text of one option."""
    if isinstance(items, dict):
        items = items.get("questions") or items.get("quiz") or [items]
    if not isinstance(items, list) or not items:
        raise ValueError("Quiz JSON must be a non-empty list of questions.")

    quiz = []
    for item in items:
        question = str(item.get("question", "")).strip()
        options = [str(o).strip() for o in item.get("options", [])]
        answer = str(item.get("answer", "")).strip()
        if not question or len(options) < 2:
            raise ValueError("A question is missing its text or options.")

        if answer not in options:
            lowered = [o.lower() for o in options]
            if answer.lower() in lowered:                      # case mismatch
                answer = options[lowered.index(answer.lower())]
            elif len(answer) == 1 and answer.upper() in "ABCD"[: len(options)]:  # "A".."D"
                answer = options[ord(answer.upper()) - 65]
            else:
                raise ValueError(f"Answer '{answer}' does not match any option.")
        quiz.append({"question": question, "options": options, "answer": answer})
    return quiz[:3]


def generate_quiz(text: str) -> list:
    prompt = f"""You are a quiz generator.

From the following passage or topic, create 3 multiple-choice questions. Each question must include:
- "question": the question text
- "options": a list of exactly 4 option strings (do NOT prefix them with A/B/C/D)
- "answer": the correct answer, which must exactly match one of the options

Return ONLY valid JSON, like this:
[
  {{"question": "What is ...?", "options": ["opt1", "opt2", "opt3", "opt4"], "answer": "opt1"}}
]

Passage or topic:
{text}
"""
    raw = generate_text(prompt, json_mode=True)
    try:
        return _normalize(json.loads(clean_json_block(raw)))
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as exc:
        raise GeminiError(f"Could not parse the quiz from the model reply ({exc}). Please try again.") from exc
