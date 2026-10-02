"""Concept explanation using the local LaMini-Flan-T5-783M model.

The model is loaded lazily on the first /explain call so the server starts instantly.
If it cannot be loaded (and EXPLAIN_FALLBACK_TO_GEMINI=true) Gemini is used instead.
"""
import os
import threading

from dotenv import load_dotenv

load_dotenv()

EXPLAIN_MODEL = os.getenv("EXPLAIN_MODEL", "MBZUAI/LaMini-Flan-T5-783M")
FALLBACK = os.getenv("EXPLAIN_FALLBACK_TO_GEMINI", "true").lower() in ("1", "true", "yes")

_lock = threading.Lock()
_tokenizer = None
_model = None
_load_error = None


def _load_model():
    global _tokenizer, _model, _load_error
    if _model is not None:
        return
    if _load_error is not None:
        raise RuntimeError(_load_error)
    with _lock:
        if _model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            print(f"Loading local explanation model '{EXPLAIN_MODEL}' (first run downloads ~3 GB)...")
            _tokenizer = AutoTokenizer.from_pretrained(EXPLAIN_MODEL)
            _model = AutoModelForSeq2SeqLM.from_pretrained(EXPLAIN_MODEL)
            _model.eval()
            print("Explanation model ready.")
        except Exception as exc:
            _load_error = f"Could not load local model '{EXPLAIN_MODEL}': {exc}"
            raise RuntimeError(_load_error) from exc


def _explain_locally(topic: str) -> str:
    import torch

    _load_model()
    input_text = f"Explain the concept of '{topic}' in a simple and clear way for a school student."
    inputs = _tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def explain_topic(topic: str) -> str:
    try:
        return _explain_locally(topic)
    except Exception as exc:
        if not FALLBACK:
            raise
        print(f"[explain] {exc}\n[explain] Falling back to Gemini.")
        from gemini_client import generate_text

        return generate_text(
            f"Explain the concept of '{topic}' in a simple and clear way for a school student, "
            "in one short paragraph."
        )
