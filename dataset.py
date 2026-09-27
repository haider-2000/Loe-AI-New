from __future__ import annotations

import re
from pathlib import Path

from database import add_contribution, export_jsonl

# Deliberately conservative obvious-PII checks; flagged items stay out of export.
PII_PATTERNS = [
    # Iraqi/local numbers, tolerating the spacing and dashes people actually type.
    re.compile(r"(?:\+?964[\s-]*|0097[\s-]*|0)?7[\s-]?\d(?:[\s-]?\d){8}"),
    re.compile(r"\b\d{9,12}\b"),
    re.compile(r"(?:العنوان|عنواني|شارع|محلة|زقاق|دارنا|بيتنا)", re.IGNORECASE),
    re.compile(r"(?:هوي(?:ة|تي)|بطاقة|جواز)|\b(?:passport|phone|tel)\b", re.IGNORECASE),
]


def has_obvious_pii(text: str | None) -> bool:
    return bool(text and any(pattern.search(text) for pattern in PII_PATTERNS))


def dialect_for(text: str | None) -> str:
    if not text:
        return "iraqi"
    iraqi_markers = ("شلون", "شنو", "هاي", "هسه", "أريد", "اريد", "ماكو", "ليش", "يمّه", "وين")
    return "iraqi" if any(marker in text for marker in iraqi_markers) else "ar"


def language_for(text: str | None) -> str:
    return "ar" if text and re.search(r"[\u0600-\u06ff]", text) else "en"


async def save_text(db_path: str, text: str, answer: str, raw_dir: str) -> bool:
    privacy = has_obvious_pii(text) or has_obvious_pii(answer)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    return await add_contribution(
        db_path, contribution_type="text", file_path=None, text_content=text,
        transcription=None, model_answer=answer, language=language_for(text),
        dialect=dialect_for(text), privacy_flag=privacy,
    )


async def save_image(db_path: str, file_path: str, caption: str, answer: str,
                     dedupe_key: str | None = None) -> bool:
    privacy = has_obvious_pii(caption) or has_obvious_pii(answer)
    return await add_contribution(
        db_path, contribution_type="image", file_path=file_path, text_content=caption,
        transcription=None, model_answer=answer, language=language_for(caption),
        dialect=dialect_for(caption), privacy_flag=privacy, dedupe_key=dedupe_key,
    )


async def save_voice_transcription(db_path: str, transcription: str, answer: str) -> bool:
    privacy = has_obvious_pii(transcription) or has_obvious_pii(answer)
    return await add_contribution(
        db_path, contribution_type="text", file_path=None, text_content=transcription,
        transcription=transcription, model_answer=answer, language=language_for(transcription),
        dialect=dialect_for(transcription), privacy_flag=privacy,
    )


async def export_dataset(db_path: str, output_path: str) -> tuple[str, int]:
    return await export_jsonl(db_path, output_path)
