from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _positive_int(name: str, fallback: int) -> int:
    """A byte limit from the environment, ignoring junk rather than crashing."""
    raw = os.getenv(name, "").strip()
    if not raw.isdigit() or int(raw) <= 0:
        return fallback
    return int(raw)


def _non_negative_float(name: str, fallback: float) -> float:
    """A seconds value from the environment; 0 is a valid "answer now"."""
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return fallback
    return value if value >= 0 else fallback


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    admin_id: int
    gemini_model: str = "gemini-3.5-flash-lite"
    # A second, separate key used for one job only: deciding whether a group
    # message is meant for the bot. Empty means that gate is off and the bot
    # answers only when it is named, mentioned or replied to -- which is the
    # behaviour that worked before this existed, so an unset key is not a
    # degraded mode but the original one. It is a separate key rather than a
    # second setting on the first because the gate runs on every message a group
    # receives, including the ones that are ignored; on a shared key that traffic
    # would spend the quota the actual answers need.
    router_api_key: str = ""
    # Deliberately the cheapest model in the chain: the gate reads a little
    # context and answers one word, so a larger model would cost more to be right
    # about less.
    router_model: str = "gemini-3.5-flash-lite"
    database_path: str = "data/edu_bot.db"
    max_download_bytes: int = 20 * 1024 * 1024
    max_message_chars: int = 12000
    # A document or a photo sent as a file. Kept apart from the photo limit
    # because a PDF carries far more meaning per megabyte than a JPEG.
    max_document_bytes: int = 20 * 1024 * 1024
    # Video is capped lower on purpose: a short MP4 already eats a large slice
    # of the daily quota, and a long one can take a whole request by itself.
    max_video_bytes: int = 10 * 1024 * 1024
    # The bot waits before it answers. The default is no wait at all, because a
    # student waiting on a maths answer would rather have it now than watch a
    # typing indicator; raise it in the environment if you want a beat.
    answer_delay_seconds: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        admin_raw = os.getenv("ADMIN_ID", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is required")
        if not admin_raw or not admin_raw.lstrip("-").isdigit():
            raise ValueError("ADMIN_ID must be a numeric Telegram user ID")
        return cls(
            telegram_bot_token=token,
            gemini_api_key=gemini_key,
            admin_id=int(admin_raw),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite",
            # Optional on purpose. Absent or blank and the bot behaves exactly as
            # it did before the gate existed, which is why this is not validated
            # the way TELEGRAM_BOT_TOKEN and GEMINI_API_KEY are: turning the gate
            # on is a decision, not a requirement for the bot to run.
            router_api_key=os.getenv("ROUTER_API_KEY", "").strip(),
            router_model=os.getenv("ROUTER_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite",
            # DB_URL wins: on Render it points at a remote database, because the
            # container disk there is wiped on every restart. A libsql:// value
            # lands in the same field as a plain local path.
            database_path=(os.getenv("DB_URL") or os.getenv("DATABASE_PATH") or "data/edu_bot.db").strip(),
            max_document_bytes=_positive_int("MAX_DOCUMENT_BYTES", 20 * 1024 * 1024),
            max_video_bytes=_positive_int("MAX_VIDEO_BYTES", 10 * 1024 * 1024),
            answer_delay_seconds=_non_negative_float("ANSWER_DELAY_SECONDS", 0.0),
        )
