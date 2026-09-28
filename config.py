from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    admin_id: int
    gemini_model: str = "gemini-3.5-flash-lite"
    # Drawing runs on a different family of models; the text ones cannot.
    gemini_image_model: str = "gemini-3.1-flash-lite-image"
    database_path: str = "data/edu_bot.db"
    max_download_bytes: int = 20 * 1024 * 1024
    max_message_chars: int = 12000

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
            gemini_image_model=(os.getenv("GEMINI_IMAGE_MODEL", "").strip()
                                or "gemini-3.1-flash-lite-image"),
            # DB_URL wins: on Render it points at a remote database, because the
            # container disk there is wiped on every restart. A libsql:// value
            # lands in the same field as a plain local path.
            database_path=(os.getenv("DB_URL") or os.getenv("DATABASE_PATH") or "data/edu_bot.db").strip(),
        )
