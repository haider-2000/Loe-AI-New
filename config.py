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
    allowed_chat_id: int
    gemini_model: str = "gemini-3.7-flash"
    database_path: str = "data/edu_bot.db"
    max_download_bytes: int = 20 * 1024 * 1024
    max_message_chars: int = 12000

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        admin_raw = os.getenv("ADMIN_ID", "").strip()
        allowed_chat_raw = os.getenv("ALLOWED_CHAT_ID", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is required")
        if not admin_raw or not admin_raw.lstrip("-").isdigit():
            raise ValueError("ADMIN_ID must be a numeric Telegram user ID")
        if not allowed_chat_raw or not allowed_chat_raw.lstrip("-").isdigit():
            raise ValueError("ALLOWED_CHAT_ID must be the numeric Telegram group ID")
        return cls(
            telegram_bot_token=token,
            gemini_api_key=gemini_key,
            admin_id=int(admin_raw),
            allowed_chat_id=int(allowed_chat_raw),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip() or "gemini-3.7-flash",
            # DB_URL wins: on Render it points at a remote database, because the
            # container disk there is wiped on every restart. A libsql:// value
            # lands in the same field as a plain local path.
            database_path=(os.getenv("DB_URL") or os.getenv("DATABASE_PATH") or "data/edu_bot.db").strip(),
        )
