"""A small sticker library the bot is allowed to send.

A sticker file_id is the bot's own, so it can be reused forever without hosting
anything, but it has to come from a sticker the bot has actually seen. There is no
public source for those ids and guessing them is the kind of thing that looks
right until the first send fails, so the owner registers them: send a sticker to
the bot and name it, and it is kept under that name.

The library is a plain JSON file rather than a table, for the same reason it is
plain tags rather than an enum: it is a handful of rows that the owner edits by
hand, it is read on every reply, and a corrupt file must not be able to take the
whole database down with it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/stickers.json")

# Which words in an answer make which sticker fit. Matching on the answer's own
# words keeps this out of the model's hands: the bot is told nothing about stickers
# and cannot waste a send, because the only thing that can trigger one is a word
# that is already in the reply.
MOODS: dict[str, tuple[str, ...]] = {
    "ضحك": ("هه", "ههه", "مضحك", "طريف", "ضحكني", "😂", "🤣"),
    "شكر": ("أحسنت", "احسنت", "برافو", "إجابك صح", "إجوابك صح", "ممتاز",
            "بدك", "✔️", "✅"),
    "حزن": ("آسف", "اسف", "ما أعرف", "ما اعرف", "لا أعرف", "😭", "😔"),
    "غضب": ("وحش", "قوي", "خطير", "🔥", "هجوم", "مرّر"),
    "عيون": ("👀", "انتبه", "ترقب", "شوف على"),
}

# At most one sticker per reply, and never two in a row: the point is a small
# reaction, and a bot that answers every message with a sticker is noise.
MIN_GAP_SECONDS = 90.0

_cache: dict | None = None
_cache_path: Path | None = None
_cache_stamp = 0.0
_last_sent: dict[str, float] = {}


def _read(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Sticker library at %s is unreadable (%s); ignoring it", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def library(path: Path | str = DEFAULT_PATH) -> dict:
    """The registered tags, cached briefly so a reply does not hit the disk."""
    global _cache, _cache_path, _cache_stamp
    target = Path(path)
    now = time.time()
    if _cache is not None and _cache_path == target and now - _cache_stamp < 30:
        return _cache
    _cache, _cache_path, _cache_stamp = _read(target), target, now
    return _cache


def remember(tag: str, file_id: str, path: Path | str = DEFAULT_PATH) -> None:
    """Store a sticker the bot has just seen, under the owner's name for it."""
    target = Path(path)
    data = _read(target)
    data[tag] = file_id
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    global _cache, _cache_path, _cache_stamp
    _cache, _cache_path, _cache_stamp = data, target, time.time()


def pick(answer: str, chat_id: int, path: Path | str = DEFAULT_PATH) -> str | None:
    """A file_id worth sending with this answer, or None.

    Mood words are read out of the reply itself. Two guards keep this from
    becoming decoration on every message: the chat has to have been quiet long
    enough, and a mood has to be present in the words at all.
    """
    data = library(path)
    if not data:
        return None
    now = time.time()
    if now - _last_sent.get(str(chat_id), 0.0) < MIN_GAP_SECONDS:
        return None
    for mood, words in MOODS.items():
        file_id = data.get(mood)
        if not file_id:
            continue
        if any(word in answer for word in words):
            _last_sent[str(chat_id)] = now
            return file_id
    return None


def reset_cooldowns() -> None:
    """Forget the last sticker per chat; tests depend on a clean slate."""
    _last_sent.clear()
    global _cache, _cache_path, _cache_stamp
    _cache, _cache_path, _cache_stamp = None, None, 0.0
