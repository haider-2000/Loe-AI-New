"""Sticker packs and tagged stickers the bot is allowed to send.

A sticker file_id only works for the bot that fetched it, and a bot cannot invent
one: there is nowhere to copy an id from and nothing to log in as. So the pack has
to be handed to the bot through Telegram itself. `getStickerSet(name)` on any pack
returns real stickers with file_ids that are valid for the calling bot, and those
ids do not expire, so one fetch per pack is enough -- the owner names a pack once
and the bot has the whole thing.

Why the emoji on each sticker is worth storing: it is the only free description of
what a sticker means that already exists. Searching the answer Leo already wrote
for an emoji gives a reaction that is exactly the one he chose, without the model
being asked for anything and without the owner tagging 120 stickers by hand. A
mood word like "هههه" is only the fallback for when he wrote no emoji.

Storage is a plain JSON file rather than a table: a handful of rows, edited by the
owner, read on every reply, and a corrupt file must not be able to take the whole
database down with it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/stickers.json")

# The pack that belongs to this bot. It is loaded on its own at startup rather than
# waiting to be asked for, because the owner made it for exactly this and every
# deploy would otherwise need a reminder to fetch it again.
DEFAULT_PACK = "Leo_AI"

# mood -> (words that mean it, emoji that mean it). Words are how Leo writes when
# he does not put an emoji; emoji are how the pack is searched first.
MOODS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ضحك": (("هه", "ههه", "مضحك", "طريف", "ضحكني"), ("😂", "🤣", "😆", "😹")),
    "شكر": (("أحسنت", "احسنت", "برافو", "إجابك صح", "ممتاز", "بدك"),
            ("👍", "✅", "🤝", "👏")),
    "حزن": (("آسف", "اسف", "ما أعرف", "لا أعرف"), ("😔", "😭", "💔")),
    "غضب": (("وحش", "خطير", "هجوم"), ("🔥", "😤", "👊")),
    "عيون": (("انتبه", "ترقب", "خلك واعي"), ("👀", "🤨")),
}

# At most one sticker per reply, and never two in a row: the point is a small
# reaction, and a bot that answers every message with a sticker is noise.
MIN_GAP_SECONDS = 90.0

_cache: dict | None = None
_cache_path: Path | None = None
_cache_stamp = 0.0
_last_sent: dict[str, float] = {}
_last_pick: dict[str, str] = {}


def _normalise(raw: dict) -> dict:
    """Accept both shapes: the tagged-only file written first, and packs too."""
    if not isinstance(raw, dict):
        return {"tags": {}, "packs": {}}
    if "tags" in raw or "packs" in raw:
        raw.setdefault("tags", {})
        raw.setdefault("packs", {})
        return raw
    # The first version of this file was {"ضحك": "CAADBA_x"}, a flat map of tag to
    # file_id. Read it as tags rather than losing what the owner already taught.
    return {"tags": {k: v for k, v in raw.items() if isinstance(v, str)},
            "packs": {}}


def _read(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"tags": {}, "packs": {}}
    except (OSError, ValueError) as exc:
        logger.warning("Sticker library at %s is unreadable (%s); ignoring it", path, exc)
        return {"tags": {}, "packs": {}}
    return _normalise(raw)


def library(path: Path | str = DEFAULT_PATH) -> dict:
    """The registered tags and packs, cached briefly so a reply skips the disk."""
    global _cache, _cache_path, _cache_stamp
    target = Path(path)
    now = time.time()
    if _cache is not None and _cache_path == target and now - _cache_stamp < 30:
        return _cache
    _cache, _cache_path, _cache_stamp = _read(target), target, now
    return _cache


def _write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    global _cache, _cache_path, _cache_stamp
    _cache, _cache_path, _cache_stamp = data, path, time.time()


def remember(tag: str, file_id: str, path: Path | str = DEFAULT_PATH) -> None:
    """Store one sticker the owner sent by hand, under a mood name."""
    target = Path(path)
    data = _read(target)
    data["tags"][tag] = file_id
    _write(data, target)


def save_pack(name: str, title: str, entries: list[dict],
              path: Path | str = DEFAULT_PATH) -> int:
    """Keep a whole pack: every sticker's file_id and the emoji that describes it."""
    target = Path(path)
    data = _read(target)
    data["packs"][name] = {
        "title": title or name,
        "stickers": [{"file_id": e["file_id"], "emoji": e.get("emoji") or ""}
                     for e in entries if e.get("file_id")],
    }
    _write(data, target)
    return len(data["packs"][name]["stickers"])


def pack_names(path: Path | str = DEFAULT_PATH) -> list[str]:
    return sorted(library(path).get("packs", {}))


def _wanted(answer: str) -> list[str]:
    """Emoji worth sending, in the order they should win.

    An emoji Leo wrote himself comes first: he chose it, so the sticker that wears
    it is the sticker he meant. Only then do the moods widen the search, because a
    mood word is this file's guess rather than Leo's.
    """
    order: list[str] = []
    for char in answer:
        if char not in order and _is_emoji(char):
            order.append(char)
    for words, emoji in MOODS.values():
        if any(word in answer for word in words):
            for face in emoji:
                if face not in order:
                    order.append(face)
    return order


def _is_emoji(char: str) -> bool:
    """Is this one character a face?

    "Above the BMP" is the wrong test and it fails on exactly the stickers that
    matter most here: ❌ is U+274C and ✅ is U+2705, both below it, because they
    were encoded as symbols long before the pictographs filled the higher planes.
    Testing for "above U+1F000" would find 🧠 and miss ❌, which is half a pack.
    So this checks the ranges the pictographs actually live in, and takes the
    variation selector too -- it is what makes a text symbol render as an emoji.
    """
    if not char:
        return False
    code = ord(char[0])
    return (0x1F000 <= code <= 0x1FAFF      # pictographs, faces, gestures
            or 0x2600 <= code <= 0x27BF     # symbols and dingbats: ❌ ✅ ✨
            or 0x2B00 <= code <= 0x2BFF     # ⭐ and the rest of the arrows
            or char == "️")                 # the emoji variation selector


def _candidates(store: dict, answer: str) -> list[dict]:
    """Every sticker that could serve this answer, best first."""
    out: list[dict] = []
    for face in _wanted(answer):
        for pack in store.get("packs", {}).values():
            for entry in pack.get("stickers", []):
                if entry.get("emoji") == face:
                    out.append(entry)
    tags = store.get("tags", {})
    for mood, (words, _faces) in MOODS.items():
        file_id = tags.get(mood)
        if file_id and any(word in answer for word in words):
            out.append({"file_id": file_id, "emoji": ""})
    return out


def pick(answer: str, chat_id: int, path: Path | str = DEFAULT_PATH) -> dict | None:
    """A sticker worth sending with this answer, or None.

    The model is not consulted and never can request one: the only thing that can
    trigger a send is a word or an emoji that is already in Leo's own reply.
    """
    store = library(path)
    now = time.time()
    key = str(chat_id)
    if now - _last_sent.get(key, 0.0) < MIN_GAP_SECONDS:
        return None
    options = _candidates(store, answer)
    if not options:
        return None
    # Never send the same face twice running; the second reply gets the next one.
    previous = _last_pick.get(key)
    chosen = next((e for e in options if e["file_id"] != previous), options[0])
    _last_sent[key] = now
    _last_pick[key] = chosen["file_id"]
    return chosen


def reset_cooldowns() -> None:
    """Forget the last sticker per chat; tests depend on a clean slate."""
    _last_sent.clear()
    _last_pick.clear()
    global _cache, _cache_path, _cache_stamp
    _cache, _cache_path, _cache_stamp = None, None, 0.0


async def ensure_default_pack(bot, path: Path | str = DEFAULT_PATH,
                              name: str = DEFAULT_PACK) -> int:
    """Fetch the bot's own pack once, and leave it alone after that.

    Called at startup instead of on the first reply, because a reply path that
    waits on the Telegram API is a reply path that can be slow, and the pack does
    not change between deploys. A pack that cannot be read is logged and dropped:
    stickers are decoration, and a failure here must not stop the bot starting.
    """
    stored = library(path)
    if name in stored.get("packs", {}):
        return len(stored["packs"][name].get("stickers", []))
    try:
        pack = await bot.get_sticker_set(name)
    except Exception as exc:
        logger.warning("Sticker pack %s is unavailable (%s); continuing without it",
                       name, exc)
        return 0
    entries = [{"file_id": s.file_id, "emoji": s.emoji or ""} for s in pack.stickers]
    kept = save_pack(pack.name, pack.title, entries, path)
    logger.info("Loaded sticker pack %s with %d stickers", pack.title, kept)
    return kept
