from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from datetime import datetime
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, User
from telegram.constants import ChatType
from telegram.error import BadRequest, Conflict, Forbidden, RetryAfter, TelegramError
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, TypeHandler, filters)

from config import Settings
from database import (all_rows, count_by_status, create_lesson, create_quiz_session,
                      enroll_student, get_quiz_session, image_names, init_db, is_remote,
                      latest_pending_id, lesson_roster, list_lessons, load_image, pending_rows,
                      quiz_history, quiz_overall, read_flag, record_quiz_answer,
                      set_contribution_status, set_privacy_flag, snapshot_database, store_image,
                      write_flag)
from dataset import export_dataset, save_image, save_text, save_voice_transcription
from gemini_client import GeminiClient, GeminiQuotaError, GeminiUnavailableError
from memory import ConversationMemory
import stickers

BUSY_MESSAGE = "الخدمة مشغولة حالياً لأن ضغط الطلبات عالية. حاول بعد دقيقة."

# Sent when the account has no quota left for a model at all. Telling a student
# to come back in a minute would be a lie here: a limit of 0 does not refill.
QUOTA_MESSAGE = "الخدمة مو متوفرة على الحساب حالياً، فما أگدر أنفّذ الطلب. جرّب بعدين."

# How long Telegram keeps a "typing" indicator alive before the client hides it.
# The wait is refreshed on this boundary so the student never sees it vanish
# while the bot is still working on their question.
TYPING_ACTION_TTL = 4.0

# Files the bot knows how to read, mapped to the extension used if it ever has
# to name one. A GIF is not in here on purpose: Telegram hands a GIF to a bot as
# a document with a video/mp4 mime type, so mp4 is the format that actually
# arrives and there is no image/gif to wait for.
DOCUMENT_MIMES = {
    "application/pdf": ".pdf",
    "video/mp4": ".mp4",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
VIDEO_MIMES = frozenset({"video/mp4"})
UNSUPPORTED_DOCUMENT_MESSAGE = (
    "ما أگدر أقرأ هذا النوع من الملفات. أرسل PDF أو صورة (JPG/PNG) "
    "أو GIF أو مقطع MP4.")

# Sent when someone calls the name without asking anything, which happens when
# they expect a chat to open rather than a one-shot answer.
WAKE_UP_MESSAGE = ("هلا بيك! شنو السؤال؟ دزه نصاً، أو صورة، أو صوت، "
                   "وأنا جاوبك.")

# Private chat is open to anybody now, so the bot asks before it starts talking.
# Telegram already blocks a bot from writing to someone who never spoke to it,
# but that only asks for a tap on Start. This asks for a decision instead, and
# the answer is kept, so the question is not repeated on every message.
CONSENT_ACCEPTED = "تم التأكيد ✅\nالحين أنت مشرف هالمحادثة الخاصة. دز سؤالك وياي."
CONSENT_DECLINED = "تم الإلغاء. وقت ما تريد تسأل، أرسل /start."
CONSENT_NOT_YOURS = "هذا الطلب مو إلك."
CONSENT_YES = "privok"
CONSENT_NO = "privno"
# Stored per user id next to the request text, so the flag name and the prompts
# that depend on it move together.
PRIVATE_CONSENT_FLAG = "pchat"

# Whether the *database* lives somewhere ephemeral, like a Render container.
# The name matters: it says nothing about the image bytes. A local disk already
# keeps the file, so the bytes only need a second home inside the database when
# that disk is wiped on every restart.
database_is_remote = True
# Whether Telegram hands the bot every plain group message. It only does so when
# privacy mode is off, and a message it never receives can never be answered, so
# this is looked up once at startup rather than left to be discovered by students.
reads_all_group_messages: bool | None = None
# Set by start_health_server when the platform gave us a port to listen on.
health_port: int | None = None

# The file handler below needs the directory to exist first, otherwise importing
# this module fails on a fresh clone where logs/ is absent.
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler("logs/edu_bot.log", encoding="utf-8"), logging.StreamHandler()],
)
# httpx logs the full request URL at INFO, which embeds the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

settings: Settings
ai: GeminiClient
# The relevance gate: a second client on a second key, built only when
# ROUTER_API_KEY is set. None means the gate is off, and off is the behaviour
# this bot had for its whole life -- answer when named, ignore the rest -- so a
# missing key costs nothing but the feature.
router: GeminiClient | None = None
# What the bot remembers between messages, so "and the second one?" keeps
# pointing at the same topic instead of starting from nothing.
memory = ConversationMemory()


def is_admin(update: Update) -> bool:
    return bool(update.effective_user and update.effective_user.id == settings.admin_id)


def _consent_flag(user_id: int) -> str:
    return f"{PRIVATE_CONSENT_FLAG}:{user_id}"


async def has_private_consent(user_id: int) -> bool:
    """Whether this person already said yes to a private conversation.

    The answer lives in the database rather than in memory: a restart on Render
    must not make the bot ask a student the same question all over again.
    """
    return await read_flag(settings.database_path, _consent_flag(user_id)) == "1"


async def set_private_consent(user_id: int, accepted: bool) -> None:
    # A refused chat is stored as "0" rather than deleted, so "declined" and
    # "never asked" stay apart, and /revoke needs no second code path.
    await write_flag(settings.database_path, _consent_flag(user_id),
                     "1" if accepted else "0")


async def private_consent_keyboard(user: User) -> InlineKeyboardMarkup:
    """The yes/no pair, with the id inside the callback data."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("تأكيد ✅", callback_data=f"{CONSENT_YES}:{user.id}"),
        InlineKeyboardButton("إلغاء ✖️", callback_data=f"{CONSENT_NO}:{user.id}"),
    ]])


def consent_request_text(user: User) -> str:
    """The request, addressed to the account it came from by name.

    Telegram already says who wrote, so there is no reason to ask an anonymous
    question. Naming the person also gives the owner the id they need for
    /revoke, and it proves to the reader that the request was built for them
    and not forwarded from somebody else's chat.
    """
    full_name = " ".join(part for part in (user.first_name, user.last_name) if part)
    who = full_name or user.username or f"المستخدم {user.id}"
    handle = f"@{user.username}" if user.username else "ماكو username"
    return (
        "طلب تأكيد\n\n"
        f"الاسم: {who}\n"
        f"المعرّف: {user.id}\n"
        f"الحساب: {handle}\n\n"
        "هسة أنت المشرف على هالمحادثة الخاصة بيني وبينك.\n"
        "إذا توافق، أگدر نكمل دردشة وياك، وتگدر ترسل لي نص أو صورة أو صوت أو ملف.\n"
        "إذا ما توافق، اضغط إلغاء وما راح أگدر أفتح لك الدردشة.")


async def ask_private_consent(update: Update) -> None:
    user = update.effective_user
    if not user or not update.effective_message:
        return
    await update.effective_message.reply_text(
        consent_request_text(user), reply_markup=await private_consent_keyboard(user))


def admin_only(handler):
    """Stay silent for anybody but the owner.

    Applied as the *outer* decorator on purpose. The chat gate below asks a
    stranger in private for confirmation before it runs anything, and a command
    this person could never use should not even be worth confirming: /backup has
    to answer silence, because a confirmation prompt would only teach a stranger
    that the command exists. Outermost, this check therefore runs first.
    """
    @wraps(handler)
    async def guarded(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update):
            return
        return await handler(update, context)

    return guarded


def allowed_chat_only(handler):
    """Serve every group the bot is a member of, and every private chat.

    There is no group allowlist any more: any group or supergroup that adds the
    bot gets an answer, so a new class does not need a deployment first. Private
    chats are open to anyone as well, so a student can ask in direct messages
    instead of hunting for the group first -- but only after they confirm they
    want the conversation, which is what ask_private_consent is for. The owner is
    never asked; they own the bot.

    The private commands that touch stored data -- /backup, /export, /approve and
    the rest of the teacher's tools -- are *not* protected by that. They used to
    be, but only by accident: a private chat used to be the owner's by
    definition, so "is this a private chat" was standing in for "is this the
    admin". The moment a stranger can write in private that reasoning is false,
    and /backup would hand the whole database to whoever asked. Each of those
    commands therefore carries admin_only, which runs before this gate.
    """
    @wraps(handler)
    async def guarded(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        if chat and chat.type == ChatType.PRIVATE:
            user = update.effective_user
            if user and not is_admin(update) and not await has_private_consent(user.id):
                await ask_private_consent(update)
                return
            return await handler(update, context)
        if not chat or chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            logger.info("Ignoring update from chat type %s",
                        chat.type if chat else None)
            return
        return await handler(update, context)

    return guarded


def private_only(handler):
    """Allow this command only in the admin's own private chat.

    A backup archive holds the whole database, so it must never be posted into
    a group or handed to a stranger who opened a private chat. Both the chat
    type and the sender are checked, because an open private chat would
    otherwise be enough to pull the whole dataset.
    """
    @wraps(handler)
    async def guarded(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        if not chat or chat.type != ChatType.PRIVATE:
            logger.info("Ignoring %s outside private chat", handler.__name__)
            return
        if not is_admin(update):
            logger.info("Ignoring %s from non-admin %s", handler.__name__,
                        update.effective_user.id if update.effective_user else None)
            return
        return await handler(update, context)

    return guarded


def _chunks(text: str, limit: int = 4096) -> list[str]:
    """Split on line breaks so a long answer never trips Telegram's size limit."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            parts.append(current)
            current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            parts.append(current)
            current = line
        else:
            current += line
    if current:
        parts.append(current)
    return [p for p in parts if p]


async def show_thinking(message) -> None:
    """Light up Telegram's typing indicator, ignoring chats that refuse it.

    A missing indicator must never cost the student their answer, so a failure
    here is swallowed rather than raised into the handler.
    """
    chat = getattr(message, "chat", None)
    if chat is None:
        return
    try:
        await chat.send_action("typing")
    except Exception:
        logger.debug("Could not show the typing indicator", exc_info=True)


async def thinking_pause(message) -> None:
    """Wait before answering so the bot reads as thinking, not as a lookup.

    The indicator is refreshed while waiting because Telegram clears it after a
    few seconds; without that the student would watch it blink out and assume
    the bot had died. A delay of 0 answers immediately.
    """
    remaining = settings.answer_delay_seconds
    while remaining > 0:
        await show_thinking(message)
        step = min(TYPING_ACTION_TTL, remaining)
        await asyncio.sleep(step)
        remaining -= step


def note_outcome(chat_id: int | None, sent: bool, error: str = "") -> None:
    """Record whether a reply actually left the bot, and if not, why.

    The routing counters say the bot decided to answer. Nothing said whether
    the answer ever reached Telegram, and that is the one step left unobserved:
    a bot that decides to answer and then dies in the middle of its handler
    looks from the outside exactly like a bot that was never asked.
    """
    if chat_id is None:
        return
    note = _ROUTING_NOTES.get(chat_id)
    if not note:
        return
    if sent:
        note["sent"] = note.get("sent", 0) + 1
        note["failed"] = 0
    else:
        note["failed"] = note.get("failed", 0) + 1
        note["err"] = error[:200] or None
    note["at"] = int(time.time())


# The last message the bot sent in each chat, so /del can take it back without
# the owner having to find and reply to the exact message. Keyed by chat id
# rather than by anything about the message, because the command and the message
# it removes do not have to arrive in the same order: an announcement goes out to
# the group and the owner decides to kill it while reading it there.
#
# Only ever written with messages the bot itself sent, which is what makes a
# remembered id safe to delete: a bot may delete its own messages, and the
# alternative -- deleting whatever a reply points at -- is a way to erase
# whatever a stranger puts in front of the command.
#
# Bounded by age because the bound is Telegram's, not ours. deleteMessage
# refuses anything older than 48 hours outright, and no argument or delay gets
# around it: a bot cannot delete a message a week old, and only a human admin
# tapping "delete" in the app can. So the memory is kept for exactly that
# window and not a minute longer -- an entry that outlived it would not be a
# longer memory, it would be an id a bare "/del" finds and then fails on, which
# is worse than not remembering. Nothing is lost by the choice: the reply shape
# takes the id from the message it points at, so a bare "/del" is the only path
# that needs the memory, and the owner who has the message in front of them
# replies instead.
LAST_MESSAGE_MEMORY_SECONDS = 48 * 3600.0
_last_sent: dict[int, tuple[int, float]] = {}


def remember_sent(chat_id: Any, sent: Any) -> None:
    """Note a message the bot just sent, so /del can find it again."""
    message_id = getattr(sent, "message_id", None)
    if chat_id is None or message_id is None:
        return
    _last_sent[chat_id] = (message_id, time.time())


def last_sent_in(chat_id: Any) -> int | None:
    """The bot's last message id in a chat, if one is still remembered."""
    entry = _last_sent.get(chat_id)
    if entry is None:
        return None
    message_id, seen = entry
    if time.time() - seen > LAST_MESSAGE_MEMORY_SECONDS:
        _last_sent.pop(chat_id, None)
        return None
    return message_id


async def reply_answer(message, text: str) -> None:
    """Send an answer, splitting it if it exceeds what Telegram accepts.

    A busy group can trip Telegram's per-group flood limit, which would silently
    drop the reply, so RetryAfter is honoured instead of being swallowed.
    """
    chat_id = getattr(getattr(message, "chat", None), "id", None)
    for chunk in _chunks(text):
        for attempt in range(3):
            try:
                remember_sent(chat_id, await message.reply_text(chunk))
                note_outcome(chat_id, True)
                break
            except RetryAfter as exc:
                if attempt == 2:
                    logger.error("Gave up sending after repeated rate limits")
                    note_outcome(chat_id, False, "rate limited by Telegram")
                    return
                wait = min(float(exc.retry_after or 1), 30.0)
                logger.warning("Telegram rate limit, waiting %.1fs before retrying", wait)
                await asyncio.sleep(wait)


# Words that wake the bot up in a group, matched as whole words so that
# "paleo" and "cleopatra" do not trip it. The Arabic spelling is matched after
# folding, otherwise "ليُو" with a harakah, a Persian yeh, or the doubled waw of
# a fast typist would all slip past and the bot would ignore its own name.
_NAME_WORDS = ("leo", "ليو")
_DIACRITIC_RANGES = ((0x064B, 0x0652), (0x0670, 0x0670), (0x06D6, 0x06ED))
# Marks that carry no letter of their own. A harakat decorates a letter, but a
# tatweel or a zero-width joiner sits *between* two letters and is dropped by
# plenty of phone keyboards and by copy-paste from a web page. Leaving them in
# turns "ليو" into something the pattern below does not recognise, and the bot
# then ignores a student writing its own name.
_IGNORED_CHARS = (
    {chr(code) for low, high in _DIACRITIC_RANGES for code in range(low, high + 1)}
    | set("\u0640"          # tatweel, typed as a straight line
          "\u200b"          # zero width space
          "\u200c"          # zero width non-joiner
          "\u200d"          # zero width joiner
          "\u200e\u200f"    # left-to-right / right-to-left marks
          "\u061c"          # Arabic letter mark
          "\u2060"          # word joiner
          "\ufeff")         # byte order mark
)
# Spellings that mean the same letter to a reader but not to a regex. Arabic
# keyboards disagree about yeh, and plenty of phones emit the Persian one.
_LETTER_FOLD = {
    "\u06CC": "\u064A",  # yeh  -> yeh
    "\u0649": "\u064A",  # alef maksura -> yeh
    "\u06D2": "\u064A",  # yeh barree -> yeh
    "\u06C6": "\u0648",  # waw with hamza above -> waw
    "\u06C0": "\u0647",  # heh with yeh above -> heh
    "\u06A9": "\u0643",  # keheh -> kaf
    "\u06AF": "\u0643",  # gaf -> kaf
}
# The waw is optional and repeatable so "ليو", "ليوو" and "ليؤو" are all the
# name, while the closing boundary keeps a word that merely starts with it out.
_NAME_RE = re.compile(
    r"(?<!\w)(?:leo|لي[وؤ][وؤ]?)(?!\w)", re.IGNORECASE)


def _fold(text: str) -> tuple[str, list[int]]:
    """Fold the text into a matchable form and keep a map back to the original.

    The ignored marks are dropped so a name split by one of them still matches,
    and dropping them also shortens the string, so the offset of every surviving
    character is recorded. Slicing the original with a folded offset would cut
    the wrong characters: in "مَرْحَبًا ليو" the two marks before the name are
    gone, the match lands two places early, and the name is left in the question
    while the greeting loses its last two letters.

    A character is also normalised on its own, which is what a paste out of Word
    or a lesson PDF needs. Such text arrives in the Arabic Presentation Forms
    block -- U+FEDD, U+FEF1, U+FEED for the name -- where the same three letters
    look identical to the ones a keyboard sends and are different codepoints.
    Matching one set and not the other means the bot answers a name a student
    typed and stays silent for the same name they pasted from the document the
    question was written in. Normalising per character rather than per string
    keeps the origin map exact: a ligature that expands to two letters records
    the one index it came from twice, so cutting the original still removes the
    whole character and not half of it.
    """
    folded: list[str] = []
    origin: list[int] = []
    for index, ch in enumerate(text):
        if ch in _IGNORED_CHARS:
            continue
        for piece in unicodedata.normalize("NFKC", ch):
            folded.append(_LETTER_FOLD.get(piece, piece))
            origin.append(index)
    return "".join(folded), origin


def _name_spans(text: str) -> list[tuple[int, int]]:
    """Where the bot's name appears, as offsets into the original text."""
    folded, origin = _fold(text)
    return [(origin[match.start()], origin[match.end() - 1] + 1)
            for match in _NAME_RE.finditer(folded)]


def is_name_call(text: str) -> bool:
    """True when the message says the bot's name as a whole word.

    Students do not always bother with the @handle, so in a group a plain
    "leo" or "ليو" is enough. Word boundaries matter: without them every
    message mentioning paleontology or Cleopatra would wake the bot.
    """
    return bool(_name_spans(text))


def is_command(message: Any) -> bool:
    """Whether this message is a slash command.

    PTB's own filter for this reads the message entities. Reusing the same test
    here means the watcher and the handlers agree on what a command is, instead
    of the watcher inventing a category the handlers never treat as one.
    """
    entities = getattr(message, "entities", None) or ()
    return any(getattr(entity, "type", None) == "bot_command" for entity in entities)


# The last few group messages, per chat, so the relevance gate can read the
# conversation instead of one line lifted out of it. Text only, and only the tail:
# enough to know what the class has been discussing, not a transcript.
#
# Held in memory and never written to the database. The dataset is for question
# and answer pairs the tutor produced, and this is other students' chat, so
# putting it there would make it survive a restart and end up in an export --
# which is not what anyone asked for when they sent a message to a group.
GROUP_CONTEXT_MESSAGES = 12
# Long enough to cover a class period's exchange, short enough that a message
# from this morning cannot make tonight's message look like a follow-up to it.
GROUP_CONTEXT_SECONDS = 1800.0
_group_context: dict[int, list[tuple[float, str]]] = {}


def remember_group_message(chat_id: int, message: Any) -> None:
    """Keep the tail of a group's text, for the relevance gate to read later."""
    if chat_id is None or is_command(message):
        return
    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if not text:
        # An attachment with no words carries no signal for the gate, and its
        # bytes are handled by the media handlers. It is described as a shape so
        # the gate knows something arrived without being handed the file.
        if any(getattr(message, field, None) for field in ("photo", "voice", "document")):
            text = "[صورة]" if getattr(message, "photo", None) else (
                "[رسالة صوتية]" if getattr(message, "voice", None) else "[ملف]")
        else:
            return
    entries = _group_context.setdefault(chat_id, [])
    entries.append((time.time(), text[:600]))
    now = time.time()
    # Cut by age as well as by count: a chat id never goes away, and a class that
    # stopped months ago should not keep its last words on the machine.
    _group_context[chat_id] = [e for e in entries
                               if now - e[0] <= GROUP_CONTEXT_SECONDS][-GROUP_CONTEXT_MESSAGES:]


def recent_group_messages(chat_id: int) -> list[str]:
    """The last group messages, oldest first, for the gate to read."""
    now = time.time()
    entries = [e for e in _group_context.get(chat_id, ()) if now - e[0] <= GROUP_CONTEXT_SECONDS]
    return [text for _, text in entries]


def forget_group_context(chat_id: int) -> None:
    """Drop a chat's remembered lines, so /forget clears it too."""
    _group_context.pop(chat_id, None)


async def log_arriving_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Record that an update arrived at all, whatever it turned out to be.

    The routing log inside should_answer only fires for a message that
    reached a handler. If Telegram delivers something this bot has no handler
    for -- an edited message, a poll, a giveaway -- that leaves no trace at all,
    and a group that has gone quiet cannot be told apart from one whose messages
    never arrived. Registered with block=False, so it observes without taking
    the update away from whoever is supposed to answer it. The shape is logged,
    never the text.
    """
    chat = update.effective_chat
    message = update.effective_message
    if getattr(update, "my_chat_member", None) is not None:
        # The one update that says whether the bot is even in this chat. A
        # group the bot was never added to produces no messages at all, and
        # without this there is no way to tell that from a routing mistake.
        change = update.my_chat_member
        logger.info("update %s: my_chat_member chat=%s title=%r status=%s -> %s",
                    getattr(update, "update_id", None),
                    getattr(chat, "id", None), getattr(chat, "title", None),
                    getattr(change.old_chat_member, "status", None),
                    getattr(change.new_chat_member, "status", None))
        # Being added to a group is the most reliable moment to learn where an
        # owner announcement should go, and it is the only one that happens when
        # the group is otherwise silent.
        if chat is not None and chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
            remember_announce_group(chat.id)
        return
    kind = "message"
    if update.callback_query is not None:
        kind = "callback"
    elif getattr(update, "edited_message", None) is not None:
        kind = "edited"
    elif getattr(update, "channel_post", None) is not None:
        kind = "channel"
    elif update.message is None:
        kind = "other"
    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "")
    logger.info(
        "update %s: %s chat=%s type=%s %d char(s) reply=%s",
        getattr(update, "update_id", None), kind,
        getattr(chat, "id", None), getattr(chat, "type", None), len(text),
        "yes" if getattr(message, "reply_to_message", None) else "no")
    if chat is not None and chat.type == ChatType.PRIVATE and message is not None:
        # A question forwarded into the owner's private chat has to be
        # answerable by a follow-up "/a" that carries no forward of its own, and
        # the two messages have no link to each other. The watcher is the only
        # thing that sees both, so this is where the pairing is kept -- and
        # still without awaiting, since it sits in front of every update.
        forwarded = _origin_chat_id(message)
        if forwarded is not None:
            remember_forward_chat(message, forwarded)
    if chat is not None and chat.type != ChatType.PRIVATE:
        remember_announce_group(chat.id)
        # The tail of this group, kept so the relevance gate can read a
        # question in the context it was asked in. In memory only and never
        # written out, and the bot's own answer is skipped so it cannot answer
        # itself.
        if kind == "message" and not (
                getattr(getattr(message, "from_user", None), "is_bot", False)):
            remember_group_message(chat.id, message)
        # What gets recorded is the shape, decided by what the message carries
        # rather than by which handler claimed it, so a message nobody handles
        # is still visible in /status.
        #
        # The media tests ask whether the field is present, not whether it is not
        # None. Telegram leaves an absent list as an empty tuple, so "photo is
        # not None" is true of every message that ever existed: plain text was
        # reported as a picture, and three rounds of testing chased a photograph
        # nobody had sent. What is absent is an empty collection.
        #
        # Nothing is awaited here, and that is the whole point. This watcher is
        # on the path of every single update, ahead of the handler that is meant
        # to answer the student, and updates are processed one after another. A
        # remote write costs about a second, and the client has no timeout, so a
        # slow moment used to hold every later message hostage behind it -- one
        # image was counted here and then never reached the handler, because the
        # write stalled in between. The timer writes the note instead.
        if kind != "message":
            shape = kind
        elif getattr(message, "photo", None):
            shape = "صورة"
        elif getattr(message, "voice", None) is not None:
            shape = "صوت"
        elif getattr(message, "audio", None) is not None:
            shape = "صوت"
        elif getattr(message, "video", None) or getattr(message, "video_note", None) is not None:
            shape = "فيديو"
        elif getattr(message, "document", None) is not None:
            shape = "ملف"
        elif getattr(message, "sticker", None) is not None:
            shape = "ملصق"
        elif text:
            shape = "نص"
        else:
            shape = "شي"
        # A command is handled by a command handler, which never asks for a
        # routing decision, so counting it would invent an arrival that reached
        # no handler -- a fault that does not exist, reported next to real ones.
        if not is_command(message):
            note_arrival(chat.id, shape, len(text),
                         bool(getattr(message, "reply_to_message", None)))


ROUTING_FLAG = "routing"
# What each chat actually sends us, kept in memory and flushed to the database
# at most once a minute. A group going quiet is the hardest thing in this bot to
# diagnose, because the only other place the answer lives is a Render log that
# nobody has open. Counting the decisions here lets /status report them in
# Telegram, which is the difference between a five second check and an hour of
# guessing. No text is kept, only which branch answered.
_ROUTING_NOTES: dict[int, dict[str, Any]] = {}
_ROUTING_WRITTEN: dict[int, float] = {}
ROUTING_FLUSH_SECONDS = 60.0
ROUTING_RECENT = 5

# Update kinds that arrive already in English; a message's own shape is named in
# Arabic by the watcher and passes through untouched.
_SHAPE_WORDS = {
    "edited": "تعديل", "channel": "قناة", "service": "خدمة", "other": "شي",
    "callback": "زر", "message": "رسالة",
}


def _routing_note(chat_id: int) -> dict[str, Any]:
    return _ROUTING_NOTES.setdefault(
        chat_id, {"seen": 0, "name": 0, "mention": 0, "reply": 0, "gate": 0,
                  "gate-soft-no": 0, "no": 0, "last": None, "at": 0,
                  "recent": []})


# Which way in this message used, in words a student can act on.
_WHY_WORDS = {
    "name": "✔️ عرفها: اسمك أولها",
    "mention": "✔️ عرفها: منشنك",
    "reply": "✔️ عرفها: رد على رسالتي",
    "gate": "✔️ عرفها: البوابة قرأت إنها سؤال للمعلّم",
    "gate-soft-no": "⚠️ البوابة ماگالت إنها إله — ردّيت على كل حال",
    "no": "✔️ ما فيها اسمك، وردت على كل شي",
}


def describe_shape(kind: str, chars: int, is_reply: bool) -> str:
    """Say what a message looked like, in words, without quoting it.

    A person who sent "ليو سؤال" into a silent group needs to see whether the
    message arrived at all. The counters alone cannot tell that apart from a
    message that arrived and was not recognised: both read zero. So the shape is
    reported instead -- a text of eight characters is unmistakable evidence that
    the message arrived, and only the length is disclosed, never the words.
    """
    word = _SHAPE_WORDS.get(kind, kind)
    text = f"{chars} حرف" if chars else "بدون نص"
    return f"{word} · {text}" + (" · رد على رسالة" if is_reply else "")


def note_arrival(chat_id: int, kind: str, chars: int, is_reply: bool) -> None:
    """Count the arrival and keep the last few shapes, for /status to show.

    "seen" is counted here, in the watcher, rather than in the routing
    decision, so that it means what a person reading it expects: how many
    messages arrived. When the two numbers disagreed -- two shapes listed,
    one decision counted -- the only way to see it was to know that a message
    had reached no handler at all, which is precisely the thing worth seeing.
    An arrival with no decision counted is now visible as the gap between the
    total and the reasons, instead of being invisible.
    """
    note = _routing_note(chat_id)
    note["seen"] = note.get("seen", 0) + 1
    note.setdefault("recent", []).append(
        {"shape": describe_shape(kind, chars, is_reply), "at": int(time.time())})
    del note["recent"][:-ROUTING_RECENT]
    note["at"] = int(time.time())


def note_routing(chat_id: int, reason: str) -> None:
    note = _routing_note(chat_id)
    if reason in note:
        note[reason] += 1
    note["last"] = reason
    # The watcher runs immediately before the handler, so the arrival sitting at
    # the end of the list is this message. Recording the verdict beside its shape
    # is what turns "I got nothing" into "your caption had no name in it" -- the
    # totals alone say only that something was ignored, not which of the five
    # ways in it missed.
    recent = note.get("recent") or []
    if recent:
        recent[-1]["why"] = reason
    note["at"] = int(time.time())


def note_fault(chat_id: int, where: str, exc: BaseException) -> None:
    """Remember a failure in the machinery itself, so /status can show it.

    Every fault so far in this bot has been visible only in a Render log that
    nobody had open at the time: a handler that raised, a database that would
    not answer, a message that reached nothing. Each of those looks identical
    from inside Telegram, which is silence. This is the one place the reason is
    written where the person waiting for an answer will actually read it.
    """
    note = _routing_note(chat_id)
    note["err"] = f"{where}: {type(exc).__name__}: {exc}"
    note["at"] = int(time.time())


def note_pause_read_failed(chat_id: int) -> None:
    note = _routing_note(chat_id)
    note["pause_db"] = "ما قدرت أقرا علم الإيقاف"
    note["at"] = int(time.time())


async def note_handler_failure(update: object, error: BaseException) -> None:
    """Write the reason a handler fell over into the chat's own note.

    PTB logs this and moves on, which leaves the student with no answer and the
    bot's owner with a log line they have to know to go looking for. The chat id
    is all it takes to put the reason in front of them instead.
    """
    chat = getattr(update, "effective_chat", None)
    if chat is not None and chat.type != ChatType.PRIVATE:
        note_fault(chat.id, "معالج", error)
    logger.exception("Handler failed", exc_info=error)


async def flush_routing(chat_id: int, force: bool = False) -> None:
    """Persist the note, at most once a minute, and never at the cost of a reply.

    Every message would otherwise become a write to the database, and the
    counter only exists for a human to read later, so it waits its turn. A
    failure to record it is not worth surfacing to a student.
    """
    note = _ROUTING_NOTES.get(chat_id)
    if not note:
        return
    now = time.time()
    if not force and now - _ROUTING_WRITTEN.get(chat_id, 0.0) < ROUTING_FLUSH_SECONDS:
        return
    _ROUTING_WRITTEN[chat_id] = now
    try:
        await write_flag(settings.database_path, f"{ROUTING_FLAG}:{chat_id}",
                         json.dumps(note, ensure_ascii=False))
    except Exception:
        logger.debug("Could not record the routing note", exc_info=True)


async def flush_all_routing() -> None:
    """Write every chat's tally, on a timer, so the last word is never lost.

    The watcher flushes on the way in, but it runs before the handlers, so the
    most recent message of a quiet group would sit in memory forever. A group
    that says one thing and then goes silent is exactly the one whose count
    matters, so the tail is written here whether or not more traffic arrives.
    """
    for chat_id in list(_ROUTING_NOTES):
        await flush_routing(chat_id, force=True)


async def routing_flush_loop() -> None:
    """Keep the tallies current for as long as the bot runs.

    Not the job queue: that needs APScheduler, which is not a dependency here,
    and the codebase already runs its other periodic work as a plain task.
    """
    while True:
        await asyncio.sleep(ROUTING_FLUSH_SECONDS)
        try:
            await flush_all_routing()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Could not write the routing tallies", exc_info=True)


async def read_routing(chat_id: int) -> dict[str, Any] | None:
    """This run's tally for a chat, written out on the way so it survives a crash.

    Only the running process's own counts are reported. The database also
    holds the tally of whichever run wrote last, and after a restart that is a
    dead one: /status quoted 16 messages for a group the live bot had never
    seen a message in, which is not a small detail to get wrong when the whole
    question is whether messages are arriving.
    """
    note = _ROUTING_NOTES.get(chat_id)
    if not note:
        return None
    await flush_routing(chat_id, force=True)
    return note


def routing_reason(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Which of the four ways in this message used, as a name we can count.

    "private" is the fourth: a direct message is always for the bot. The name
    matters as much as the answer, because "no" is the only one that means the
    message arrived and was still ignored, which is the whole puzzle when a
    group stops answering.
    """
    message = update.effective_message
    if not message:
        return "no"
    if message.chat.type == ChatType.PRIVATE:
        return "private"
    bot_username = context.bot.username or ""
    mention = f"@{bot_username.lower()}" if bot_username else ""
    text = message.text or message.caption or ""
    if mention and mention.lower() in text.lower():
        reason = "mention"
    elif is_name_call(text):
        reason = "name"
    elif (message.reply_to_message and message.reply_to_message.from_user
          and message.reply_to_message.from_user.id == context.bot.id):
        reason = "reply"
    else:
        reason = "no"
    logger.info(
        "group %s: msg %s, %d char(s), bot username %s -> %s",
        getattr(message.chat, "title", None) or message.chat.id,
        getattr(message, "message_id", None), len(text),
        bot_username or "(none)", reason)
    return reason


async def should_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Whether this message is for the bot. In a group: always yes, except bots.

    This used to be a decision, and it went wrong in a way that could not be seen
    from the outside. The owner asked for a tutor that engages with everything,
    and the count in the database said the gate had been consulted zero times in
    eight messages: every message that arrived had been dropped earlier, by the
    rules that only answered a name or a reply to the bot. So the gate was
    working and simply never reached. Widening it was not enough, because a
    message like "ليو شلونك" is answered by the name path in under a millisecond
    and never asks a model anything -- which is right, and also means the quiet
    could never be fixed from the gate alone.

    So the gate is now advisory rather than final. It runs on a message that
    arrived at nothing, so its verdict is still counted under "gate" and /status
    can show it, but a "no" no longer silences the room. The bot reads what the
    group says and answers; the prompt tells it when to keep quiet of its own
    accord, in a single short line, rather than dropping the message on the floor
    where nobody sees the decision being made.

    One exclusion survives, and it has to be here rather than in the gate: another
    bot's message. A class group runs bots, and "answer everything" taken
    literally would mean answering all of them, forever, in a loop that costs
    quota on both sides and reads as a malfunction to the class. A student is
    never excluded; a machine is.
    """
    message = update.effective_message
    if message is not None and getattr(
            getattr(message, "from_user", None), "is_bot", False):
        logger.info("Ignoring a message from another bot")
        return False
    reason, chat_id = _judge(update, context)
    if reason != "no":
        if chat_id is not None:
            note_routing(chat_id, reason)
        return True
    if chat_id is not None:
        # Asked anyway, so the count and the log show what the gate thought, and
        # so turning it off later is a one-line change rather than a dig. The
        # verdict no longer decides anything.
        if await gate_reads_it_as_a_question(update, chat_id):
            note_routing(chat_id, "gate")
        else:
            note_routing(chat_id, "gate-soft-no")
    return True


def _judge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, int | None]:
    """The routing reason, and the chat to record it against, or None.

    A private message is not a routing decision at all, so it has no chat id to
    be counted under; the four group reasons all have one.
    """
    reason = routing_reason(update, context)
    chat = getattr(update, "effective_chat", None)
    if reason == "private" or chat is None:
        return reason, None
    return reason, chat.id


def _addressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Whether the student aimed this message at the bot, ignoring the gate.

    Answering a message and keeping it are two different decisions, and the gap
    between them only became visible once the bot started answering everything.
    The dataset is the owner's record of questions and the model's answers, and it
    is what the fine-tune is built from, so a message the student did not ask the
    bot is not a training example of anything. In a busy group that is now the
    majority of what arrives, and storing it would bury the real questions under a
    class's small talk -- so it would be worse, not merely noisier.

    So a message only joins the dataset when it was actually addressed: named,
    mentioned, or a reply to the bot. It is still answered, still remembered in
    the conversation, and still costs the same quota. The archive is the only
    thing that needs the student's words to have been aimed at us.

    This is the single place that distinction is made, and every handler that
    writes to the dataset asks here rather than re-deciding it for itself. The
    gate is deliberately not consulted: by the time anything reaches the archive
    the gate is advisory, and letting it in here would put a stranger's opinion
    back in charge of what is kept.
    """
    reason, _ = _judge(update, context)
    return reason != "no"


async def gate_reads_it_as_a_question(update: Update, chat_id: int) -> bool:
    """Ask the separate-key gate whether this message is meant for the bot.

    A second key and a second client, not a flag on the existing one. The gate
    runs on every message a group sends, including the ones nobody wanted
    answered, which is a different kind of traffic from the questions the tutor
    actually answers and deserves its own quota and its own model choice.

    Every failure answers no. A missing key, a busy model, a refused call or a
    verdict that is neither yes nor no all mean the same thing, and it is the
    thing that was true before this existed. What changed afterwards is not the
    gate but what a no is worth: it is recorded and printed in /status, and it no
    longer decides whether the student gets an answer. The counter is the point,
    not the veto -- a gate whose opinion is never recorded is indistinguishable
    from a gate that is not running.
    """
    if router is None:
        return False
    message = update.effective_message
    if message is None or is_command(message):
        return False
    if getattr(getattr(message, "from_user", None), "is_bot", False):
        # Another bot's message is never a question for this one, and in a group
        # that runs bots the calls would otherwise cost a request each.
        return False
    text = message.text or message.caption or ""
    try:
        verdict = await router.wants_reply(text, recent_group_messages(chat_id))
    except Exception:
        # A gate that raises must not take the tutor down: the reply it was
        # deciding about is still answerable when it was named.
        logger.warning("Relevance gate failed, staying quiet",
                       exc_info=True)
        return False
    if verdict is None:
        return False
    logger.info("group %s: gate says %s for message %s",
                chat_id, "yes" if verdict else "no",
                getattr(message, "message_id", None))
    return verdict


def directed_to_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Whether this message is the bot being spoken to, by the three sure signs.

    The handlers do not call this: they call should_answer, which is this plus
    the relevance gate. It stays as the plain routing decision because it is
    answerable without a network call, and because "would the bot have picked
    this up before the gate existed" has to stay checkable on its own.

    In a group there are three ways in, checked in the order a student is most
    likely to use them: the @handle, the bare name, or a reply to the bot. The
    decision is logged with the chat and the reason but never the text, since a
    group going quiet leaves no other trace and a student's question has no
    business being written to disk just to explain routing.
    """
    reason = routing_reason(update, context)
    chat = getattr(update, "effective_chat", None)
    if reason != "private" and chat is not None:
        note_routing(chat.id, reason)
    return reason != "no"


def clean_prompt(text: str, bot_username: str = "") -> str:
    """Strip the bot mention and clamp to the configured limit."""
    if bot_username:
        text = re.sub(rf"@{re.escape(bot_username)}\b", "", text, flags=re.IGNORECASE)
    # A bare "leo" is a wake-up call, not part of the question, so it should
    # never reach the model as content. The span is located in the folded text
    # but cut from the original, so a message written with a Persian yeh loses
    # exactly its own name and keeps the rest of its spelling.
    spans = _name_spans(text)
    if spans:
        pieces, cursor = [], 0
        for start, end in spans:
            pieces.append(text[cursor:start])
            cursor = end
        pieces.append(text[cursor:])
        text = "".join(pieces)
    text = re.sub(r"\s{2,}", " ", text).strip(" -:،,")
    return text[: settings.max_message_chars]


def thread_key(update: Update) -> tuple[int, int] | None:
    """Identify whose memory this update belongs to, or None if it is unknown.

    A channel post has no author we can attribute, and a channel is not a
    conversation with one student, so it gets no memory at all.
    """
    chat, user = update.effective_chat, update.effective_user
    if not chat or not user:
        return None
    return chat.id, user.id


PAUSED_FLAG = "paused"
# A pause belongs to the chat it was typed in, not to the whole bot: the same
# bot can be wanted in one group and quiet in another. The per-chat truth is
# written to the database so a restart does not undo a pause; this cache only
# saves the round trip, and the single Render instance owns the truth.
_paused: dict[int, bool] = {}


def _pause_flag(chat_id: int) -> str:
    return f"{PAUSED_FLAG}:{chat_id}"


async def is_paused(chat_id: int) -> bool:
    if chat_id not in _paused:
        try:
            _paused[chat_id] = (await read_flag(settings.database_path,
                                               _pause_flag(chat_id))) == "1"
        except Exception:
            # This sits in front of every message handler, and the remote client
            # has no timeout, so an unreachable database used to take the handler
            # down with it: the update was counted by the watcher and then never
            # answered, with nothing anywhere saying the database was the reason.
            # A pause is a choice to be quiet, so when we cannot read the choice
            # we answer instead -- a student gets a reply, and the outage shows
            # in /status instead of as silence.
            logger.exception("Could not read the pause flag for chat %s", chat_id)
            _paused[chat_id] = False
            note_pause_read_failed(chat_id)
    return _paused[chat_id]


async def set_paused(chat_id: int, paused: bool) -> None:
    _paused[chat_id] = paused
    await write_flag(settings.database_path, _pause_flag(chat_id),
                     "1" if paused else "0")


ANNOUNCE_GROUP_FLAG = "announce_group"
# The group an owner announcement goes to. There is no group setting in the
# environment and there should not be one: the bot is in whichever groups it was
# added to, and the owner announcing homework to the wrong class is a worse
# mistake than any missing feature. So the group is whichever one the bot last
# heard from, which is the group being used right now.
_announce_group: int | None = None


# The most recent question forwarded into each private chat, kept as the message
# itself and not just its group. The owner forwards and then types "/a" as a
# separate message, and neither carries a link to the other, so the command has
# to find both *where* the question came from and *what* it said. Remembering
# only the group left a bare "/a" with nowhere to post and nothing to ask, and
# it reported that no question had been forwarded seconds after one was.
#
# Keyed by the chat object rather than its id because it is only ever read while
# handling that same chat's next message, and holding one chat's question under
# another's key would answer the wrong class.
_last_forward: dict[int, tuple[Any, int]] = {}

# How long a remembered forward stays usable. Long enough to cover the seconds
# between forwarding and typing, short enough that a forgotten forward does not
# send tomorrow's answer to yesterday's class. Only consulted when the message
# carries no origin of its own, so a real forward is never affected.
FORWARD_MEMORY_SECONDS = 900.0
_last_forward_at: dict[int, float] = {}


def remember_forward_chat(message: Any, chat_id: int) -> None:
    """Note the question just forwarded into this private chat, and where it came from."""
    chat = getattr(message, "chat", None)
    if chat is None:
        return
    _last_forward[id(chat)] = (message, chat_id)
    _last_forward_at[id(chat)] = time.time()


def _expire_forward(chat_key: int) -> None:
    seen = _last_forward_at.get(chat_key)
    if seen is not None and time.time() - seen > FORWARD_MEMORY_SECONDS:
        _last_forward.pop(chat_key, None)
        _last_forward_at.pop(chat_key, None)


async def _store_announce_group(chat_id: int) -> None:
    try:
        await write_flag(settings.database_path, ANNOUNCE_GROUP_FLAG, str(chat_id))
    except Exception:
        # The running process already knows the group, so only a restart would
        # lose it, and the next change writes it again. Failing the write is not
        # worth an announcement that has already been delivered.
        logger.warning("Could not remember the announcement group %s", chat_id,
                       exc_info=True)


def remember_announce_group(chat_id: int) -> None:
    """Note a group the bot has just heard from, without waiting for anything.

    Called from the watcher, which sits in front of every single update, so this
    is the one place that must not await: a remote write costs about a second and
    the client has no timeout, which is exactly how one slow image used to hold
    every later message hostage. The write is therefore scheduled, not awaited,
    and the group only changes when the bot is added somewhere new -- so the
    deferred task is rare rather than per-message.
    """
    global _announce_group
    if not chat_id or chat_id == _announce_group:
        return
    _announce_group = chat_id
    try:
        asyncio.create_task(_store_announce_group(chat_id))
    except RuntimeError:
        logger.debug("No running loop; the group will be stored on the next change")


@admin_only
async def show_announce_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Report where an announcement will go, and what /post accepts.

    The group is learned from traffic, which means its value is invisible until
    it is wrong -- and "the wrong class got my announcement" is only debuggable
    after the fact. Asking the bot is the cheap way to find out first.
    """
    message = update.effective_message
    chat_id = await load_announce_group()
    if chat_id is None:
        await message.reply_text(
            "ما حددت مجموعة بعد.\n"
            "اكتب /post -1001234567890 الرسالة، أو /post @اسم_المجموعة الرسالة.")
        return
    try:
        chat = await context.bot.get_chat(chat_id)
        title = getattr(chat, "title", None) or "بدون اسم"
    except Exception as exc:
        logger.warning("Could not name chat %s: %s", chat_id, exc)
        title = "ما أگدر أعرف اسمها"
    await message.reply_text(f"الإعلانات تروح لـ: {title}\n{chat_id}")


async def load_announce_group() -> int | None:
    """Recover the announcement group after a restart, when nothing has arrived yet.

    A fresh container has heard from nobody, so until the first group message
    the owner has no group to post to and the command would fail for a reason
    that has nothing to do with the announcement.
    """
    global _announce_group
    if _announce_group is not None:
        return _announce_group
    try:
        stored = await read_flag(settings.database_path, ANNOUNCE_GROUP_FLAG)
    except Exception:
        logger.warning("Could not read the announcement group", exc_info=True)
        return None
    if stored and stored.lstrip("-").isdigit():
        _announce_group = int(stored)
    return _announce_group


def active_only(handler):
    """Stay completely silent in a chat that is paused.

    A student must not be able to tell a paused bot from an offline one, so
    nothing is sent at all: no reply, no hint, no Gemini call. /start is the one
    way back, which is why it is deliberately not wrapped in this.
    """
    @wraps(handler)
    async def guarded(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        if chat is not None and await is_paused(chat.id):
            return
        return await handler(update, context)

    return guarded


@allowed_chat_only
async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop answering in this chat until somebody here runs /start.

    Open to any member on purpose: the group is the bot's audience, and the
    owner asked for the switch to be a group decision rather than a private one.
    The pause is still durable, so a mistaken press is undone with /start and
    never by a restart. Other chats are untouched.
    """
    message = update.effective_message
    chat_id = update.effective_chat.id
    if await is_paused(chat_id):
        await message.reply_text("البوت مسكوت بهالمحادثة أصلاً. أرسل /start يرجّعه.")
        return
    await set_paused(chat_id, True)
    logger.warning("Bot paused in chat %s by user %s; silence there until /start",
                   chat_id, update.effective_user.id)
    await message.reply_text(
        "سكّيت البوت بهالمحادثة بس. باقي المحادثات تضل تشتغل، "
        "وأول ما أحد هنا يرسل /start يرجع يرد.")


@allowed_chat_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if await is_paused(chat_id):
        # Anyone in that chat can bring it back, which is what makes the pause
        # temporary rather than a setting only the owner controls.
        await set_paused(chat_id, False)
        logger.warning("Bot resumed in chat %s by /start from %s",
                       chat_id, update.effective_user.id)
        await update.effective_message.reply_text(
            "رجّعت البوت، وياك راح يرد على الكل من هالحين. اسأل براحتك!")
        return
    await update.effective_message.reply_text(
        "هلا بيك! آني Leo، كيف أساعدك اليوم؟ "
        "دز سؤالك بالنص، صورة، أو رسالة صوتية، ووجّهها إليّ بالمجموعة بكتابة اسمي (leo أو ليو) أو بالرد على رسالتي. "
        "أتذكر آخر ٥ أسئلة منك، فتكدر تسأل «والثاني؟» وتكمل بنفس الموضوع."
    )


@private_only
@admin_only
async def sticker_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Teach the bot a sticker: send one here, name it, and it can answer with it.

    A sticker file_id only works for a bot that has the sticker in its own set, so
    there is nothing to fetch or guess -- the owner hands them over. Naming it is
    the part that matters: the tags are moods, and the bot decides by looking for
    that mood in the answer it already wrote, not by asking a model.
    """
    message = update.effective_message
    sticker = getattr(message, "sticker", None)
    tags = "، ".join(sorted(stickers.MOODS))
    if sticker is None:
        await message.reply_text(
            "دزلي ملصق هنا وسجله.\n"
            f"الاسم وحدة من هالأسماء: {tags}\n"
            "مثال: ترسل الملصق وبعده /sticker ضحك.")
        return
    raw = message.text or message.caption or ""
    matched = STICKER_COMMAND_RE.match(raw)
    asked = (raw[matched.end():] if matched else " ".join(context.args or [])).strip()
    if not asked:
        await message.reply_text(
            f"شو اسم هالملصق؟ وحدة من: {tags}\nمثال: /sticker ضحك")
        return
    tag = asked
    if tag not in stickers.MOODS:
        await message.reply_text(f"ما أعرف الوسم «{tag}». استخدم وحدة من: {tags}")
        return
    file_id = getattr(sticker, "file_id", None)
    if not file_id:
        await message.reply_text("ما أگدر أقرا هالملصق، جرّب وحد ثاني.")
        return
    stickers.remember(tag, file_id, STICKER_LIBRARY_PATH)
    await message.reply_text(
        f"سجلت «{tag}». راح أبعثه لمّا الجواب يناسبه، ومو كل رسالة.")


# Stickers the owner registered, next to the database rather than inside it.
STICKER_LIBRARY_PATH = Path("data/stickers.json")


async def maybe_send_sticker(message, answer: str) -> bool:
    """Add one sticker when the answer's own words call for one.

    Nothing here talks to the model: the sticker is chosen by finding a mood word
    in what Leo already wrote, so a wrong pick can only ever be a slightly odd
    reaction, and the model cannot waste sends by asking for stickers.
    """
    chat_id = getattr(getattr(message, "chat", None), "id", None)
    if chat_id is None:
        return False
    file_id = stickers.pick(answer, chat_id, STICKER_LIBRARY_PATH)
    if not file_id:
        return False
    try:
        remember_sent(chat_id, await message.get_bot().send_sticker(chat_id, file_id))
        return True
    except RetryAfter:
        logger.warning("Rate limited sending a sticker; skipping it")
    except (BadRequest, Forbidden):
        # A sticker the owner registered long ago can stop being usable, and a
        # failed reaction must never cost the answer it was following.
        logger.warning("Sticker %s could not be sent", file_id)
    except Exception:
        logger.exception("Sticker send failed")
    return False


@allowed_chat_only
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Stated from the live state of the gate rather than written once, because the
    # sentence is now a promise about behaviour: with the gate on, a plain question
    # is answered, and telling a class to always add a name would be wrong advice
    # that makes the feature look broken.
    in_group = (
        "بالgroup تكفي تسأل عادي، وأنا أقرأ السياق وأجاوب إذا السؤال كان للمجموعة كلها."
        if router is not None else
        "بالgroup ما أرد على كل الرسائل؛ اكتب اسمي (leo أو ليو) بأول السؤال، أو رد على رسالتي."
    )
    await update.effective_message.reply_text(
        "الأوامر المتاحة:\n/start - بدء الاستخدام\n/help - المساعدة\n/privacy - الخصوصية\n/status - حالة الخدمة\n/quiz الموضوع - اختبار\n/progress - نتائجك\n/forget - نسيان آخر المواضيع\n/cancel - إلغاء العملية الحالية\n"
        + in_group + "\n\n"
        "وتكتب /quiz تطلع لك قائمة باختصاصات الذكاء الاصطناعي تختار منها، أو تكتب /quiz الموضوع مباشرة مثل /quiz تعلم الآلة.\n"
        "كل اختبار ٥ أسئلة اختيار من متعدد، تجاوب بالأزرار (أ ب ج د) وتعرف النتيجة مع شرح ليش جوابك صح أو غلط.\n"
        "النتائج تنحفظ لكل طالب لحاله وتگدر تشوفها بـ /progress.\n\n"
        "و anyone يگدر يسكّت البوت بهالمحادثة بـ/pause، ويرجع يرد أول ما أحد من هنا يرسل /start.\n\n"
        "أگدر تقرا برضو ملفات PDF و GIF والمقاطع القصيرة، دزها مثل أي سؤال.\n\n"
        "وإذا تريدني أجاوب عن رسالة معيّنة، ترد عليها وتكتب /سؤال وسؤالك (أو /سؤال لحاله يعني اشرح الرسالة الي ردّيت عليها).\n\n"
        "أتذكر آخر ٥ أسئلة وجواباتها منك فقط (لكل مجموعة على حدة) عشان تكمل بنفس الموضوع. "
        "هذه الذاكرة مؤقتة بالجهاز وما تنحفظ بالداتابيز، وبتقدر تمسحها بـ /forget.\n\n"
        "تنظيم الحصص (للمدير):\n/newlesson العنوان | التاريخ | الوقت | السعة\n"
        "/addstudent رقم_الحصة | رمز_الطالب\n/roster رقم_الحصة\n"
        "ولكل الطلاب: /lessons لعرض الحصص والمقاعد المتبقية.\n\n"
        "للمدير: بعد ما تراجع المحتوى بـ /pending، اعتمده بـ /approve رقم_العنصر approved بالخاص، "
        "وبعدين /export يطلع ملف التدريب.\n\n"
        "وتنشر إعلان بالمجموعة: اكتب /post وبعدها الرسالة بالخاص، أو صوّر شي مع تعليق — "
        "بتروح للمجموعة باسم البوت، بنفس الشكل بالضبط.\n"
        "ومرة وحدة تحسم المجموعة: /post -1001234567890 الرسالة، أو /post @اسم_المجموعة الرسالة، "
        "وبعدها بروح لحاله. و /group يوريك وين تروح الإعلانات.\n\n"
        "وتجاوب سؤال محوّل: حوّل رسالة طالب من المجموعة للخاص واكتب /answer، "
        "بتجاوب بالمجموعة الي إجا منها السؤال، مو بالخاص.\n\n"
        "وإذا ندمت على شي بعثه البوت: اكتب /del لحاله — بتمسح آخر رسالة بعثها البوت "
        "بهالمحادثة — أو ترد على رسالته وتحط /del، فتحديد الي تريد تمسحها."
    )


@allowed_chat_only
async def privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "الخصوصية: ما نخزّن Telegram IDs أو أسماء المستخدمين أو usernames ضمن dataset. "
        "الصوت يُستخدم للفهم فقط ولا يُحفظ نهائيًا؛ قد تُحفظ الترجمة كنص تعليمي. "
        "الصور التعليمية والنصوص المفيدة تدخل مراجعة داخلية، وأي محتوى عليه مؤشرات شخصية واضحة يُعلّم privacy_flag ويُستبعد من التصدير. "
        "ذاكرة المحادثة (آخر ٥ أسئلة وجواباتها) تبقى بجهاز الخادم فقط، لكل طالب على حدة، "
        "وما تنكتب بالداتابيز ولا بالنسخة الاحتياطية، وتروح لما يعيد الخادم يشتغل — وبتقدر تمسحها فوراً بـ /forget."
        + (
            "\nوبالمجموعات، إذا البوابة تشتغل: آخر ١٢ رسالة بالـ30 دقيقة بتفهمها لأجل الحكم إذا السؤال "
            "للمجموعة، وهي بجهاز الخادم فقط، ما تنكتب بالداتابيز ولا بالنسخة الاحتياطية، وبتقدر تمسحها بـ /forget."
            if router is not None else ""
        )
    )


@allowed_chat_only
async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    key = thread_key(update)
    dropped = memory.forget(*key) if key else 0
    chat = update.effective_chat
    if chat is not None and chat.type != ChatType.PRIVATE:
        # The lines the gate reads are another kind of memory about the group,
        # not about the student, and /forget is the command that says stop
        # remembering. It is not counted in `dropped` because that number counts
        # questions, and clearing the context forgets none.
        forget_group_context(chat.id)
    if dropped:
        await update.effective_message.reply_text(f"انسيت آخر {dropped} سؤال. بكرة نبدأ من جديد.")
    else:
        await update.effective_message.reply_text("ماكو شي محفوظ أنساه.")


@allowed_chat_only
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await is_paused(update.effective_chat.id):
        # Saying "the service is running" here would be the same lie as
        # promising a retry on a dead quota, so the pause is reported instead.
        await update.effective_message.reply_text(
            "البوت مسكوت بهالمحادثة حالياً. يرجع يرد أول ما أحد هنا يرسل /start.")
        return
    counts = await count_by_status(settings.database_path)
    summary = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "لا توجد إدخالات بعد"
    # Reported here because this is the one reason a name call in a group can
    # get no answer at all, and it cannot be seen from inside a message that was
    # never delivered.
    if reads_all_group_messages is False:
        reach = ("\nتحذير: تيليجرام ما يوصلني إلا المنشن والأوامر. "
                 "لازم تطفي Privacy Mode من @BotFather: /setprivacy ثم اختر البوت ثم Disable، "
                 "وبعدين أعيد تشغيل البوت. لينها، «ليو سؤال» بالمجموعة ما راح يوصلني.")
    elif reads_all_group_messages:
        reach = "\n📨 أقرأ كل رسائل المجموعات، فـ«ليو سؤال» يوصلني بدون منشن."
    elif router is not None:
        # Without every message there is nothing for the gate to read the
        # conversation out of, so this is the fault that looks like the gate
        # being broken rather than a misconfiguration.
        reach = ("\n🚦 البوابة تشتغل، بس تيليجرام ما يوصلني إلا المنشن والأوامر. "
                 "لازم تطفي Privacy Mode من @BotFather: /setprivacy ثم اختر البوت ثم Disable، "
                 "وبعدين أعيد تشغيل البوت.")
    else:
        reach = ""
    # Whether the gate is running, said outside the per-chat block: it is a
    # property of this bot, not of one group, and it is the first thing to check
    # when a group goes quiet after a deploy that added a key. A missing secret
    # looks exactly like a broken feature from inside Telegram.
    reach += ("\n🚦 البوابة: شغالة على " + settings.router_model
              if router is not None else
              "\n🚦 البوابة: مقفلة (ماكو ROUTER_API_KEY) — أرد بس إذا جات اسمي")
    # A group that goes quiet is almost always the wrong group, and the chat id
    # is the only thing that settles it. Naming the chat here means the answer
    # comes back in Telegram instead of from a log nobody has open.
    chat = update.effective_chat
    where = ""
    if chat.type != ChatType.PRIVATE:
        note = await read_routing(chat.id)
        if note:
            # The one line that says which of the four ways in this group has
            # been used. "وصلني: 0" means Telegram is not delivering anything,
            # which no amount of code reading can reveal.
            # An arrival that reached no handler is a different fault from one
            # that reached a handler and was turned away, and the gap between
            # the two figures is the only place that difference shows.
            seen = note.get("seen", 0)
            judged = sum(note.get(k, 0) for k in
                         ("name", "mention", "reply", "gate", "gate-soft-no", "no"))
            unhandled = seen - judged
            if unhandled:
                note["unhandled"] = unhandled
            else:
                note.pop("unhandled", None)
            where = (f"🧭 هالمحادثة: {getattr(chat, 'title', None) or chat.id} "
                     f"— المعرّف: {chat.id}\n"
                     f"📊 وصلني: {seen} رسالة · "
                     f"اسم: {note.get('name', 0)} · منشن: {note.get('mention', 0)} · "
                     f"رد: {note.get('reply', 0)} · "
                     f"بوابة: {note.get('gate', 0)} · "
                     f"البوابة ماگالت: {note.get('gate-soft-no', 0)}\n"
                     + (f"🚶 {unhandled} وصلت وما وصلها هاندلر أصلاً\n" if unhandled else "")
                     + (f"📤 ردت: {note.get('sent', 0)} · فشل: {note.get('failed', 0)}\n"
                        if ("sent" in note or "failed" in note) else "")
                     + (f"🗄️ {note['pause_db']}\n" if note.get("pause_db") else "")
                     + (f"⚠️ آخر خطأ: {note['err']}\n" if note.get("err") else ""))
            # The shapes of the last few arrivals settle the one question the
            # counters cannot: did the message arrive and get ignored, or did it
            # never arrive at all.
            recent = note.get("recent") or []
            if recent:
                lines = []
                for entry in reversed(recent):
                    age = max(0, int(time.time()) - entry.get("at", 0))
                    why = _WHY_WORDS.get(entry.get("why", ""), "")
                    lines.append(f"   · {entry.get('shape', '؟')} — {age} ثانية"
                                 + (f"\n     {why}" if why else ""))
                where += ("🕒 آخر ما وصلني:\n" + "\n".join(lines) + "\n")
                # When a picture is the way in, the caption is the only place the
                # name can go, and a caption that misses it fails silently. Say
                # how to reach the bot, in the same message that shows the miss.
                if any(entry.get("why") == "no" for entry in recent):
                    # Said from the live state, because with the gate on a name in
                    # the caption is no longer the only way in, and telling the
                    # class otherwise is advice for a bot that is not running.
                    where += ("💡 بالبوت شغالة تقدر تسأل عادي، وإذا ما جاوب "
                              "اكتب `ليو` أول الكابشن لحالها.\n" if router is not None else
                              "💡 بالصورة، اكتب `ليو` أول الكابشن لحالها، "
                              "والسؤال بالكتابة أو بالصورة نفسها.\n")
        else:
            where = (f"🧭 هالمحادثة: {getattr(chat, 'title', None) or chat.id} "
                     f"— المعرّف: {chat.id}\n"
                     f"📊 ما وصلني أي نص عادي بهالمحادثة بعد.\n")
    await update.effective_message.reply_text(
        f"{where}الخدمة تعمل. حالات البيانات: {summary}{reach}")


@admin_only
@allowed_chat_only
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    counts = await count_by_status(settings.database_path)
    await update.effective_message.reply_text("إحصاءات داخلية: " + (", ".join(f"{k}: {v}" for k, v in counts.items()) or "فارغة"))


@allowed_chat_only
async def lessons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = await list_lessons(settings.database_path)
    if not rows:
        await update.effective_message.reply_text("ماكو حصص مرتبة حاليًا.")
        return
    lines = [f"#{r['id']} — {r['title']} | {r['lesson_date']} {r['lesson_time']} | {r['enrolled']}/{r['capacity']} طالب | المتبقي {r['capacity'] - r['enrolled']}" for r in rows]
    await update.effective_message.reply_text("الحصص المتاحة:\n" + "\n".join(lines))


@admin_only
@allowed_chat_only
async def newlesson(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = (update.effective_message.text or "").split(" ", 1)
    values = [part.strip() for part in parts[1].split("|")] if len(parts) == 2 else []
    if len(values) not in {3, 4}:
        await update.effective_message.reply_text("الصيغة: /newlesson المادة | 2026-10-01 | 10:00 | 50")
        return
    try:
        capacity = int(values[3]) if len(values) == 4 else 50
        lesson_id = await create_lesson(settings.database_path, values[0], values[1], values[2], capacity)
        await update.effective_message.reply_text(f"تم ترتيب الحصة #{lesson_id} بسعة {capacity} طالب.")
    except Exception:
        logger.exception("Lesson creation failed")
        await update.effective_message.reply_text("ما كدرت أضيف الحصة. تأكد من الصيغة وأن السعة بين 1 و50.")


@admin_only
@allowed_chat_only
async def addstudent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.effective_message.text or ""
    command, separator, code = text.partition("|")
    raw_id = command.replace("/addstudent", "", 1).strip()
    if not separator or not raw_id.isdigit() or not code.strip():
        await update.effective_message.reply_text("الصيغة: /addstudent رقم_الحصة | رمز_الطالب")
        return
    _, message = await enroll_student(settings.database_path, int(raw_id), code)
    await update.effective_message.reply_text(message)


@admin_only
@allowed_chat_only
async def roster(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw = (update.effective_message.text or "").split(maxsplit=1)
    if len(raw) != 2 or not raw[1].isdigit():
        await update.effective_message.reply_text("الصيغة: /roster رقم_الحصة")
        return
    row = await lesson_roster(settings.database_path, int(raw[1]))
    if row is None:
        await update.effective_message.reply_text("الحصة غير موجودة.")
        return
    students = row["students"]
    listing = "\n".join(f"{i}. {code}" for i, code in enumerate(students, 1)) or "لا يوجد طلاب بعد."
    await update.effective_message.reply_text(f"حضور الحصة #{row['id']} — {row['title']} ({len(students)}/{row['capacity']}):\n{listing}")


def _origin_chat_id(message: Any) -> int | None:
    """The group a single message was forwarded from, or None.

    Telegram delivers the origin rather than a group reference, and the shape
    differs by what the source was: a forward from another group or a channel
    names the chat, while a forward from a person does not -- there is no group
    to answer in, and guessing the remembered one would post a stranger's
    question to a class that never asked.

    Both the current ``forward_origin`` and the older ``forward_from_chat`` are
    read. The newer field is what the library populates now, but the old one is
    what a payload from an older client still carries, and the whole feature
    fails silently without it.
    """
    origin = getattr(message, "forward_origin", None)
    chat = getattr(origin, "chat", None) or getattr(origin, "sender_chat", None)
    if chat is not None and getattr(chat, "type", None) in {ChatType.GROUP,
                                                            ChatType.SUPERGROUP,
                                                            ChatType.CHANNEL}:
        return getattr(chat, "id", None)
    legacy = getattr(message, "forward_from_chat", None)
    if legacy is not None and getattr(legacy, "type", None) in {ChatType.GROUP,
                                                               ChatType.SUPERGROUP,
                                                               ChatType.CHANNEL}:
        return getattr(legacy, "id", None)
    return None


def _forwarded_chat_id(message: Any) -> int | None:
    """The group to answer a forwarded question in.

    Three places carry it, and in practice the owner uses whichever Telegram
    makes easiest, so all three are read. The message itself, when the command
    was typed on the forward. The message it replies to, because replying to a
    forward is what typing a separate "/a" next to it does. And the most recent
    forward in this private chat, because the common shape is two messages --
    forward, then command -- and there is no reply link between them at all.

    Only the last of those is remembered state, and it is keyed to the private
    chat rather than used globally: the remembered group expires, and a
    remembered answer target is only ever consulted for a question the owner has
    just forwarded. A question forwarded from somewhere always goes to the place
    it came from, because the first two lookups are tried first and the fallback
    is the owner's own most recent forward.
    """
    direct = _origin_chat_id(message)
    if direct is not None:
        return direct
    replied = getattr(message, "reply_to_message", None)
    if replied is not None:
        found = _origin_chat_id(replied)
        if found is not None:
            return found
    return _remembered_forward(message)[1] if _remembered_forward(message) else None


def _remembered_forward(message: Any) -> tuple[Any, int] | None:
    """The last question forwarded into this chat, or None if there is none."""
    chat_key = id(getattr(message, "chat", None))
    _expire_forward(chat_key)
    return _last_forward.get(chat_key)


@private_only
async def answer_forwarded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer a forwarded group question in the group, not in the private chat.

    The owner sees a student stuck in the group while the group is busy, and
    forwards the question here to ask about it. Answering in private would give
    the owner the answer and leave the student -- the one who asked, and the one
    the answer is for -- still stuck, and the answer would arrive in a place
    nobody is reading.

    The forwarded origin decides where it goes, never a remembered group, so a
    question forwarded from anywhere lands in the place it was asked. Text,
    photos and documents all take the same path as answering in the group, so
    an owner's forwarded question gets the same treatment a student's would.
    """
    message = update.effective_message
    # The question can be on the command itself, on a message it replies to, or
    # on the forward the owner sent a moment ago, and the bare "/a" carries
    # nothing at all. Each falls back to the next, because which of the three the
    # owner used is a detail of how they hold their phone, not a rule they should
    # have to know.
    source = message
    chat_id = _origin_chat_id(source)
    if chat_id is None:
        replied = getattr(message, "reply_to_message", None)
        if replied is not None and _origin_chat_id(replied) is not None:
            # A reply to the forward is the instruction; the forward is the
            # question. Reading the reply's own text is what lets "سوّ لي" next to
            # a photo mean anything.
            source, chat_id = replied, _origin_chat_id(replied)
            if re.sub(r"^/(?:answer|a)\S*(?:\s+|$)", "",
                      (replied.text or replied.caption or "").strip()).strip() == "":
                source = getattr(replied, "reply_to_message", None) or source
    if chat_id is None:
        remembered = _remembered_forward(message)
        if remembered is not None:
            # The remembered message is the question, not the command, so a bare
            # "/a" answers what was forwarded rather than answering itself.
            source, chat_id = remembered
            if not _origin_chat_id(source):
                source = getattr(source, "reply_to_message", None) or source
    if chat_id is None:
        await message.reply_text(
            "ما لگيت رسالة محوّلة.\n"
            "حوّل رسالة الطالب للخاص، وبعدها مباشرة اكتب /a — بجاوبها بالمجموعة الي إجا منها.")
        return
    photo = (getattr(source, "photo", None) or [None])[-1]
    document = getattr(source, "document", None)
    body = (getattr(source, "caption", None) or getattr(source, "text", None) or "").strip()
    # The command word is not part of the question, whichever message it landed
    # on, so "/a" typed next to a forward is a command and the forwarded text
    # stays the question -- neither is sent to the model as the other.
    body = re.sub(r"^/(?:answer|a)\S*(?:\s+|$)", "", body, count=1).strip()
    if not body and photo is None and document is None:
        await message.reply_text("حوّل رسالة الطالب — نص أو صورة أو ملف — وأنا بجاوبها بالمجموعة.")
        return
    if not body and photo is None and document is None:
        await message.reply_text("حوّل رسالة الطالب — نص أو صورة أو ملف — وأنا بجاوبها بالمجموعة.")
        return
    try:
        await thinking_pause(message)
        if photo is not None:
            if photo.file_size and photo.file_size > settings.max_download_bytes:
                await message.reply_text("الصورة أكبر من الحد المسموح، ما أگدر أجيبها.")
                return
            data = await _download_bytes(context, photo.file_id, settings.max_download_bytes)
            answer = await ai.answer_image(data, "image/jpeg", body)
            remember_sent(chat_id, await context.bot.send_photo(chat_id, photo.file_id,
                                                                caption=answer))
        elif document is not None:
            limit = settings.max_document_bytes
            if document.file_size and document.file_size > limit:
                await message.reply_text("الملف أكبر من الحد المسموح، ما أگدر أجيبه.")
                return
            data = await _download_bytes(context, document.file_id, limit)
            mime = getattr(document, "mime_type", None) or "application/octet-stream"
            answer = await ai.answer_file(data, mime, body)
            remember_sent(chat_id, await context.bot.send_document(chat_id, document.file_id,
                                                                   caption=answer))
        else:
            answer = await ai.answer_text(body)
            # Split for the same reason as every other send: the limit is a
            # property of the message, not of who asked for the answer. Only the
            # last chunk is remembered, because a bare "/del" is one deletion and
            # deleting the tail of a split answer would leave its head behind.
            for chunk in _chunks(answer):
                remember_sent(chat_id, await context.bot.send_message(chat_id, chunk))
    except Forbidden:
        await message.reply_text("ما أگدر أرسل بالمجموعة — تأكد البوت لسه فيها وصلاحية إرسال.")
        return
    except GeminiQuotaError:
        await message.reply_text("ماكو حصة على Gemini. جرّب بعد شوي.")
        return
    except GeminiUnavailableError:
        await message.reply_text("كل النماذج مشغولة هسه. جرّب بعد شوي.")
        return
    except Exception as exc:
        logger.exception("Forwarded answer to %s failed", chat_id)
        await message.reply_text(f"صار خطأ: {type(exc).__name__}")
        return
    logger.info("Forwarded question answered in chat %s (%d char(s), %s)",
                chat_id, len(body),
                "photo" if photo is not None else "document" if document is not None else "text")
    # The group is remembered so the next "/a" needs no forward next to it,
    # which is what makes the two-message form -- forward, then command -- work.
    remember_forward_chat(source, chat_id)
    await message.reply_text("جاوبتها بالمجموعة.")


async def _download_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str,
                          limit: int) -> bytes:
    """Fetch a file into memory, refusing anything over the limit.

    Read into memory rather than to disk because these bytes go straight to the
    model: a forwarded question is not a lesson to keep, and the image handlers
    write to the dataset on purpose while this path deliberately does not.
    """
    telegram_file = await context.bot.get_file(file_id)
    data = bytes(await telegram_file.download_as_bytearray())
    if len(data) > limit:
        raise ValueError("forwarded file over the limit")
    return data


# The words that ask Leo about the message a student replied to. Telegram command
# names may only contain a-z0-9_ and PTB refuses to build a handler for anything
# else, so this is matched with a regex on a plain message handler instead of a
# CommandHandler. Both spellings are here because an Arabic keyboard turns one into
# the other by habit, and neither is the "right" one to expect from a student.
QUOTED_COMMANDS = ("سؤال", "اسأل", "ask")
QUOTED_COMMAND_RE = re.compile(
    r"^/(?:" + "|".join(re.escape(word) for word in QUOTED_COMMANDS) + r")(?![\wء-ي])(?:\s+|$)",
    re.UNICODE)

# The same restriction is why the owner-facing sticker command is matched here too:
# "ستيكر" and "ملصق" are the words the owner will actually type, and PTB will not
# build a CommandHandler for either of them.
STICKER_COMMAND_RE = re.compile(
    r"^/(?:sticker|ستيكر|ملصق)(?![\wء-ي])(?:\s+|$)", re.UNICODE)


@allowed_chat_only
@active_only
async def answer_quoted(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer the message this one replies to, as a question of the asker's own.

    Answering everything solved the silence and created a different problem: in a
    busy group every reply carries an answer, so a student who wants to ask about
    someone else's question has nowhere to put the question except a new message
    that repeats what is already on screen. That is also the shape that most often
    got no answer, because the new message is not addressed to anybody in
    particular.

    So this is the explicit form of the same request: reply to the message, type
    the command, and the bot answers about that message instead of about the
    words in the command. The quoted message is the subject, whatever it is -- text,
    a caption, a photo, a document or a voice note -- and whatever follows the
    command is the question about it. Both are needed: with the command alone there
    is nothing to ask about, and with the message alone there is no question.
    """
    message = update.effective_message
    # filters.COMMAND does not match an Arabic command, so context.args is empty
    # here and the question has to be read out of the message by hand.
    raw = message.text or message.caption or ""
    matched = QUOTED_COMMAND_RE.match(raw)
    asked = (raw[matched.end():] if matched else raw).strip()
    quoted = getattr(message, "reply_to_message", None)
    if quoted is None:
        await message.reply_text(
            "رد على الرسالة الي تريدني أجاوب عنها، وبعدها اكتب الأمر وسؤالك.\n"
            "مثلاً: ترد على سؤال زميلك وتكتب: /سؤال اشرح لي الفكرة بشكل أبسط.")
        return
    photo = (getattr(quoted, "photo", None) or [None])[-1]
    document = getattr(quoted, "document", None)
    voice = getattr(quoted, "voice", None)
    quoted_body = (getattr(quoted, "caption", None)
                   or getattr(quoted, "text", None) or "").strip()
    if not asked and not quoted_body and photo is None and document is None and voice is None:
        await message.reply_text("الرسالة الي ردت عليها فاضية. اكتب سؤالك ولّا ترد على رسالة فيها شي.")
        return
    try:
        await thinking_pause(message)
        if asked and quoted_body:
            question = f"{asked}\n\nهذي الرسالة الي ردّت عليها:\n{quoted_body}"
        elif asked:
            question = asked
        else:
            question = quoted_body
        if photo is not None:
            if photo.file_size and photo.file_size > settings.max_download_bytes:
                await message.reply_text("الصورة أكبر من الحد المسموح، ما أگدر أجيبها.")
                return
            data = await _download_bytes(context, photo.file_id, settings.max_download_bytes)
            answer = await ai.answer_image(data, "image/jpeg", question)
            remember_sent(update.effective_chat.id,
                          await context.bot.send_photo(update.effective_chat.id,
                                                       photo.file_id, caption=answer))
            note_outcome(update.effective_chat.id, True)
        elif document is not None:
            limit = settings.max_document_bytes
            if document.file_size and document.file_size > limit:
                await message.reply_text("الملف أكبر من الحد المسموح، ما أگدر أجيبه.")
                return
            data = await _download_bytes(context, document.file_id, limit)
            mime = getattr(document, "mime_type", None) or "application/octet-stream"
            answer = await ai.answer_file(data, mime, question)
            remember_sent(update.effective_chat.id,
                          await context.bot.send_document(update.effective_chat.id,
                                                           document.file_id, caption=answer))
            note_outcome(update.effective_chat.id, True)
        elif voice is not None:
            limit = settings.max_download_bytes
            data = await _download_bytes(context, voice.file_id, limit)
            answer = await ai.answer_voice(
                data, voice.mime_type or "audio/ogg",
                question or "شرّح هذا التسجيل بالتفصيل.")
            await reply_answer(message, answer)
        else:
            answer = await ai.answer_text(question)
            await reply_answer(message, answer)
    except Forbidden:
        await message.reply_text("ما أگدر أرسل بالمجموعة — تأكد البوت لسه فيها وصلاحية إرسال.")
        return
    except GeminiQuotaError:
        await message.reply_text("ماكو حصة على Gemini. جرّب بعد شوي.")
        return
    except GeminiUnavailableError:
        await message.reply_text("كل النماذج مشغولة هسه. جرّب بعد شوي.")
        return
    except Exception as exc:
        logger.exception("Quoted answer in %s failed", update.effective_chat.id)
        await message.reply_text(f"صار خطأ: {type(exc).__name__}")
        return
    # The student aimed the bot at this message, so it is a real question and
    # belongs in the dataset; see _addressed for why that is not automatic.
    if asked:
        await save_text(settings.database_path, question, answer, "data/raw/text")
    await maybe_send_sticker(message, answer)
    logger.info("Answered a quoted message in %s (%s)", update.effective_chat.id,
                "photo" if photo is not None else
                "document" if document is not None else
                "voice" if voice is not None else "text")


def _split_target(body: str) -> tuple[int | str | None, str]:
    """Peel an explicit group off the front of an announcement.

    A raw id comes back as an int, an @name as a string, and anything else
    leaves the message untouched -- a number at the start of a message is far
    more likely to be part of the message than a group id, so it is only
    treated as a target when it carries a group's negative id shape or an @
    """
    stripped = body.lstrip()
    if not stripped:
        return None, body
    head, _, rest = stripped.partition(" ")
    if head.startswith("@") and len(head) > 1:
        return head, rest.strip()
    # Telegram supergroup ids are negative, which is what keeps a lesson like
    # "page 12 is the assignment" from being read as a group id.
    if head.lstrip("-").isdigit() and head.startswith("-"):
        return int(head), rest.strip()
    return None, body


async def _resolve_announce_target(context: ContextTypes.DEFAULT_TYPE,
                                   body: str) -> tuple[int | None, str]:
    """Work out which group an announcement goes to, and remember it if named.

    An explicit target is stored, so the owner types the id once rather than
    prefixing every announcement -- and a target the bot cannot post to is
    reported instead of silently falling back to some other group, because an
    announcement landing in the wrong class is not recoverable by resending.
    """
    target, body = _split_target(body)
    if isinstance(target, int):
        _set_announce_group(target)
        return target, body
    if isinstance(target, str):
        try:
            chat = await context.bot.get_chat(target)
        except Exception as exc:
            logger.warning("Could not resolve %s: %s", target, exc)
            return None, body
        chat_id = getattr(chat, "id", None)
        if chat_id is None:
            return None, body
        _set_announce_group(chat_id)
        return chat_id, body
    return await load_announce_group(), body


def _set_announce_group(chat_id: int) -> None:
    global _announce_group
    if chat_id == _announce_group:
        return
    _announce_group = chat_id
    try:
        asyncio.create_task(_store_announce_group(chat_id))
    except RuntimeError:
        logger.debug("No running loop; the group will be stored on the next change")


@private_only
async def post(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Say something in the group as the bot, at the owner's request.

    The owner writes privately and it appears in the class as the bot's own
    message, which is how a teacher announces homework without a second phone
    or a screenshot. Deliberately a command rather than any private message: a
    student asking a real question in private must never be broadcast, and
    silently reposting someone's homework to 200 people is not recoverable by
    apologising.

    Telegram's forward is not used. A forward keeps a "forwarded from" header
    and points at the original message, so the group would see the owner's
    account and the announcement would be a lie about who is speaking. Sending
    it as a new message is the whole request.
    """
    message = update.effective_message
    photo = (message.photo or [None])[-1]
    raw = message.caption if photo is not None else message.text
    # The command word itself is not the announcement. "/post" with nothing
    # after it must say how to use the command rather than post the word "post"
    # to the class, so it is stripped here instead of being left to a length
    # check that would happily send it.
    body = re.sub(r"^/post(?:@\S+)?(?:\s+|$)", "", raw or "", count=1).strip()
    if not body and photo is None:
        await message.reply_text(
            "اكتب الرسالة بعد الأمر، أو صوّر شي مع تعليق.\n"
            "مثال: /post بكرةPc أول — يجيبوا كتاب Chapter 3.")
        return
    if body and len(body) > settings.max_message_chars:
        await message.reply_text("الرسالة طويلة أكثر من الحد المسموح.")
        return
    if photo is not None and photo.file_size and photo.file_size > settings.max_download_bytes:
        await message.reply_text("الصورة أكبر من الحد المسموح (20 MB).")
        return
    # The target can be named outright, and learning it by listening is only the
    # fallback. Deriving the group from "the last one that spoke" means the
    # first announcement after a fresh deploy fails, because nothing has been
    # heard yet -- so the feature is broken exactly when it is first tried.
    chat_id, body = await _resolve_announce_target(context, body)
    if chat_id is None:
        await message.reply_text(
            "ما أعرف المجموعة بعد.\n"
            "اكتب رقمها قبل الرسالة:\n"
            "  /post -1003560630016 الرسالة\n"
            "أو آخذ اسم المجموعة: /post @اسم_المجموعة الرسالة\n"
            "وأحفظها، فالمرة الجاية بروح لحالها.")
        return
    try:
        if photo is not None:
            # The file id is reposted straight back rather than downloaded and
            # re-uploaded: Telegram already has the bytes, and a round trip
            # through this container is one more thing that can fail between the
            # owner pressing the button and the class seeing the picture.
            remember_sent(chat_id, await context.bot.send_photo(chat_id, photo.file_id,
                                                                caption=body or None))
        else:
            # Split for the same reason as every other send: the length limit is
            # a property of the message, not of who asked for the answer. Only
            # the last chunk is remembered, so a "/del" after a long announcement
            # removes its tail rather than silently eating the whole thing.
            for chunk in _chunks(body):
                remember_sent(chat_id, await context.bot.send_message(chat_id, chunk))
    except Forbidden:
        # By far the likeliest failure, and the one the owner can act on: the
        # group remembered is one the bot has since been removed from, or where
        # it lost the right to speak.
        await message.reply_text("ما أگدر أرسل — البوت ما بقى يگدر يچيكي بالمجموعة.")
        return
    except BadRequest:
        await message.reply_text("ما أگدر أرسل — يمكن المجموعة انحذفت.")
        return
    except Exception as exc:
        logger.exception("Announcement to %s failed", chat_id)
        await message.reply_text(f"صار خطأ وقت الإرسال: {type(exc).__name__}")
        return
    logger.info("Announcement delivered to chat %s (%d char(s), %s)",
                chat_id, len(body), "photo" if photo is not None else "text")
    # Echoing the text back is the point: the owner confirms what went out, and
    # a typo is visible here instead of only in the group.
    await message.reply_text("انرسلت:\n\n" + body if body
                             else "انرسلت الصورة.")


@admin_only
async def delete_last(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Take back the bot's last message, so a wrong post is not a permanent one.

    Two shapes, because there are two moments. Replying to a message says which
    one: the owner can see the mistake, or cannot see it, and replying removes the
    guesswork from both. A bare "/del" is for when the owner already knows what
    went out and does not want to hunt for it -- the bot remembers what it last
    sent in that chat and deletes exactly that.

    A remembered id is only ever written by this bot's own sends, so the bare
    shape cannot be aimed at a student. The reply shape is the one that could be,
    so it checks who wrote the message being pointed at and refuses otherwise.
    Telegram would refuse it too, but with "message can't be deleted" and no
    reason given, which reads as a broken command rather than as the bot
    protecting a student.

    A delete that worked says nothing at all, and that is the whole shape of the
    command: the message is gone and no second one replaces it.
    """
    message = update.effective_message
    replied = getattr(message, "reply_to_message", None)
    bot_id = getattr(getattr(context, "bot", None), "id", None)
    if replied is not None:
        author = getattr(replied, "from_user", None)
        if bot_id is None or author is None or author.id != bot_id:
            await message.reply_text("هذي مو رسالة البوت — ما أگدر أحذفها.")
            return
        message_id = replied.message_id
    else:
        message_id = last_sent_in(getattr(getattr(message, "chat", None), "id", None))
        if message_id is None:
            await message.reply_text(
                "ما أگدر ألگى رسالة بعثتها هالمحادثة.\n"
                "أو ترد على الرسالة وتحط /del — بجيحة الرسالة الي تريد تحذفها.")
            return
    try:
        await context.bot.delete_message(message.chat.id, message_id)
    except BadRequest as exc:
        # Telegram's wording is deliberately unhelpful: a message older than 48
        # hours, one from a different chat and one that was never sent all come
        # back the same way. The detail goes to the log; the owner gets the
        # likeliest cause, because a command that says "it failed" and nothing
        # else is not actionable.
        logger.warning("Delete of %s in %s refused: %s", message_id, message.chat.id, exc)
        await message.reply_text("ما أگدر أحذفها — يمكن صارت أقدم من ٤٨ ساعة.")
        return
    except Forbidden:
        await message.reply_text("ما أگدر أحذف — البوت ما بقى يگدر يحذف بهالمجموعة.")
        return
    except Exception as exc:
        logger.exception("Delete of %s in %s failed", message_id, message.chat.id)
        await message.reply_text(f"صار خطأ: {type(exc).__name__}")
        return
    # The message the command pointed at is gone, so it must not stay in the
    # memory: a second "/del" would otherwise delete nothing.
    _last_sent.pop(message.chat.id, None)
    logger.info("Deleted message %s in chat %s", message_id, message.chat.id)
    # Nothing is written back. The owner asked for a message to be taken back,
    # and a confirmation of that is a second message where the mistake was: in a
    # class group a bot that says "انحذفت" leaves a trace of the wrong post that
    # has just been erased, and the whole point of the command is that the post
    # was not a permanent one. Silence is the confirmation. Failures still speak,
    # because a delete that did not happen is the one case the owner has to hear
    # about, and silence there would be a lie.


@admin_only
@allowed_chat_only
async def pending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = await pending_rows(settings.database_path)
    if not rows:
        await update.effective_message.reply_text("ماكو عناصر بانتظار المراجعة.")
        return
    lines = [f"#{r['id']} {r['type']} lang={r['language']} dialect={r['dialect']} "
             f"privacy={bool(r['privacy_flag'])} status={r['status']}" for r in rows]
    await update.effective_message.reply_text("العناصر غير المعتمدة:\n" + "\n".join(lines))


@admin_only
@allowed_chat_only
@private_only
async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Move a collected item to approved, or to rejected, by id.

    Private chat only, so a decision about what leaves the dataset is never
    taken in front of the class that sent it in.
    """
    message = update.effective_message
    parts = (update.effective_message.text or "").split()
    if len(parts) != 3 or parts[2] not in {"approved", "rejected", "review"}:
        await message.reply_text(
            "الصيغة: /approve رقم_العنصر approved | rejected | review\n"
            "مثال: /approve 12 approved")
        return
    raw_id, wanted = parts[1], parts[2]
    if not raw_id.isdigit():
        await message.reply_text("رقم العنصر لازم يكون رقم.")
        return
    changed = await set_contribution_status(settings.database_path, int(raw_id), wanted)
    if not changed:
        await message.reply_text(f"ماكو عنصر بالمعرّف {raw_id}.")
        return
    counts = await count_by_status(settings.database_path)
    if wanted == "approved":
        flagged = await flagged_count()
        blocked = (f" تنبيه: فيه {flagged} عنصر عليه علم خصوصية، "
                   "وما يطلعون بالتصدير حتى لو يعتمدونه.") if flagged else ""
        await message.reply_text(
            f"اعتمدت العنصر #{raw_id}. الحالات: " + _status_summary(counts) + blocked)
    elif wanted == "rejected":
        await message.reply_text(
            f"رفضت العنصر #{raw_id}. الحالات: " + _status_summary(counts))
    else:
        await message.reply_text(
            f"رجّعت العنصر #{raw_id} للمراجعة. الحالات: " + _status_summary(counts))


async def flagged_count() -> int:
    rows = await all_rows(settings.database_path)
    return sum(1 for row in rows if row.get("privacy_flag"))


def _status_summary(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "فارغة"


@admin_only
@allowed_chat_only
async def export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the dataset as JSONL. `all` includes unreviewed and flagged rows."""
    message = update.effective_message
    everything = "all" in (update.effective_message.text or "").split()
    if everything and message.chat.type != ChatType.PRIVATE:
        # Unfiltered output can contain whatever a student typed, including
        # personal details, so it is kept out of the group.
        await message.reply_text("التصدير الكامل بالخاص فقط. أرسل /export all مباشرة مع البوت.")
        return
    output = Path("data/exports") / f"dataset_{uuid.uuid4().hex[:10]}.jsonl"
    path, count = await export_dataset(settings.database_path, str(output),
                                        everything=everything)
    if not count:
        # Sending an empty file would look like a successful export.
        await message.reply_text(
            "ماكو عناصر للتصدير. اعتمد العناصر أولاً بـ /approve، "
            "أو استعمل /export all للكل.")
        return
    caption = (f"تصدير كامل: {count} عنصر (يشمل غير المعتمد والعلم عليهم).\n"
               "احتفظ به لنفسك، ما تنشره." if everything
               else f"تم تصدير {count} عنصر معتمد وآمن.")
    with open(path, "rb") as handle:
        await message.reply_document(document=handle, caption=caption)


async def build_backup_archive(stamp: str) -> tuple[Path, dict[str, Any]]:
    """Zip the database and the raw images into data/backups/<stamp>.zip.

    The database copy is a real SQLite file even when the data lives in a remote
    database, so the archive is always something the owner can open or restore.
    """
    workdir = Path("data/backups") / stamp
    db_copy = workdir / "edu_bot.db"
    await snapshot_database(settings.database_path, str(db_copy))
    images = sorted(p for p in Path("data/raw/images").glob("*") if p.is_file())
    for image in images:
        target = workdir / "images" / image.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "database": db_copy.name,
        "counts": await count_by_status(settings.database_path),
        "images": [p.name for p in images],
    }
    (workdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = Path(shutil.make_archive(str(workdir), "zip", root_dir=workdir))
    shutil.rmtree(workdir, ignore_errors=True)
    return archive, manifest


def drop_archive(archive: Path) -> None:
    """Remove the temp zip; the caller has already sent or discarded it."""
    archive.unlink(missing_ok=True)
    for leftover in archive.parent.glob(f"{archive.stem}*"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)


async def private_consent_answered(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Record the answer to the confirmation request and act on it.

    The id travels inside the callback data rather than being read off the chat,
    so a tap cannot confirm on somebody else's behalf, and the presser still has
    to be the person the request was addressed to.
    """
    query = update.callback_query
    if query is None:
        return
    decision, _, raw_id = (query.data or "").partition(":")
    try:
        asked_user = int(raw_id)
    except ValueError:
        await query.answer(CONSENT_NOT_YOURS, show_alert=True)
        return
    if not query.from_user or query.from_user.id != asked_user:
        await query.answer(CONSENT_NOT_YOURS, show_alert=True)
        return
    if decision == CONSENT_YES:
        await set_private_consent(asked_user, True)
        await query.answer("تم التأكيد ✅")
        await query.edit_message_text(CONSENT_ACCEPTED)
    else:
        await set_private_consent(asked_user, False)
        await query.answer("تم الإلغاء")
        await query.edit_message_text(CONSENT_DECLINED)


async def revoke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Take back a private confirmation, so the bot asks that person again.

    Without this one mistaken tap on تأكيد would be permanent, and the owner
    would have no way to close a private chat they did not want open.
    """
    if not is_admin(update):
        return
    message = update.effective_message
    parts = (message.text or "").split()
    if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
        await message.reply_text("الصيغة: /revoke رقم_المستخدم")
        return
    target = int(parts[1])
    if target == settings.admin_id:
        await message.reply_text("هذا أنت، وما تحتاج.")
        return
    await set_private_consent(target, False)
    logger.info("Owner revoked the private confirmation of user %s", target)
    await message.reply_text(
        f"رفت التأكيد عن {target}. أول ما يرسل /start راح ينطلب تأكيد جديد.")


@admin_only
@allowed_chat_only
@private_only
async def backup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Zip the database and raw images and send them to the admin.

    Private chat only, so the archive never lands in front of the whole group.
    """
    message = update.effective_message
    await message.reply_text("جاري تجهيز النسخة الاحتياطية...")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive: Path | None = None
    try:
        archive, manifest = await build_backup_archive(stamp)
        total = sum(manifest["counts"].values())
        with open(archive, "rb") as handle:
            await message.reply_document(
                document=handle,
                caption=(f"نسخة احتياطية {stamp}\n"
                         f"العناصر: {total} | الصور: {len(manifest['images'])}\n"
                         f"احتفظ بهذا الملف على لابتوبك."))
    except Exception:
        logger.exception("Backup failed")
        await message.reply_text("فشلت النسخة الاحتياطية. شوف اللوق.")
    finally:
        if archive is not None:
            drop_archive(archive)


@allowed_chat_only
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("تم الإلغاء. دز سؤال جديد بأي وقت.")


@allowed_chat_only
@active_only
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer from a file: PDF, photo, GIF or short MP4.

    The bytes are held in memory and never written to disk: a document is read
    once and answered, it is not a dataset item, so there is no file for a later
    restart to lose. Only the question and the answer are kept, as text.
    """
    if not await should_answer(update, context):
        return
    message = update.effective_message
    document = message.document
    mime = (document.mime_type or "").lower()
    if mime not in DOCUMENT_MIMES:
        # Said plainly rather than staying silent, so a student who sent the
        # wrong file learns why nothing came back.
        await message.reply_text(UNSUPPORTED_DOCUMENT_MESSAGE)
        return
    is_video = mime in VIDEO_MIMES
    limit = settings.max_video_bytes if is_video else settings.max_document_bytes
    if document.file_size and document.file_size > limit:
        megabytes = limit // (1024 * 1024)
        if is_video:
            await message.reply_text(
                f"المقطع أكبر من الحد المسموح ({megabytes} MB). "
                "المقاطع الكبيرة تاكل حصة الخدمة، فاقصّر الفيديو قبل ما ترسله.")
        else:
            await message.reply_text(
                f"الملف أكبر من الحد المسموح ({megabytes} MB). "
                "صغّره أو أرسل صفحاته كصور بدال الملف كله.")
        return
    caption = message.caption or ""
    question = clean_prompt(caption, context.bot.username or "") or "اشرحلي الملف هذا."
    key = thread_key(update)
    # A worksheet posted with no caption is answered but not kept; see _addressed.
    keep = _addressed(update, context)
    try:
        telegram_file = await context.bot.get_file(document.file_id)
        data = bytes(await telegram_file.download_as_bytearray())
        # The wait goes after the download, so a slow or failed transfer does
        # not spend the student's reading time before anything is even read.
        await thinking_pause(message)
        history = memory.recent(*key) if key else []
        answer = await ai.answer_file(data, mime, caption, history=history)
        if key:
            memory.remember(*key, question, answer)
        # Stored as text: the PII scan on the text is what keeps a document
        # full of names and phone numbers out of any export.
        if keep:
            await save_text(settings.database_path, question, answer, "data/raw/documents")
        await reply_answer(message, answer)
    except GeminiQuotaError:
        logger.error("No quota for any model while reading a %s", mime)
        await message.reply_text(QUOTA_MESSAGE)
    except GeminiUnavailableError:
        logger.warning("All Gemini models are busy while reading a %s", mime)
        await message.reply_text(BUSY_MESSAGE)
    except Exception:
        logger.exception("Document processing failed")
        await message.reply_text("ما كدرت أقرأ الملف هذا. تأكد إنه PDF أو صورة واضحة.")


@allowed_chat_only
@active_only
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await should_answer(update, context):
        return
    message = update.effective_message
    chat_id = update.effective_chat.id
    raw = message.text or ""
    text = clean_prompt(raw, context.bot.username or "")
    if not text:
        # Somebody wrote just "ليو", or just the @handle with nothing after it.
        # Answering nothing reads as a broken bot, so the wake-up call is
        # acknowledged the way /start is. Reaching here already means the
        # message was directed at the bot, so no name check is needed: that
        # check is what used to leave a bare "@handle" in silence, because a
        # handle is not the word "ليو" and never matched the name pattern.
        if raw.strip():
            await reply_answer(message, WAKE_UP_MESSAGE)
        return
    key = thread_key(update)
    # Answering everything does not mean archiving everything; see _addressed.
    keep = _addressed(update, context)
    try:
        history = memory.recent(*key) if key else []
        await thinking_pause(message)
        answer = await ai.answer_text(text, history=history)
        if key:
            memory.remember(*key, text, answer)
        if keep:
            await save_text(settings.database_path, text, answer, "data/raw/text")
        await reply_answer(message, answer)
        await maybe_send_sticker(message, answer)
    except GeminiQuotaError:
        logger.error("No quota for any model; the account needs billing")
        note_outcome(chat_id, False, "no Gemini quota")
        await message.reply_text(QUOTA_MESSAGE)
    except GeminiUnavailableError:
        logger.warning("All Gemini models are busy; answering %s later", settings.gemini_model)
        note_outcome(chat_id, False, "every model busy")
        await message.reply_text(BUSY_MESSAGE)
    except Exception as exc:
        logger.exception("Text processing failed")
        note_outcome(chat_id, False, f"{type(exc).__name__}: {exc}")
        await message.reply_text("صار خلل مؤقت بالخدمة. حاول بعد شوي.")


@allowed_chat_only
@active_only
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await should_answer(update, context):
        return
    message = update.effective_message
    photo = message.photo[-1]
    if photo.file_size and photo.file_size > settings.max_download_bytes:
        await message.reply_text("الصورة أكبر من الحد المسموح (20 MB).")
        return
    image_dir = Path("data/raw/images")
    image_dir.mkdir(parents=True, exist_ok=True)
    local_path = image_dir / f"{uuid.uuid4().hex}.jpg"
    key = thread_key(update)
    # A photo posted with no caption is answered but not kept; see _addressed.
    keep = _addressed(update, context)
    try:
        telegram_file = await context.bot.get_file(photo.file_id)
        await telegram_file.download_to_drive(custom_path=str(local_path))
        data = local_path.read_bytes()
        # A photo with no caption still continues the topic, so the turn is
        # remembered as a picture rather than skipped.
        asked = clean_prompt(message.caption or "", context.bot.username or "") or "[صورة]"
        history = memory.recent(*key) if key else []
        await thinking_pause(message)
        answer = await ai.answer_image(data, "image/jpeg", message.caption or "",
                                       history=history)
        if key:
            memory.remember(*key, asked, answer)
        if keep:
            # The PII scan is the reason an image can be kept at all, so it is
            # only worth the call when the image is going to be kept.
            visual_pii = await ai.image_has_obvious_pii(data, "image/jpeg")
            saved = await save_image(settings.database_path, str(local_path),
                                     message.caption or "", answer,
                                     dedupe_key=hashlib.sha256(data).hexdigest())
            if not saved:
                # Identical content is already on file; drop the redundant download.
                local_path.unlink(missing_ok=True)
            elif visual_pii:
                # Keep the image available for internal review, but prevent export.
                await set_privacy_flag(settings.database_path, str(local_path))
        else:
            # Not addressed, so the answer is not an example and the picture is
            # not part of the record. It had to be downloaded to be read, and the
            # copy on disk is deleted rather than left behind for a later export
            # to find -- the same picture would otherwise sit in data/raw/images
            # with no row pointing at it.
            saved = False
            local_path.unlink(missing_ok=True)
        if saved and database_is_remote:
            # The disk is wiped on every restart, so on a remote database the
            # bytes have to live in the database too, or the stored row ends up
            # pointing at a file that no longer exists. This is the case
            # materialize_images puts back on the next boot.
            try:
                await store_image(settings.database_path, local_path.name, data)
            except Exception:
                logger.exception("Could not store the image in the database")
        await reply_answer(message, answer)
    except GeminiQuotaError:
        logger.error("No quota for any model while handling a photo")
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        await message.reply_text(QUOTA_MESSAGE)
    except GeminiUnavailableError:
        logger.warning("All Gemini models are busy while handling a photo")
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        await message.reply_text(BUSY_MESSAGE)
    except Exception:
        logger.exception("Image processing failed")
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        await message.reply_text("ما كدرت أقرأ الصورة حاليًا. جرّب صورة أوضح.")


@allowed_chat_only
@active_only
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await should_answer(update, context):
        return
    message = update.effective_message
    voice = message.voice
    if voice.file_size and voice.file_size > settings.max_download_bytes:
        await message.reply_text("الرسالة الصوتية أكبر من الحد المسموح.")
        return
    audio_bytes = b""
    key = thread_key(update)
    # A voice note sent to nobody in particular is answered but not kept; see
    # _addressed.
    keep = _addressed(update, context)
    try:
        telegram_file = await context.bot.get_file(voice.file_id)
        audio_bytes = await telegram_file.download_as_bytearray()
        history = memory.recent(*key) if key else []
        await thinking_pause(message)
        transcription, answer = await ai.answer_voice(bytes(audio_bytes),
                                                      voice.mime_type or "audio/ogg",
                                                      history=history)
        if key:
            memory.remember(*key, transcription or "[رسالة صوتية]", answer)
        if keep:
            await save_voice_transcription(settings.database_path, transcription, answer)
        await reply_answer(message, answer)
    except GeminiQuotaError:
        logger.error("No quota for any model while handling a voice note")
        await message.reply_text(QUOTA_MESSAGE)
    except GeminiUnavailableError:
        logger.warning("All Gemini models are busy while handling a voice note")
        await message.reply_text(BUSY_MESSAGE)
    except Exception:
        logger.exception("Voice processing failed")
        await message.reply_text("ما كدرت أفهم الصوت حاليًا. جرّب مرة ثانية أو اكتب السؤال.")
    finally:
        # Audio is held only in memory and is explicitly released; no audio path is ever written.
        audio_bytes = b""


SEED_DIR = Path("seed")


def restore_from_seed() -> bool:
    """Re-seed an empty data dir from the seed/ folder kept in the repo.

    Render starts every deploy on a blank disk. Dropping the latest backup into
    seed/ before pushing means the server comes back with the real data instead
    of an empty database.
    """
    db_target = Path(settings.database_path)
    if db_target.exists() and db_target.stat().st_size > 0:
        return False
    seed_db = SEED_DIR / db_target.name
    if not seed_db.is_file():
        return False
    db_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_db, db_target)
    seed_images = SEED_DIR / "images"
    if seed_images.is_dir():
        for image in seed_images.glob("*"):
            if image.is_file():
                target = Path("data/raw/images") / image.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target)
    logger.warning("Restored database and images from seed/ after a fresh start")
    return True


async def materialize_images() -> int:
    """Put the stored image bytes back on disk after a fresh start.

    On Render the container always starts blank while the database keeps the
    rows, so without this every image record would point at a file that is gone.
    """
    if not is_remote(settings.database_path):
        return 0
    target_dir = Path("data/raw/images")
    target_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    for name in await image_names(settings.database_path):
        if (target_dir / name).is_file():
            continue
        content = await load_image(settings.database_path, name)
        if not content:
            continue
        (target_dir / name).write_bytes(content)
        restored += 1
    return restored


def start_health_server() -> int | None:
    """Serve a tiny HTTP endpoint so Render can route to this service.

    A Telegram bot never accepts an inbound request, so without a bound port
    Render would treat the service as having failed to start. Set PORT and this
    listens on it; locally PORT is absent and nothing is opened.
    """
    raw_port = os.getenv("PORT", "").strip()
    if not raw_port:
        return None
    port = int(raw_port)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - the name is fixed by the base class
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    logger.info("Health endpoint listening on port %d", port)
    return port


async def keep_awake(port: int, interval: int = 240) -> None:
    """Ping our own health endpoint so the instance is never treated as idle.

    The target has to be the public URL, not 127.0.0.1. Render spins a Free web
    service down after 15 minutes without *inbound* traffic, and a request that
    starts inside the container never reaches the edge that does the counting,
    so a loopback ping keeps the process alive and the service dead. Polling
    Telegram does not help either: that is outbound. RENDER_EXTERNAL_URL is set
    by Render for web services and is the only self-addressable way to produce
    the inbound request we need. UptimeRobot pointed at the same URL is a useful
    second pair of eyes, since a crash cannot stop an outside pinger.
    """
    public = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    target = f"{public}/health" if public else f"http://127.0.0.1:{port}/health"
    logger.info("Keep-alive will ping %s every %d second(s)", target, interval)
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.get(target)
        except Exception:
            logger.debug("self ping failed", exc_info=True)
        await asyncio.sleep(interval)


async def auto_backup_loop(telegram_bot: Any, minutes: int) -> None:
    """Push a fresh archive to the owner whenever new data has landed.

    The database survives a restart on its own, but the laptop copy is the one
    the owner actually keeps, so it is refreshed without being asked.
    """
    sent_at: int | None = None
    while True:
        await asyncio.sleep(minutes * 60)
        try:
            total = sum((await count_by_status(settings.database_path)).values())
            if total == 0 or total == sent_at:
                continue
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive, manifest = await build_backup_archive(stamp)
            try:
                with open(archive, "rb") as handle:
                    await telegram_bot.send_document(
                        chat_id=settings.admin_id, document=handle,
                        caption=(f"نسخة احتياطية تلقائية {stamp}\n"
                                 f"العناصر: {total} | الصور: {len(manifest['images'])}"))
                sent_at = total
            finally:
                drop_archive(archive)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Automatic backup failed")


# The exam topics offered as buttons. The label is short enough for a phone
# screen; the topic is what actually goes to the model, so it can be more
# precise than the button says. Kept to eight so the menu stays two columns.
QUIZ_TOPICS: tuple[tuple[str, str], ...] = (
    ("الذكاء الاصطناعي", "أسس الذكاء الاصطناعي: تعريفه، فروعه، تطبيقاته، وأهميته"),
    ("تعلم الآلة", "تعلم الآلة: أنواع التعلم، التدريب، التقييم، والتحسين"),
    ("التعلم العميق", "التعلم العميق: الشبكات العصبية، الطبقات، والتدريب"),
    ("معالجة اللغة", "معالجة اللغة الطبيعية: التوكنيز، تمثيل النص، والمهام اللغوية"),
    ("الرؤية الحاسوبية", "الرؤية الحاسوبية: الصور، كشف الأجسام، والتعرف على الأنماط"),
    ("الذكاء التوليدي", "الذكاء الاصطناعي التوليدي: النماذج التوليدية، النصوص والصور"),
    ("البيانات الضخمة", "البيانات الضخمة: التخزين، التحليل، وأدوات المعالجة"),
    ("أخلاقيات الذكاء", "أخلاقيات الذكاء الاصطناعي: الخصوصية، التحيّز، والمسؤولية"),
)
QUIZ_LETTERS = ("أ", "ب", "ج", "د")


def quiz_menu() -> InlineKeyboardMarkup:
    """A button per exam topic, two to a row."""
    buttons = [InlineKeyboardButton(label, callback_data=f"qtopic:{index}")
               for index, (label, _topic) in enumerate(QUIZ_TOPICS)]
    return InlineKeyboardMarkup([buttons[i:i + 2] for i in range(0, len(buttons), 2)])


def question_keyboard(session_id: int, index: int) -> InlineKeyboardMarkup:
    """One row of answer buttons, one letter each.

    The options are written in the message rather than the button, because a
    long option would either be cut off or push the whole row off screen.
    """
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(letter, callback_data=f"qa:{session_id}:{index}:{letter}")
        for letter in QUIZ_LETTERS]])


def render_question(topic: str, questions: list[dict], index: int) -> str:
    """The question and its options as the student sees them."""
    item = questions[index]
    lines = [f"📝 {item['question']}", ""]
    for letter, option in zip(QUIZ_LETTERS, item["options"]):
        lines.append(f"{letter}) {option}")
    lines.append("")
    lines.append(f"الموضوع: {topic}  •  سؤال {index + 1} من {len(questions)}")
    return "\n".join(lines)


async def start_quiz(message, topic: str, user) -> None:
    """Generate an exam on a topic and ask the first question.

    The student is passed in rather than read off the message: when the exam
    starts from a button the message is the bot's own, so its author would
    otherwise be the bot and the student would own nothing they could answer.
    """
    status = await message.reply_text(f"🧠 أكتّب اختبار على «{topic}»...")
    try:
        questions = await ai.make_quiz(topic)
    except GeminiQuotaError:
        logger.error("No quota for any model while writing a quiz")
        await message.reply_text(QUOTA_MESSAGE)
        return
    except GeminiUnavailableError:
        logger.warning("All Gemini models are busy while writing a quiz")
        await message.reply_text(BUSY_MESSAGE)
        return
    except Exception:
        logger.exception("Quiz generation failed")
        await message.reply_text("ما كدرت أكتب اختبار هلح. جرّب بعد شوية.")
        return
    if not questions:
        # An unparsable answer is a model problem, not the student's fault, so
        # it is not worth charging them a retry.
        logger.warning("Quiz generator returned nothing usable for topic %r", topic)
        await message.reply_text("ما كدرت أجمع اختبار مفيد على هذا الموضوع. "
                                 "جرّب تكتب الموضوع بشكل أوضح.")
        return

    session_id = await create_quiz_session(
        settings.database_path, chat_id=message.chat.id, user_id=user.id, topic=topic,
        questions=json.dumps(questions, ensure_ascii=False), total=len(questions))
    await status.edit_text(f"🧠 اختبار «{topic}» — {len(questions)} أسئلة. يلاّ نبدأ!")
    await message.reply_text(render_question(topic, questions, 0),
                             reply_markup=question_keyboard(session_id, 0))


@allowed_chat_only
@active_only
async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start an exam, either on a topic the student types or from the menu."""
    message = update.effective_message
    args = context.args or []
    if not args:
        await message.reply_text("شنو الموضوع اللي تريد اختبار؟ "
                                 "اكتبه بعد الأمر، أو اختار من القائمة:",
                                 reply_markup=quiz_menu())
        return
    topic = " ".join(args).strip()[:120]
    if not topic:
        await message.reply_text("اكتب اسم الموضوع بعد الأمر، مثل: /quiz تعلم الآلة")
        return
    await start_quiz(message, topic, message.from_user)


@active_only
async def quiz_topic_picked(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    message = query.message
    # No admin check here: the buttons belong to the person who ran /quiz, and
    # private is open to any confirmed student. allowed_chat_only has already
    # settled who is allowed to be talking to the bot at all.
    try:
        index = int(query.data.split(":", 1)[1])
        _label, topic = QUIZ_TOPICS[index]
    except (IndexError, ValueError):
        logger.warning("Bad quiz topic callback: %r", query.data)
        await message.reply_text("ما فهمت أي موضوع. أرسل /quiz وحيدر اختيارات.")
        return
    # The exam belongs to whoever tapped the button, not to the bot that
    # posted the menu, so the student is the one answering their own exam.
    await start_quiz(message, topic, query.from_user)


@active_only
async def quiz_answered(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Grade one answer, explain it, and move to the next question."""
    query = update.callback_query
    try:
        _prefix, session_raw, index_raw, letter = query.data.split(":")
        session_id, index = int(session_raw), int(index_raw)
    except (ValueError, AttributeError):
        await query.answer()
        return
    if letter not in QUIZ_LETTERS:
        await query.answer()
        return
    message = query.message
    session = await get_quiz_session(settings.database_path, session_id)
    if session is None:
        await query.answer("انتهى هذا الاختبار، أرسل /quiz لبداية جديد.", show_alert=True)
        return
    # Only the student who started the exam may answer it, and only in the chat
    # it was started in, so a busy group cannot answer on someone else's behalf.
    if query.from_user.id != session["user_id"] or message.chat.id != session["chat_id"]:
        await query.answer("هذا الاختبار مو إلك.", show_alert=True)
        return
    try:
        questions = json.loads(session["questions"])
    except json.JSONDecodeError:
        await query.answer("ما أگدر أقرأ الاختبار. أرسل /quiz من جديد.", show_alert=True)
        return
    if not 0 <= index < len(questions):
        await query.answer("هذا السؤال ما موجود.", show_alert=True)
        return
    # A replayed tap must not be graded twice, so the banked count is checked
    # before anything is written.
    if index < session["answered"]:
        await query.answer("هالجواب انقبل خلّص.", show_alert=True)
        return

    item = questions[index]
    chosen = QUIZ_LETTERS.index(letter)
    correct = chosen == item["answer"]
    await query.answer("صحيح ✅" if correct else "غلط ❌")
    updated = await record_quiz_answer(settings.database_path, session_id, correct)
    verdict = "✅ إجابة صحيحة" if correct else "❌ غلط"
    if not correct:
        verdict += f" — الجواب الصحيح: {QUIZ_LETTERS[item['answer']]}) {item['options'][item['answer']]}"
    body = f"{verdict}\n\n{item['explanation']}"
    try:
        await query.edit_message_text(body, reply_markup=None)
    except TelegramError:
        # A repeated tap on the same answer changes nothing, so the message is
        # already correct and there is nothing left to do.
        logger.debug("Quiz message already graded", exc_info=True)

    if updated and updated["answered"] >= updated["total"]:
        await finish_quiz(message, session["topic"], updated["score"], updated["total"])
    elif index + 1 < len(questions):
        await message.reply_text(render_question(session["topic"], questions, index + 1),
                                 reply_markup=question_keyboard(session_id, index + 1))


async def finish_quiz(message, topic: str, score: int, total: int) -> None:
    """Report the result once every answer is in."""
    percent = round(100 * score / total) if total else 0
    if percent >= 80:
        comment = "ممتاز 👏 ما شاء الله، فاهم الموضوع زين."
    elif percent >= 50:
        comment = "جيّد 👍 شوية تركيز وراح تكوّن أحسن."
    else:
        comment = "تحتاج مراجعة 💪 راجع الموضوع وگرّب مرة ثانية."
    await message.reply_text(
        f"🏁 انتهى اختبار «{topic}»\n\n"
        f"النتيجة: {score} من {total}  ({percent}%)\n{comment}\n\n"
        "شنو تريد؟ /quiz لاختبار جديد، /progress لنتائجك كلها.")


@allowed_chat_only
@active_only
async def progress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show this student's own exam history and totals."""
    message = update.effective_message
    user = message.from_user
    totals = await quiz_overall(settings.database_path, message.chat.id, user.id)
    if not totals["quizzes"]:
        await message.reply_text("ما أچت أي اختبار بعد. أرسل /quiz وابدأ أول اختبار إلك.")
        return
    history = await quiz_history(settings.database_path, message.chat.id, user.id, limit=8)
    percent = round(100 * totals["correct"] / totals["answered"]) if totals["answered"] else 0
    lines = ["📊 نتائجك", "",
             f"الاختبارات: {totals['quizzes']}",
             f"الأسئلة المجاوبة: {totals['answered']} من {totals['questions']}",
             f"الإجابات الصحيحة: {totals['correct']}  ({percent}%)", "", "آخر اختبار:"]
    for row in history:
        mark = "✅" if row["finished_at"] else "⏳"
        lines.append(f"{mark} {row['topic']}: {row['score']} من {row['total']}")
    await message.reply_text("\n".join(lines))


async def check_group_privacy_mode(bot_api) -> bool | None:
    """Look up whether Telegram delivers plain group messages to this bot.

    With privacy mode on, Telegram never sends an ordinary message that does not
    mention the bot, so a student writing "ليو" gets no answer at all and there
    is nothing in the bot's own logs to explain why. The Bot API reports the
    setting, so the question is asked once here instead of being left to guess
    work. A failure to ask is not fatal and reports None.
    """
    global reads_all_group_messages
    try:
        me = await bot_api.get_me()
    except Exception:
        logger.warning("Could not read the bot's privacy setting", exc_info=True)
        reads_all_group_messages = None
        return None
    reads_all_group_messages = bool(getattr(me, "can_read_all_group_messages", None))
    # The username belongs in the log on purpose: when a group goes quiet, the
    # first question is which bot answered, and a stale token in the platform's
    # settings starts a second, different bot on the same service.
    logger.info("Running as @%s", getattr(me, "username", None))
    if reads_all_group_messages is False:
        logger.warning(
            "Telegram privacy mode is ON: plain group messages never reach this "
            "bot, so writing 'ليو' in a group will get no answer. Fix it in "
            "BotFather: /setprivacy -> pick the bot -> Disable, then restart the "
            "bot so the new setting applies.")
    else:
        logger.info("Telegram privacy mode is off, so name calls in a group will be answered")
    return reads_all_group_messages


# The loops that outlive a single update are owned here rather than by
# Application.create_task(). post_init runs while the application is still
# booting, and PTB warns there that such tasks are never awaited and never
# cancelled: the container is killed mid-loop and every pending task is
# reported as "Task was destroyed but it is pending", which is indistinguishable
# from a bot that has stopped polling. Holding the handles lets post_shutdown
# end them in order instead of dropping them on the floor.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def spawn_background(coroutine: Any) -> asyncio.Task[Any]:
    """Run a long-lived coroutine on PTB's loop and remember it for shutdown."""
    task = asyncio.get_running_loop().create_task(coroutine)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


async def stop_background(_: Application) -> None:
    """Cancel and await every loop we started, so none is destroyed pending."""
    tasks = list(_BACKGROUND_TASKS)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _BACKGROUND_TASKS.clear()


async def post_init(application: Application) -> None:
    global database_is_remote
    for directory in ("data/raw/images", "data/raw/text", "data/raw/documents", "data/exports",
                      "data/backups"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    database_is_remote = is_remote(settings.database_path)
    restored = False if database_is_remote else restore_from_seed()
    await init_db(settings.database_path)
    images_back = await materialize_images()
    logger.info(
        "Database initialized%s%s",
        " (restored from seed/)" if restored else "",
        f" ({images_back} image(s) restored from the database)" if images_back else "",
    )
    # Recovered at startup rather than on the first /post, because a fresh
    # container has heard from nobody and would otherwise refuse an announcement
    # for the one reason that matters least: it does not know the group yet.
    await load_announce_group()
    if application is not None:
        await check_group_privacy_mode(application.bot)
        # The flush is scheduled here and not in main(): a task can only be
        # created once the loop is running, and main() runs before that.
        # Scheduling it from main() raised "no running event loop" and killed
        # the bot at startup, which is the one failure that looks exactly like
        # a bot that has gone quiet in a group.
        #
        # A group that goes quiet after its last message is the one whose tally
        # we most need, so the counters are written on a timer and not only
        # when the next message happens to arrive.
        spawn_background(routing_flush_loop())
    if health_port:
        spawn_background(keep_awake(health_port))
    minutes = int(os.getenv("AUTO_BACKUP_MINUTES", "0") or 0)
    if minutes > 0:
        spawn_background(auto_backup_loop(application.bot, minutes))
        logger.info("Automatic backup every %d minute(s) to the admin's private chat", minutes)


def build_application() -> Application:
    application = (Application.builder()
                   .token(settings.telegram_bot_token)
                   .concurrent_updates(True)
.post_init(post_init)
                    .post_shutdown(stop_background)
                    .build())
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("privacy", privacy))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("lessons", lessons))
    application.add_handler(CommandHandler("newlesson", newlesson))
    application.add_handler(CommandHandler("addstudent", addstudent))
    application.add_handler(CommandHandler("roster", roster))
    application.add_handler(CommandHandler("pending", pending))
    application.add_handler(CommandHandler("export", export))
    application.add_handler(CommandHandler("approve", approve))
    application.add_handler(CommandHandler("backup", backup))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(CommandHandler("revoke", revoke))
    application.add_handler(CommandHandler("forget", forget))
    application.add_handler(CommandHandler("pause", pause))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(CommandHandler("progress", progress))
    application.add_handler(CommandHandler("post", post))
    application.add_handler(CommandHandler("answer", answer_forwarded))
    application.add_handler(CommandHandler("a", answer_forwarded))
    application.add_handler(MessageHandler(filters.Regex(STICKER_COMMAND_RE),
                                           sticker_command))
    # Ask about the message you replied to. Registered as a plain message handler
    # on a regex, not a CommandHandler: Telegram command names are restricted to
    # a-z0-9_ and PTB refuses to build one for "سؤال", and the word students
    # actually type has to be Arabic. Several spellings of it, because an Arabic
    # keyboard makes "سؤال" and "اسأل" a different habit each. It sits above the
    # generic text handler on purpose: filters.COMMAND does not recognise these
    # words, so without this the text handler would answer "/سؤال ..." as a
    # message and the quoted question would be lost.
    application.add_handler(MessageHandler(filters.Regex(QUOTED_COMMAND_RE),
                                           answer_quoted))
    application.add_handler(CommandHandler("group", show_announce_group))
    # "del" and the full word, because both are what people try first and a
    # command that exists under one name and not the other reads as a bug.
    application.add_handler(CommandHandler("del", delete_last))
    application.add_handler(CommandHandler("delete", delete_last))
    application.add_handler(CallbackQueryHandler(quiz_topic_picked, pattern=r"^qtopic:"))
    application.add_handler(CallbackQueryHandler(quiz_answered, pattern=r"^qa:"))
    application.add_handler(CallbackQueryHandler(private_consent_answered,
                                                 pattern=r"^priv(?:ok|no):\d+$"))
    # Group -1 so it runs before every other handler: registered in the same
    # group, the text handler would always match first and the watcher would
    # never see the update it exists to witness. block=False keeps it from
    # taking the update away from whoever is meant to answer.
    application.add_handler(TypeHandler(Update, log_arriving_update, block=False),
                            group=-1)
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # A handler that falls over leaves no trace the person waiting for an answer
    # can see, and the one place they can see anything is this chat's own note.
    application.add_error_handler(note_handler_failure)
    return application


# How long to wait before starting over after the polling loop gives up.
RESTART_AFTER_SECONDS = 5.0


def main() -> None:
    """Keep the bot polling, whatever the reason it stopped.

    Losing the right to poll is not a reason to quit. Telegram gives the token
    to exactly one poller at a time, and any overlap produces a Conflict --
    which is what a deploy does by itself, since the new instance starts before
    the old one has let go. PTB's default gives that conflict zero retries and
    re-raises it, so the process stopped and Render started it again, into the
    same conflict: a deploy turned the bot off. The group then looks exactly
    like a group the bot stopped listening to, which is the wrong conclusion to
    reach and an hour of guessing.

    So polling is retried forever rather than allowed to end, and the run is
    rebuilt from scratch each time because a stopped application cannot be
    started again.
    """
    global settings, ai, router, health_port
    settings = Settings.from_env()
    ai = GeminiClient(settings.gemini_api_key, settings.gemini_model)
    # Built only when the key is there, on its own model and its own client, so
    # the traffic the gate generates cannot spend the quota the answers need.
    # Unset is not an error and not a quieter bot: every message still gets an
    # answer, this one without the gate's second opinion, and /status says the
    # gate is off so nobody goes looking for a bug that is a missing key.
    if settings.router_api_key:
        router = GeminiClient(settings.router_api_key, settings.router_model)
    else:
        router = None
    health_port = start_health_server()
    logger.info("Leo starting with model %s", settings.gemini_model)
    logger.info("Relevance gate: %s", f"on, model {settings.router_model}"
                if router is not None else "off (no ROUTER_API_KEY)")
    attempt = 0
    while True:
        application = build_application()
        try:
            # bootstrap_retries=-1 means retry every network error, forever,
            # instead of the first one ending the process.
            application.run_polling(allowed_updates=Update.ALL_TYPES,
                                    bootstrap_retries=-1)
            return
        except Conflict:
            attempt += 1
            logger.warning(
                "Another instance holds the token (attempt %d); waiting %.0fs. "
                "If this repeats, another service is polling this bot.",
                attempt, RESTART_AFTER_SECONDS)
        except Exception:
            attempt += 1
            logger.exception("Polling stopped (attempt %d); waiting %.0fs",
                             attempt, RESTART_AFTER_SECONDS)
        time.sleep(RESTART_AFTER_SECONDS)


if __name__ == "__main__":
    main()
