from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatType
from telegram.error import RetryAfter, TelegramError
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

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
# What the bot remembers between messages, so "and the second one?" keeps
# pointing at the same topic instead of starting from nothing.
memory = ConversationMemory()


def is_admin(update: Update) -> bool:
    return bool(update.effective_user and update.effective_user.id == settings.admin_id)


def allowed_chat_only(handler):
    """Serve every group the bot is a member of.

    There is no group allowlist any more: any group or supergroup that adds the
    bot gets an answer, so a new class does not need a deployment first. Private
    chats stay reserved for the owner, which is where /backup lives and where
    the bot gets driven by hand.
    """
    @wraps(handler)
    async def guarded(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        if chat and chat.type == ChatType.PRIVATE:
            if is_admin(update):
                return await handler(update, context)
            logger.info("Ignoring private chat from non-admin %s",
                        update.effective_user.id if update.effective_user else None)
            return
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


async def reply_answer(message, text: str) -> None:
    """Send an answer, splitting it if it exceeds what Telegram accepts.

    A busy group can trip Telegram's per-group flood limit, which would silently
    drop the reply, so RetryAfter is honoured instead of being swallowed.
    """
    for chunk in _chunks(text):
        for attempt in range(3):
            try:
                await message.reply_text(chunk)
                break
            except RetryAfter as exc:
                if attempt == 2:
                    logger.error("Gave up sending after repeated rate limits")
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


def _fold(text: str) -> str:
    """Strip diacritics and unify letter variants for matching.

    Length is preserved, one character in and one out, so a name matched here
    can be cut out of the original text and leave the student's own spelling
    of the rest of the message untouched.
    """
    out = []
    for ch in text:
        if any(low <= ord(ch) <= high for low, high in _DIACRITIC_RANGES):
            continue
        out.append(_LETTER_FOLD.get(ch, ch))
    return "".join(out)


def _name_spans(text: str) -> list[tuple[int, int]]:
    """Where the bot's name appears, found in the folded text."""
    return [match.span() for match in _NAME_RE.finditer(_fold(text))]


def is_name_call(text: str) -> bool:
    """True when the message says the bot's name as a whole word.

    Students do not always bother with the @handle, so in a group a plain
    "leo" or "ليو" is enough. Word boundaries matter: without them every
    message mentioning paleontology or Cleopatra would wake the bot.
    """
    return bool(_NAME_RE.search(_fold(text)))


def directed_to_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    message = update.effective_message
    if not message:
        return False
    if message.chat.type == ChatType.PRIVATE:
        return True
    bot_username = context.bot.username or ""
    mention = f"@{bot_username.lower()}" if bot_username else ""
    text = message.text or message.caption or ""
    return bool(
        (mention and mention.lower() in text.lower())
        or is_name_call(text)
        or (message.reply_to_message and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == context.bot.id)
    )


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
        _paused[chat_id] = (await read_flag(settings.database_path,
                                           _pause_flag(chat_id))) == "1"
    return _paused[chat_id]


async def set_paused(chat_id: int, paused: bool) -> None:
    _paused[chat_id] = paused
    await write_flag(settings.database_path, _pause_flag(chat_id),
                     "1" if paused else "0")


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


@allowed_chat_only
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "الأوامر المتاحة:\n/start - بدء الاستخدام\n/help - المساعدة\n/privacy - الخصوصية\n/status - حالة الخدمة\n/quiz الموضوع - اختبار\n/progress - نتائجك\n/forget - نسيان آخر المواضيع\n/cancel - إلغاء العملية الحالية\n"
        "بالمجموعة ما أرد على كل الرسائل؛ اكتب اسمي (leo أو ليو) بأول السؤال، أو رد على رسالتي.\n\n"
        "وتكتب /quiz تطلع لك قائمة باختصاصات الذكاء الاصطناعي تختار منها، أو تكتب /quiz الموضوع مباشرة مثل /quiz تعلم الآلة.\n"
        "كل اختبار ٥ أسئلة اختيار من متعدد، تجاوب بالأزرار (أ ب ج د) وتعرف النتيجة مع شرح ليش جوابك صح أو غلط.\n"
        "النتائج تنحفظ لكل طالب لحاله وتگدر تشوفها بـ /progress.\n\n"
        "و anyone يگدر يسكّت البوت بهالمحادثة بـ/pause، ويرجع يرد أول ما أحد من هنا يرسل /start.\n\n"
        "أگدر تقرا برضو ملفات PDF و GIF والمقاطع القصيرة، دزها مثل أي سؤال.\n\n"
        "أتذكر آخر ٥ أسئلة وجواباتها منك فقط (لكل مجموعة على حدة) عشان تكمل بنفس الموضوع. "
        "هذه الذاكرة مؤقتة بالجهاز وما تنحفظ بالداتابيز، وبتقدر تمسحها بـ /forget.\n\n"
        "تنظيم الحصص (للمدير):\n/newlesson العنوان | التاريخ | الوقت | السعة\n"
        "/addstudent رقم_الحصة | رمز_الطالب\n/roster رقم_الحصة\n"
        "ولكل الطلاب: /lessons لعرض الحصص والمقاعد المتبقية.\n\n"
        "للمدير: بعد ما تراجع المحتوى بـ /pending، اعتمده بـ /approve رقم_العنصر approved بالخاص، "
        "وبعدين /export يطلع ملف التدريب."
    )


@allowed_chat_only
async def privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "الخصوصية: ما نخزّن Telegram IDs أو أسماء المستخدمين أو usernames ضمن dataset. "
        "الصوت يُستخدم للفهم فقط ولا يُحفظ نهائيًا؛ قد تُحفظ الترجمة كنص تعليمي. "
        "الصور التعليمية والنصوص المفيدة تدخل مراجعة داخلية، وأي محتوى عليه مؤشرات شخصية واضحة يُعلّم privacy_flag ويُستبعد من التصدير. "
        "ذاكرة المحادثة (آخر ٥ أسئلة وجواباتها) تبقى بجهاز الخادم فقط، لكل طالب على حدة، "
        "وما تنكتب بالداتابيز ولا بالنسخة الاحتياطية، وتروح لما يعيد الخادم يشتغل — وبتقدر تمسحها فوراً بـ /forget."
    )


@allowed_chat_only
async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    key = thread_key(update)
    dropped = memory.forget(*key) if key else 0
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
    else:
        reach = ""
    await update.effective_message.reply_text(f"الخدمة تعمل. حالات البيانات: {summary}{reach}")


@allowed_chat_only
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update):
        return
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


@allowed_chat_only
async def newlesson(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update):
        return
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


@allowed_chat_only
async def addstudent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update):
        return
    text = update.effective_message.text or ""
    command, separator, code = text.partition("|")
    raw_id = command.replace("/addstudent", "", 1).strip()
    if not separator or not raw_id.isdigit() or not code.strip():
        await update.effective_message.reply_text("الصيغة: /addstudent رقم_الحصة | رمز_الطالب")
        return
    _, message = await enroll_student(settings.database_path, int(raw_id), code)
    await update.effective_message.reply_text(message)


@allowed_chat_only
async def roster(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update):
        return
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


@allowed_chat_only
async def pending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update):
        return
    rows = await pending_rows(settings.database_path)
    if not rows:
        await update.effective_message.reply_text("ماكو عناصر بانتظار المراجعة.")
        return
    lines = [f"#{r['id']} {r['type']} lang={r['language']} dialect={r['dialect']} "
             f"privacy={bool(r['privacy_flag'])} status={r['status']}" for r in rows]
    await update.effective_message.reply_text("العناصر غير المعتمدة:\n" + "\n".join(lines))


@allowed_chat_only
@private_only
async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Move a collected item to approved, or to rejected, by id.

    Private chat only, so a decision about what leaves the dataset is never
    taken in front of the class that sent it in.
    """
    if not is_admin(update):
        return
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


@allowed_chat_only
async def export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the dataset as JSONL. `all` includes unreviewed and flagged rows."""
    if not is_admin(update):
        return
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


@allowed_chat_only
@private_only
async def backup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Zip the database and raw images and send them to the admin.

    Private chat only, so the archive never lands in front of the whole group.
    """
    if not is_admin(update):
        return
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
    if not directed_to_bot(update, context):
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
    if not directed_to_bot(update, context):
        return
    message = update.effective_message
    raw = message.text or ""
    text = clean_prompt(raw, context.bot.username or "")
    if not text:
        # Somebody said just "ليو". Answering nothing reads as a broken bot, so
        # the wake-up call is acknowledged the way /start is.
        if raw.strip() and is_name_call(raw):
            await message.reply_text(WAKE_UP_MESSAGE)
        return
    key = thread_key(update)
    try:
        history = memory.recent(*key) if key else []
        await thinking_pause(message)
        answer = await ai.answer_text(text, history=history)
        if key:
            memory.remember(*key, text, answer)
        await save_text(settings.database_path, text, answer, "data/raw/text")
        await reply_answer(message, answer)
    except GeminiQuotaError:
        logger.error("No quota for any model; the account needs billing")
        await message.reply_text(QUOTA_MESSAGE)
    except GeminiUnavailableError:
        logger.warning("All Gemini models are busy; answering %s later", settings.gemini_model)
        await message.reply_text(BUSY_MESSAGE)
    except Exception:
        logger.exception("Text processing failed")
        await message.reply_text("صار خلل مؤقت بالخدمة. حاول بعد شوي.")


@allowed_chat_only
@active_only
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not directed_to_bot(update, context):
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
        visual_pii = await ai.image_has_obvious_pii(data, "image/jpeg")
        saved = await save_image(settings.database_path, str(local_path), message.caption or "",
                                 answer, dedupe_key=hashlib.sha256(data).hexdigest())
        if not saved:
            # Identical content is already on file; drop the redundant download.
            local_path.unlink(missing_ok=True)
        elif visual_pii:
            # Keep the image available for internal review, but prevent export.
            await set_privacy_flag(settings.database_path, str(local_path))
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
    if not directed_to_bot(update, context):
        return
    message = update.effective_message
    voice = message.voice
    if voice.file_size and voice.file_size > settings.max_download_bytes:
        await message.reply_text("الرسالة الصوتية أكبر من الحد المسموح.")
        return
    audio_bytes = b""
    key = thread_key(update)
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
    if message.chat.type == ChatType.PRIVATE and not is_admin(update):
        return
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
    if reads_all_group_messages is False:
        logger.warning(
            "Telegram privacy mode is ON: plain group messages never reach this "
            "bot, so writing 'ليو' in a group will get no answer. Fix it in "
            "BotFather: /setprivacy -> pick the bot -> Disable, then restart the "
            "bot so the new setting applies.")
    else:
        logger.info("Telegram privacy mode is off, so name calls in a group will be answered")
    return reads_all_group_messages


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
    if application is not None:
        await check_group_privacy_mode(application.bot)
    if health_port:
        application.create_task(keep_awake(health_port))
    minutes = int(os.getenv("AUTO_BACKUP_MINUTES", "0") or 0)
    if minutes > 0:
        application.create_task(auto_backup_loop(application.bot, minutes))
        logger.info("Automatic backup every %d minute(s) to the admin's private chat", minutes)


def main() -> None:
    global settings, ai, health_port
    settings = Settings.from_env()
    ai = GeminiClient(settings.gemini_api_key, settings.gemini_model)
    health_port = start_health_server()
    application = (Application.builder()
                   .token(settings.telegram_bot_token)
                   .concurrent_updates(True)
                   .post_init(post_init)
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
    application.add_handler(CommandHandler("forget", forget))
    application.add_handler(CommandHandler("pause", pause))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(CommandHandler("progress", progress))
    application.add_handler(CallbackQueryHandler(quiz_topic_picked, pattern=r"^qtopic:"))
    application.add_handler(CallbackQueryHandler(quiz_answered, pattern=r"^qa:"))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Leo starting with model %s", settings.gemini_model)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
