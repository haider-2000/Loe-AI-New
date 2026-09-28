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
from telegram import Update
from telegram.constants import ChatType
from telegram.error import RetryAfter
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import Settings
from database import (count_by_status, create_lesson, enroll_student, image_names, init_db,
                      is_remote, lesson_roster, list_lessons, load_image, pending_rows,
                      set_privacy_flag, snapshot_database, store_image)
from dataset import export_dataset, save_image, save_text, save_voice_transcription
from gemini_client import GeminiClient, GeminiUnavailableError

BUSY_MESSAGE = "الخدمة مشغولة حالياً لأن ضغط الطلبات عالية. حاول بعد دقيقة."

# A local file already survives on the laptop, so the image bytes only need a
# second home when the database lives somewhere ephemeral, like a Render container.
stored_remotely = True
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


def directed_to_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    message = update.effective_message
    if not message:
        return False
    if message.chat.type == ChatType.PRIVATE:
        return True
    bot_username = context.bot.username or ""
    mention = f"@{bot_username.lower()}" if bot_username else ""
    text = (message.text or message.caption or "").lower()
    return bool(
        (mention and mention in text)
        or (message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.id == context.bot.id)
    )


def clean_prompt(text: str, bot_username: str = "") -> str:
    """Strip the bot mention and clamp to the configured limit."""
    if bot_username:
        text = re.sub(rf"@{re.escape(bot_username)}\b", "", text, flags=re.IGNORECASE)
    return text.strip()[: settings.max_message_chars]


@allowed_chat_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "هلا بيك! آني Leo، كيف أساعدك اليوم؟ "
        "دز سؤالك بالنص، صورة، أو رسالة صوتية، ووجّهها إليّ بالمجموعة بذكر اسمي أو بالرد على رسالتي."
    )


@allowed_chat_only
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "الأوامر المتاحة:\n/start - بدء الاستخدام\n/help - المساعدة\n/privacy - الخصوصية\n/status - حالة الخدمة\n/cancel - إلغاء العملية الحالية\n"
        "بالمجموعة ما أرد على كل الرسائل؛ لازم تذكرني أو ترد على رسالتي.\n\n"
        "تنظيم الحصص (للمدير):\n/newlesson العنوان | التاريخ | الوقت | السعة\n"
        "/addstudent رقم_الحصة | رمز_الطالب\n/roster رقم_الحصة\n"
        "ولكل الطلاب: /lessons لعرض الحصص والمقاعد المتبقية."
    )


@allowed_chat_only
async def privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "الخصوصية: ما نخزّن Telegram IDs أو أسماء المستخدمين أو usernames ضمن dataset. "
        "الصوت يُستخدم للفهم فقط ولا يُحفظ نهائيًا؛ قد تُحفظ الترجمة كنص تعليمي. "
        "الصور التعليمية والنصوص المفيدة تدخل مراجعة داخلية، وأي محتوى عليه مؤشرات شخصية واضحة يُعلّم privacy_flag ويُستبعد من التصدير."
    )


@allowed_chat_only
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    counts = await count_by_status(settings.database_path)
    summary = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "لا توجد إدخالات بعد"
    await update.effective_message.reply_text(f"الخدمة تعمل. حالات البيانات: {summary}")


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
async def export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update):
        return
    output = Path("data/exports") / f"dataset_{uuid.uuid4().hex[:10]}.jsonl"
    path, count = await export_dataset(settings.database_path, str(output))
    with open(path, "rb") as handle:
        await update.effective_message.reply_document(
            document=handle, caption=f"تم تصدير {count} عنصر معتمد وآمن.")


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
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not directed_to_bot(update, context):
        return
    message = update.effective_message
    text = clean_prompt(message.text or "", context.bot.username or "")
    if not text:
        return
    try:
        answer = await ai.answer_text(text)
        await save_text(settings.database_path, text, answer, "data/raw/text")
        await reply_answer(message, answer)
    except GeminiUnavailableError:
        logger.warning("All Gemini models are busy; answering %s later", settings.gemini_model)
        await message.reply_text(BUSY_MESSAGE)
    except Exception:
        logger.exception("Text processing failed")
        await message.reply_text("صار خلل مؤقت بالخدمة. حاول بعد شوي.")


@allowed_chat_only
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
    try:
        telegram_file = await context.bot.get_file(photo.file_id)
        await telegram_file.download_to_drive(custom_path=str(local_path))
        data = local_path.read_bytes()
        answer = await ai.answer_image(data, "image/jpeg", message.caption or "")
        visual_pii = await ai.image_has_obvious_pii(data, "image/jpeg")
        saved = await save_image(settings.database_path, str(local_path), message.caption or "",
                                 answer, dedupe_key=hashlib.sha256(data).hexdigest())
        if not saved:
            # Identical content is already on file; drop the redundant download.
            local_path.unlink(missing_ok=True)
        elif visual_pii:
            # Keep the image available for internal review, but prevent export.
            await set_privacy_flag(settings.database_path, str(local_path))
        if saved and not stored_remotely:
            # The disk is wiped on every restart, so the bytes have to live in the
            # database too or the stored row ends up pointing at a missing file.
            try:
                await store_image(settings.database_path, local_path.name, data)
            except Exception:
                logger.exception("Could not store the image in the database")
        await reply_answer(message, answer)
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
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not directed_to_bot(update, context):
        return
    message = update.effective_message
    voice = message.voice
    if voice.file_size and voice.file_size > settings.max_download_bytes:
        await message.reply_text("الرسالة الصوتية أكبر من الحد المسموح.")
        return
    audio_bytes = b""
    try:
        telegram_file = await context.bot.get_file(voice.file_id)
        audio_bytes = await telegram_file.download_as_bytearray()
        transcription, answer = await ai.answer_voice(bytes(audio_bytes), voice.mime_type or "audio/ogg")
        await save_voice_transcription(settings.database_path, transcription, answer)
        await reply_answer(message, answer)
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


async def post_init(application: Application) -> None:
    global stored_remotely
    for directory in ("data/raw/images", "data/raw/text", "data/exports", "data/backups"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    stored_remotely = is_remote(settings.database_path)
    restored = False if stored_remotely else restore_from_seed()
    await init_db(settings.database_path)
    images_back = await materialize_images()
    logger.info(
        "Database initialized%s%s",
        " (restored from seed/)" if restored else "",
        f" ({images_back} image(s) restored from the database)" if images_back else "",
    )
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
    application.add_handler(CommandHandler("backup", backup))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Leo starting with model %s", settings.gemini_model)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
