from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence

from google import genai
from google.genai import types

from memory import Turn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Leo, a tutor for Iraqi students.

Creator:
- You were created and built by Haider Aqeel.
- If anyone asks who made, built, developed, programmed, trained or owns you, the only correct answer is Haider Aqeel.
- Never say that Google, Gemini, OpenAI, Anthropic, Meta or any other company or person made you. Do not mention the underlying model or provider.

Rules:
- Reply in Iraqi Arabic dialect, and match the student's own language. Never convert to Modern Standard Arabic unless the student asks.
- Answer the question directly and correctly, in the fewest words that fully solve it.
- Never introduce yourself, never list your abilities, and never ask what the student wants to study. Do that only if the student explicitly asks who you are.
- Greet back briefly only when the student greets you. One short line, then stop.
- For maths, show the steps briefly and end with the final answer.
- For images, read handwritten or printed educational content and answer it. For voice, transcribe and answer, but never mention storing audio.
- Plain text only: no markdown, no asterisks, no bold, no headings, no bullet symbols.
- No personal-data commentary."""

# Appended only when earlier turns are replayed, so the model treats them as
# background for a question that is still open rather than as work to redo.
CONTINUATION_RULE = (
    "\n\nConversation: the earlier turns are the same student's recent exchanges, "
    "replayed for context only. Use them to follow the topic, do not answer them "
    "again, do not repeat a previous answer, and do not greet again unless the "
    "student greets you now."
)

# Tried in order when the primary model is temporarily unavailable, so a 503 on
# one model does not take the whole bot offline. The two lite models lead the
# chain because they hold up under a busy period, and the stronger ones follow in
# case their quota frees up again. Skipping straight to them would otherwise burn
# a retry delay per model on every single reply.
FALLBACK_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite-preview",
                   "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.6-flash")
FALLBACK_DELAY_SECONDS = 1.5
RETRYABLE_CODES = {429, 500, 502, 503, 504}
# A busy group can fire many requests at once; this keeps us inside the API's
# per-minute quota instead of letting every caller burst at the same moment.
MAX_CONCURRENT_CALLS = 8

_gate: asyncio.Semaphore | None = None
_gate_loop: asyncio.AbstractEventLoop | None = None


def _call_gate() -> asyncio.Semaphore:
    """A semaphore bound to the running loop, rebuilt if the loop changed."""
    global _gate, _gate_loop
    loop = asyncio.get_running_loop()
    if _gate is None or _gate_loop is not loop:
        _gate = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
        _gate_loop = loop
    return _gate


class GeminiUnavailableError(RuntimeError):
    """Every configured model was temporarily unavailable (capacity/quota)."""


def _strip_label(text: str, label: str) -> str:
    """Drop a leading 'LABEL:' prefix so raw scaffolding never reaches the user."""
    cleaned = re.sub(rf"^\s*{label}\s*:", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def _is_unavailable(exc: Exception) -> bool:
    """True for capacity/quota style failures that another model can dodge."""
    if getattr(exc, "code", None) in RETRYABLE_CODES:
        return True
    name = type(exc).__name__
    return name in {"ServerError", "ResourceExhausted", "ServiceUnavailable"}


class GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.models = (model,) + tuple(m for m in FALLBACK_MODELS if m != model)

    async def _complete(self, parts: list[types.Part | str],
                        system_prompt: str | None = SYSTEM_PROMPT,
                        empty_message: str = "Gemini returned an empty response",
                        allow_empty: bool = False,
                        history: Sequence[Turn] = ()) -> str:
        """Send one request, optionally preceded by the earlier turns.

        The history is replayed as real alternating turns rather than pasted
        into the prompt, so the model reads it as a conversation it is already
        in. Anything earlier than memory's own cap is simply not here.
        """
        content_parts = [p if isinstance(p, types.Part) else types.Part(text=p) for p in parts]
        if system_prompt is not None:
            if history:
                system_prompt += CONTINUATION_RULE
            content_parts.insert(0, types.Part(text=system_prompt))
        contents: list[types.Content] = []
        for question, answer in history:
            contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
            contents.append(types.Content(role="model", parts=[types.Part(text=answer)]))
        contents.append(types.Content(role="user", parts=content_parts))

        last: Exception | None = None
        gate = _call_gate()
        for index, model in enumerate(self.models):
            try:
                async with gate:
                    response = await self.client.aio.models.generate_content(
                        model=model, contents=contents)
            except Exception as exc:
                if not _is_unavailable(exc):
                    raise
                last = exc
                logger.warning("Model %s unavailable (%s: %s), trying fallback",
                               model, type(exc).__name__, _brief(exc))
                if index < len(self.models) - 1:
                    await asyncio.sleep(FALLBACK_DELAY_SECONDS)
                continue
            text = (response.text or "").strip()
            if not text:
                if allow_empty:
                    return ""
                raise RuntimeError(empty_message)
            if index:
                logger.info("Answered using fallback model %s", model)
            return text
        raise GeminiUnavailableError(
            f"All Gemini models unavailable; last error: {_brief(last)}") from last

    async def _generate(self, parts: list[types.Part | str],
                        history: Sequence[Turn] = ()) -> str:
        return await self._complete(parts, history=history)

    async def answer_text(self, text: str, history: Sequence[Turn] = ()) -> str:
        return await self._generate([text], history=history)

    async def answer_image(self, image_bytes: bytes, mime_type: str, caption: str = "",
                           history: Sequence[Turn] = ()) -> str:
        parts: list[types.Part | str] = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        if caption:
            parts.append(caption)
        return await self._generate(parts, history=history)

    async def image_has_obvious_pii(self, image_bytes: bytes, mime_type: str) -> bool:
        text = await self._complete(
            [types.Part(text="Inspect this image for obvious personal information such as a person's name, phone number, ID, address, or a recognizable face. Reply with exactly YES or NO. Do not transcribe anything."),
             types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
            system_prompt=None, allow_empty=True)
        return text.strip().upper().startswith("YES")

    async def answer_voice(self, audio_bytes: bytes, mime_type: str,
                           history: Sequence[Turn] = ()) -> tuple[str, str]:
        transcription_prompt = "Transcribe this voice message exactly, then answer the educational question. Format: TRANSCRIPTION:\\n...\\nANSWER:\\n..."
        text = await self._complete(
            [types.Part(text=transcription_prompt),
             types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)],
            empty_message="Gemini returned an empty voice response",
            history=history)
        split = re.search(r"\bANSWER\s*:", text, re.IGNORECASE)
        if split:
            transcription = _strip_label(text[: split.start()], "TRANSCRIPTION")
            answer = _strip_label(text[split.end():], "ANSWER")
        else:
            # The model ignored the requested format; never leak the raw
            # "TRANSCRIPTION:" scaffolding back to the student or the dataset.
            transcription = _strip_label(text, "TRANSCRIPTION")
            answer = transcription or text
        return transcription, answer or text


def _brief(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    return " ".join(str(exc).split())[:120]
