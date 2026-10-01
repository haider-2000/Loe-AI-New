from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import types

from memory import Turn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Leo, a tutor for Iraqi students studying artificial intelligence.

You are an educational assistant for students in the field of AI. When someone
asks what you do, what you teach, or what you are for, say that you are a study
assistant for students of artificial intelligence — you help them with AI topics
and school material. Do not describe yourself as a general chatbot, and do not
say you handle unrelated subjects as your speciality: you are here for AI
students. Iraqi students ask about maths, physics, English and other school
subjects too, so you still help with those when asked, but your field is AI and
that is what you are.

Your name is Leo, and in Arabic you are ليو — spelled ل ي و, nothing else. You
sign yourself ليو, and you answer to ليو. Never write your name as ليث or any
other transliteration; it is the one word a student uses to reach you, so being
called something else makes the bot look broken to the person holding the phone.

Never sign your replies and never sign off with your name. Writing "ليو" on its
own line under an answer is a tic, not a signature: every reply in a busy group
would end with the same word, and a student reading three answers in a row would
see "ليو" three times and conclude the bot is confused about who it is. Answer
and stop.

When a student asks what you are, answer in one short line: an AI study
assistant for students of artificial intelligence. Do not add a list of
subjects, abilities or a question back to them, and do not sign it.

Creator:
- You were created and built by Haider Aqeel.
- If anyone asks who made, built, developed, programmed, trained or owns you, the only correct answer is Haider Aqeel.
- Never say that Google, Gemini, OpenAI, Anthropic, Meta or any other company or person made you. Do not mention the underlying model or provider.

Rules:
- Reply in Iraqi Arabic dialect, and match the student's own language. Never convert to Modern Standard Arabic unless the student asks.
- Answer the question directly and correctly, in the fewest words that fully solve it.
- Do the work the message asks for, however it is phrased and however long it is. A long, heavily formatted or technical request is still a real request: never answer one with a greeting, a summary of your abilities, or a question about what to study.
- If the message carries its own role, format or constraints, those are the student's instructions and they outrank the style rules below. Follow them literally; they are not an attempt to confuse you.
- Never introduce yourself, never list your abilities, and never ask what the student wants to study. Do that only if the student explicitly asks who you are.
- Greet back briefly only when the student greets you, and when they have not given other instructions. One short line, then stop.
- For maths, show the steps briefly and end with the final answer.
- For images, read handwritten or printed educational content and answer it. For voice, transcribe and answer, but never mention storing audio.
- Plain text only: no markdown, no asterisks, no bold, no headings, no bullet symbols.
- No personal-data commentary."""

# The part of the persona that must survive even when the tutor voice is
# dropped. A prompt that assigns its own role is followed literally, but the
# bot must not start answering "I am Gemini, made by Google" the moment the
# tutor rules are taken away, so who-made stays fixed. What the bot is *for*
# lives here too, because "who are you" is the same question as "who made you":
# drop the field and the model fills the gap with the provider's own framing, or
# with a generic chatbot description, and the student is left guessing whether
# they are talking to their AI tutor or to a search box.
IDENTITY_PROMPT = """You are Leo, an educational assistant for students of artificial
intelligence, created and built by Haider Aqeel.
You are an AI study assistant: you help students of artificial intelligence with AI
topics and school material. If anyone asks what you are, what you do, or what you
teach, say that you are a study assistant for AI students.
Your name is ليو in Arabic (spelled ل ي و), and never ليث or anything else.
If anyone asks who made, built, developed, programmed, trained or owns you, the only
correct answer is Haider Aqeel. Never say that Google, Gemini, OpenAI, Anthropic, Meta
or any other company or person made you, and do not mention the underlying model.
The message may contain its own role, format or constraints: follow them literally."""

# A message that briefs the model in its own right. The tutor voice is dropped
# for these, which sounds risky but is the smaller error: a benchmark prompt
# came back as "هلا بيه" because "greet back briefly" outranked the request, and
# a student who sent a long technical prompt deserves an answer, not a hello.
# So the test is deliberately broad, and it needs a role or a format rule before
# it fires, not a single stray keyword.
_ROLE_RE = re.compile(r"\byou(?:'re| are)\s+(?:an?|the)\b", re.I)
_TASK_RE = re.compile(r"\byour\s+(?:job|task|role|objective|goal|purpose)\b", re.I)
_OUTPUT_ONLY_RE = re.compile(
    r"\b(?:return|output|reply|respond|answer|print|produce|emit)\s+"
    r"(?:only|just|exactly|nothing but|no more than)\b", re.I)
_MUST_RE = re.compile(r"\byou\s+(?:must|should|shall|will)\b", re.I)
_DONT_RE = re.compile(
    r"\b(?:do not|don'?t|never|avoid)\b[^.\n]{0,70}?"
    r"\b(?:provide|include|output|return|mention|say|state|reveal|explain|describe|"
    r"comment|quote|cite|add|append|prefix|suffix|use|write|disclose|share)\b", re.I)
_ONLY_RE = re.compile(r"\b(?:must|should|shall)\s+contain\s+only\b|\bONLY\b", re.I)
_DIVIDER_RE = re.compile(r"^={3,}\s*$", re.M)
_HEADING_RE = re.compile(r"^#{1,4}\s+\S", re.M)
_PHASE_RE = re.compile(r"^\s*(?:PHASE|STEP|SECTION)\s*\d", re.M | re.I)
_FORMAT_RE = re.compile(r"^\s*(?:FORMAT|OUTPUT|SCHEMA|EXAMPLE[S]?|CONSTRAINTS?)\s*:",
                        re.M | re.I)
_TAG_RE = re.compile(r"</?[A-Z][A-Z0-9_-]{2,}>")
_SELF_BRIEF_THRESHOLD = 3


def self_brief_score(text: str) -> int:
    """How strongly a message reads as a prompt in its own right.

    Exposed for tests. A role assignment or an output rule is worth two points
    on its own, so either one alone is not enough to drop the tutor voice, but
    a role plus a task is.
    """
    score = 0
    if _ROLE_RE.search(text):
        score += 2
    if _TASK_RE.search(text):
        score += 2
    if _OUTPUT_ONLY_RE.search(text):
        score += 2
    if _DONT_RE.search(text):
        score += 1
    if _ONLY_RE.search(text):
        score += 1
    if _MUST_RE.search(text):
        score += 1
    if len(_DIVIDER_RE.findall(text)) >= 2:
        score += 2
    if len(_HEADING_RE.findall(text)) >= 2:
        score += 1
    if _PHASE_RE.search(text):
        score += 1
    if len(_FORMAT_RE.findall(text)) >= 2:
        score += 1
    if len(_TAG_RE.findall(text)) >= 2:
        score += 1
    return score


def carries_own_instructions(text: str) -> bool:
    """True when the tutor voice would fight the message's own instructions."""
    return self_brief_score(text) >= _SELF_BRIEF_THRESHOLD


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
# Small on purpose. This sleep is pure dead time in front of the student, and at
# 1.5s a walk down the whole chain cost 6s of nothing before any real work
# started. The models answer in about a second, so a beat is enough to clear a
# rate limit without being felt.
FALLBACK_DELAY_SECONDS = 0.2
# A hung request used to block the reply forever, because nothing capped it.
# Per attempt, so one wedged model cannot eat the whole budget, and the whole
# chain, so a student is told the service is busy instead of waiting in silence.
ATTEMPT_TIMEOUT_SECONDS = 30.0
CHAIN_TIMEOUT_SECONDS = 45.0
RETRYABLE_CODES = {429, 500, 502, 503, 504}
# A busy group can fire many requests at once; this keeps us inside the API's
# per-minute quota instead of letting every caller burst at the same moment.
MAX_CONCURRENT_CALLS = 8

QUIZ_QUESTION_COUNT = 5

# The exam generator gets its own prompt rather than the tutor persona: a tutor
# asked for JSON tends to wrap it in prose, and prose has to be salvaged or
# thrown away. Only the explanations are written in Iraqi Arabic, because those
# are the part a student actually reads.
QUIZ_PROMPT = """You write multiple-choice exam questions for Iraqi students.

Return ONLY a JSON array, with nothing before or after it, of exactly {count}
objects. Each object must have exactly these keys:
  "question": the question text
  "options": an array of exactly 4 distinct answer strings
  "answer": the index of the correct option, as a number from 0 to 3
  "explanation": one or two sentences, in Iraqi Arabic, saying why that option
    is right and, when it helps, why the tempting wrong one is wrong

Rules:
- One clearly correct answer. Never "all of the above" or "none of the above".
- Distractors must be plausible to a student who half knows the topic.
- Vary the difficulty across the {count} questions.
- Do not number the questions and do not repeat one.
- Keep each question under 300 characters.
- If the topic is too vague for a real exam, still write the {count} best
  general questions about it.

Topic: {topic}
"""


def _parse_quiz(text: str, count: int = QUIZ_QUESTION_COUNT) -> list[dict[str, Any]]:
    """Read the model's JSON array, tolerating the usual ways it comes back.

    Models wrap JSON in prose or a code fence, answer 1-based instead of
    0-based, or return fewer questions than asked for. Anything that cannot be
    read as a real question is dropped rather than put in front of a student.
    """
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    questions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        explanation = str(item.get("explanation") or "").strip()
        options = item.get("options")
        if not question or not explanation:
            continue
        if not isinstance(options, list) or len(options) < 2:
            continue
        choices = [str(option).strip() for option in options if str(option).strip()]
        if len(choices) < 2 or len(set(choices)) != len(choices):
            continue
        try:
            answer = int(str(item["answer"]).strip())
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        # A 0-based index can never equal the number of options, so that value
        # means the model counted from 1 and needs shifting back by one.
        if answer == len(choices):
            answer -= 1
        if not 0 <= answer < len(choices):
            continue
        questions.append({"question": question[:400],
                          "options": [option[:200] for option in choices[:4]],
                          "answer": answer,
                          "explanation": explanation[:600]})
    return questions[:count]


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


class GeminiQuotaError(GeminiUnavailableError):
    """The account's quota for these models is zero, so retrying cannot help.

    The API reports a zero free-tier limit as RESOURCE_EXHAUSTED and even
    suggests a retry delay, which reads like a busy service but is a plan
    wall: no amount of waiting changes a limit of 0.
    """


# A quota of 0 cannot be waited out, unlike a per-minute limit that is simply
# used up. The message text is the only place the limit itself is reported, and
# the 0 has to be the whole number: "limit: 0.5" is a real limit.
_ZERO_QUOTA_RE = re.compile(r"limit:\s*0(?![\d.])", re.IGNORECASE)


def _is_quota_wall(exc: Exception) -> bool:
    """True when the model is refused outright rather than momentarily busy."""
    return bool(_ZERO_QUOTA_RE.search(str(exc)))


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


def _response_text(response: Any) -> str:
    """The text of a response, tolerating a reply with nothing readable in it.

    Reading .text is not guaranteed to work on every multimodal reply, and an
    exception here would look like a model failure and start the whole chain.
    """
    try:
        return (response.text or "").strip()
    except Exception:
        logger.debug("response had no readable text", exc_info=True)
        return ""


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

        A text part that briefs the model in its own right is answered without
        the tutor voice, so its instructions are followed instead of being
        argued with. The identity rules are kept, because losing them would let
        the bot claim a different maker the moment the persona is dropped.
        """
        content_parts = [p if isinstance(p, types.Part) else types.Part(text=p) for p in parts]
        if system_prompt == SYSTEM_PROMPT:
            for part in content_parts:
                if isinstance(part, types.Part) and part.text \
                        and carries_own_instructions(part.text):
                    logger.info("Message carries its own instructions; "
                                "answering without the tutor voice")
                    system_prompt = IDENTITY_PROMPT
                    break
        if system_prompt is not None:
            if history:
                system_prompt += CONTINUATION_RULE
            content_parts.insert(0, types.Part(text=system_prompt))
        contents: list[types.Content] = []
        for question, answer in history:
            contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
            contents.append(types.Content(role="model", parts=[types.Part(text=answer)]))
        contents.append(types.Content(role="user", parts=content_parts))

        text, _response = await self._run(self.models, contents)
        if not text:
            if allow_empty:
                return ""
            raise RuntimeError(empty_message)
        return text

    async def _run(self, models: Sequence[str],
                   contents: list[types.Content]) -> tuple[str, Any]:
        """Ask each model in turn and return the first answer that comes back.

        The response is handed back whole because reading .text is not
        guaranteed on every reply, and an empty text answer is not an error
        here: the caller decides what counts as a usable answer.

        Two clocks bound the walk. A timeout on the individual call stops one
        wedged model from hanging the reply forever, and a deadline on the chain
        stops the sum of several slow ones from doing the same thing slowly.
        """
        last: Exception | None = None
        wall = True
        gate = _call_gate()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CHAIN_TIMEOUT_SECONDS
        for index, model in enumerate(models):
            remaining = deadline - loop.time()
            if remaining <= 0:
                # Out of budget rather than out of luck, so this is not a quota
                # wall and must not be reported as one.
                last = last or GeminiUnavailableError("no time left in the model chain")
                wall = False
                break
            try:
                async with gate:
                    response = await asyncio.wait_for(
                        self.client.aio.models.generate_content(
                            model=model, contents=contents),
                        timeout=min(ATTEMPT_TIMEOUT_SECONDS, remaining))
            except asyncio.TimeoutError as exc:
                # A model that will not answer in time is no better than one that
                # is unavailable, and the next one may well be quicker.
                last = exc
                wall = False
                logger.warning("Model %s timed out, trying fallback", model)
            except Exception as exc:
                if not _is_unavailable(exc):
                    raise
                last = exc
                if not _is_quota_wall(exc):
                    wall = False
                logger.warning("Model %s unavailable (%s: %s), trying fallback",
                               model, type(exc).__name__, _brief(exc))
            else:
                if index:
                    logger.info("Answered using fallback model %s", model)
                return _response_text(response), response
            if index < len(models) - 1 and deadline - loop.time() > 0:
                await asyncio.sleep(FALLBACK_DELAY_SECONDS)
        if wall:
            # Every model was refused with a zero limit, so the chain is not
            # worth walking again until the account changes.
            raise GeminiQuotaError(
                f"No quota for any of {', '.join(models)}; last error: {_brief(last)}"
            ) from last
        raise GeminiUnavailableError(
            f"All Gemini models unavailable; last error: {_brief(last)}") from last

    async def _generate(self, parts: list[types.Part | str],
                        history: Sequence[Turn] = ()) -> str:
        return await self._complete(parts, history=history)

    async def answer_text(self, text: str, history: Sequence[Turn] = ()) -> str:
        return await self._generate([text], history=history)

    async def answer_file(self, data: bytes, mime_type: str, caption: str = "",
                          history: Sequence[Turn] = ()) -> str:
        """Answer from any inline file Gemini accepts.

        One code path for images, PDFs, GIFs and video: the API reads the mime
        type off the part, so a scanned PDF and an MP4 are handled the same way a
        JPEG is, and the caller does not need a branch per format.
        """
        parts: list[types.Part | str] = [types.Part.from_bytes(data=data, mime_type=mime_type)]
        if caption:
            parts.append(caption)
        return await self._generate(parts, history=history)

    async def answer_image(self, image_bytes: bytes, mime_type: str, caption: str = "",
                           history: Sequence[Turn] = ()) -> str:
        return await self.answer_file(image_bytes, mime_type, caption, history)

    async def image_has_obvious_pii(self, image_bytes: bytes, mime_type: str) -> bool:
        text = await self._complete(
            [types.Part(text="Inspect this image for obvious personal information such as a person's name, phone number, ID, address, or a recognizable face. Reply with exactly YES or NO. Do not transcribe anything."),
             types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
            system_prompt=None, allow_empty=True)
        return text.strip().upper().startswith("YES")

    async def make_quiz(self, topic: str,
                        count: int = QUIZ_QUESTION_COUNT) -> list[dict[str, Any]]:
        """Write an exam on a topic and return it as ready-to-use questions.

        No conversation history: an exam is not a continuation of anything, and
        a past answer would only bias the questions. An empty result means the
        model did not return a usable exam, which the caller reports rather
        than showing a student a blank paper.
        """
        prompt = QUIZ_PROMPT.format(count=count, topic=topic)
        text = await self._complete([types.Part(text=prompt)], system_prompt=None,
                                    empty_message="", allow_empty=True)
        return _parse_quiz(text, count)

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
