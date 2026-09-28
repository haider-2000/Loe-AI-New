from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

# How many question/answer pairs the bot carries into the next prompt. Five is
# enough to follow a topic across a class without dragging yesterday's chatter
# along, and it keeps the prompt small on a free tier.
DEFAULT_TURNS = 5
# A single lesson answer can be long, and five of them would blow past what is
# worth carrying. The oldest tail of a long answer adds nothing to the topic.
MAX_CHARS_PER_TURN = 500
# The bot now answers every group, so the number of live threads is bounded by
# how many people actually talk to it. The cap keeps a flood of new students
# from growing the process without limit; the least recently used thread goes.
MAX_CONVERSATIONS = 400

Turn = tuple[str, str]


@dataclass
class ConversationMemory:
    """Short-term memory of the last few turns, per student and per chat.

    A group is not a private room: two students asking in the same group must
    not be handed each other's history, so a thread is keyed by chat *and*
    user. It lives only in the process, which is the point - the dataset keeps
    the answers that were worth keeping, never the back-and-forth, and a
    restart honestly starts a fresh conversation.
    """

    max_turns: int = DEFAULT_TURNS
    max_chars: int = MAX_CHARS_PER_TURN
    max_conversations: int = MAX_CONVERSATIONS
    _threads: "OrderedDict[str, list[Turn]]" = field(default_factory=OrderedDict)

    @staticmethod
    def _key(chat_id: int, user_id: int) -> str:
        return f"{chat_id}:{user_id}"

    def _clip(self, text: str | None) -> str:
        collapsed = " ".join((text or "").split())
        if len(collapsed) <= self.max_chars:
            return collapsed
        return collapsed[: self.max_chars].rstrip() + "…"

    def recent(self, chat_id: int, user_id: int) -> list[Turn]:
        """The stored turns, oldest first, ready to hand to the model."""
        return list(self._threads.get(self._key(chat_id, user_id), ()))

    def remember(self, chat_id: int, user_id: int, question: str, answer: str) -> None:
        """Keep one finished exchange, dropping the oldest once the cap is hit."""
        clean_question, clean_answer = self._clip(question), self._clip(answer)
        if not clean_question or not clean_answer:
            return
        key = self._key(chat_id, user_id)
        thread = self._threads.get(key)
        if thread is None:
            thread = self._threads[key] = []
        else:
            self._threads.move_to_end(key)
        thread.append((clean_question, clean_answer))
        while len(thread) > self.max_turns:
            thread.pop(0)
        while len(self._threads) > self.max_conversations:
            self._threads.popitem(last=False)

    def forget(self, chat_id: int, user_id: int) -> int:
        """Drop one thread; returns how many turns were discarded."""
        thread = self._threads.pop(self._key(chat_id, user_id), None)
        return len(thread) if thread else 0

    def turn_count(self, chat_id: int, user_id: int) -> int:
        return len(self._threads.get(self._key(chat_id, user_id), ()))

    def conversation_count(self) -> int:
        return len(self._threads)

    def clear(self) -> None:
        self._threads.clear()
