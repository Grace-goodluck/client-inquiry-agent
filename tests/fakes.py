"""Fakes for tests. No test touches the network, Gemini or the Second Brain."""

import json

from client_inquiry_agent.brain_client import BrainReply
from client_inquiry_agent.retrieval import Chunk

WEEKENDS = Chunk(
    "2026-11-03_how-i-work.md",
    "Communication",
    "I work Monday to Friday and do not work on weekends. I reply to messages within 24 hours on working days.",
)
REFUNDS = Chunk(
    "2026-11-03_faq.md",
    "Do you offer refunds?",
    "No. Payments are non-refundable. A client who stops early keeps everything completed up to that point.",
)
PAYMENT = Chunk(
    "2026-11-03_services-and-pricing.md",
    "Payment terms",
    "50% deposit before work starts, 50% on delivery.",
)
CHUNKS = [WEEKENDS, REFUNDS, PAYMENT]

WEEKENDS_ID = "2026-11-03_how-i-work.md > Communication"
REFUNDS_ID = "2026-11-03_faq.md > Do you offer refunds?"
PAYMENT_ID = "2026-11-03_services-and-pricing.md > Payment terms"

GOOD_REPLY = (
    "Hi Ada,\n\nThank you for getting in touch! I work Monday to Friday and don't work on weekends, "
    "and I reply to messages within 24 hours on working days.\n\nWarm regards,\nGrace, TechByGrace"
)


def gemini_json(
    category="policy",
    reply=GOOD_REPLY,
    sources=(WEEKENDS_ID,),
    needs_grace=(),
) -> str:
    return json.dumps(
        {"category": category, "reply": reply, "sources": list(sources), "needs_grace": list(needs_grace)}
    )


class FakeLLM:
    """Returns scripted replies in order and records every prompt.

    A scripted item that is an Exception instance is raised instead.
    """

    def __init__(self, replies: list[object], model: str = "fake-model") -> None:
        self.model = model
        self._replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._replies:
            raise AssertionError(f"FakeLLM got an unexpected call #{len(self.prompts)}")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSource:
    """A knowledge source that returns fixed chunks and records queries."""

    def __init__(self, chunks=CHUNKS) -> None:
        self.chunks = list(chunks)
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        self.queries.append((query, top_k))
        return self.chunks[:top_k]


class FakeBrain:
    def __init__(self, reply: BrainReply) -> None:
        self.reply = reply
        self.questions: list[str] = []

    def ask(self, question: str) -> BrainReply:
        self.questions.append(question)
        return self.reply


class FakeAPIError(Exception):
    """Shaped like google-genai's APIError: an int .code."""

    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(f"{code} {message}")
        self.code = code


class FakeClock:
    """Stands in for time.monotonic/time.sleep: sleeping advances the clock.

    wait(event, seconds) stands in for the Gemini timeout's wait: it gives a quick fake call a
    moment of real time to finish; if it hasn't, the clock jumps forward `seconds` (no real waiting).
    """

    REAL_GRACE_S = 0.5

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def wait(self, event, seconds: float) -> None:
        if event.wait(self.REAL_GRACE_S):
            return
        self.waits.append(seconds)
        self.t += seconds
