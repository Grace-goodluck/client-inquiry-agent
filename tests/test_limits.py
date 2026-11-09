"""Web rate limiter: 5 per visitor per rolling hour, 20 per day in total (SCOPE.md, Web)."""

from datetime import datetime, timezone

from client_inquiry_agent.web.limits import Limiter, Limits, seconds_until_pacific_midnight
from fakes import FakeNow


def test_defaults_match_scope():
    assert Limits() == Limits(drafts_per_visitor_per_hour=5, drafts_per_day=20)


def test_refusals_are_not_counted():
    now = FakeNow()
    limiter = Limiter(Limits(drafts_per_visitor_per_hour=1, drafts_per_day=2), now)
    assert limiter.try_start("a") is None
    assert limiter.try_start("a") is not None  # refused, not counted toward the day
    assert limiter.try_start("b") is None
    assert limiter.try_start("c") is not None  # day limit of 2 reached


def test_retry_after_counts_down_to_the_oldest_visit():
    now = FakeNow()
    limiter = Limiter(Limits(drafts_per_visitor_per_hour=1), now)
    limiter.try_start("a")
    now.advance(minutes=20)
    refusal = limiter.try_start("a")
    assert refusal.retry_after_s == 40 * 60
    assert "about 40 minutes" in refusal.message


def test_day_resets_at_pacific_midnight():
    # 2026-11-09 07:59 UTC is 23:59 Pacific (PST, UTC-8).
    now = FakeNow(datetime(2026, 11, 9, 7, 59, tzinfo=timezone.utc))
    limiter = Limiter(Limits(drafts_per_visitor_per_hour=100, drafts_per_day=1), now)
    assert limiter.try_start("a") is None
    assert limiter.try_start("b") is not None
    now.advance(minutes=2)  # 00:01 Pacific, a new day
    assert limiter.try_start("b") is None


def test_seconds_until_pacific_midnight():
    assert seconds_until_pacific_midnight(datetime(2026, 11, 9, 7, 0, tzinfo=timezone.utc)) == 3600
