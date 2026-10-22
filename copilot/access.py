"""Zugangsprüfung und flüchtige Limits für die Web-Demo, ohne Dokumentdaten."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import hashlib
import hmac
import logging
import os
from threading import BoundedSemaphore, RLock
from typing import Callable, Iterator
from zoneinfo import ZoneInfo


SESSION_LIMIT = 5
DAILY_LIMIT = 50
CONCURRENT_LIMIT = 2
TIMEZONE = ZoneInfo("Europe/Berlin")
logger = logging.getLogger(__name__)


class AccessError(RuntimeError):
    """Feste Meldungen, die keine Eingaben oder Geheimnisse enthalten."""


class AccessDenied(AccessError):
    def __init__(self):
        super().__init__("Die Live-Analyse ist auf eingeladene Besucher begrenzt.")


class SessionLimitReached(AccessError):
    def __init__(self):
        super().__init__("Das Analyse-Limit für diese Sitzung ist erreicht. Danke fürs Ausprobieren!")


class DailyLimitReached(AccessError):
    def __init__(self):
        super().__init__("Das tägliche Analyse-Limit ist erreicht. Bitte versuche es morgen erneut.")


class AnalysesBusy(AccessError):
    def __init__(self):
        super().__init__("Gerade werden bereits Analysen ausgeführt. Bitte versuche es in einem Moment erneut.")


def _configured_code() -> str | None:
    try:
        code = os.environ["ACCESS_CODE"]
    except KeyError:
        return None
    return code if code.strip() else None


def valid_code(code: str | None) -> bool:
    expected = _configured_code()
    return (expected is not None and isinstance(code, str)
            and hmac.compare_digest(code.encode("utf-8"), expected.encode("utf-8")))


class SessionAccess:
    """Eine Sitzung hält nur Freigabe-Fingerabdruck, Zähler und laufenden Aufruf."""

    def __init__(self):
        self._lock = RLock()
        self._fingerprint: bytes | None = None
        self._analyses = 0
        self._active: AnalysisPermit | None = None

    def refresh(self, code: str | None = None) -> bool:
        with self._lock:
            if code is not None:
                # Den Klartext-Code weder speichern noch protokollieren.
                self._fingerprint = hashlib.sha256(code.encode("utf-8")).digest() if valid_code(code) else None
            return self.authorized

    @property
    def authorized(self) -> bool:
        with self._lock:
            expected = _configured_code()
            current = hashlib.sha256(expected.encode("utf-8")).digest() if expected else None
            if (self._fingerprint is None or current is None
                    or not hmac.compare_digest(self._fingerprint, current)):
                self._fingerprint = None
                return False
            return True

    @property
    def analyses(self) -> int:
        with self._lock:
            return self._analyses


class AnalysisLimits:
    """Ein Prozess teilt Lock, Tageszähler und zwei gleichzeitig belegbare Plätze."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None):
        self._clock = clock or (lambda: datetime.now(TIMEZONE))
        self._lock = RLock()
        self._slots = BoundedSemaphore(CONCURRENT_LIMIT)
        self._day = self._today()
        self._daily_count = 0

    def _today(self):
        return self._clock().astimezone(TIMEZONE).date()

    def _rollover(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._daily_count = today, 0

    @property
    def daily_count(self) -> int:
        with self._lock:
            self._rollover()
            return self._daily_count

    def _available(self, session: SessionAccess) -> None:
        self._rollover()
        if not session.authorized:
            raise AccessDenied()
        if session._analyses >= SESSION_LIMIT:
            raise SessionLimitReached()
        if self._daily_count >= DAILY_LIMIT:
            raise DailyLimitReached()

    def check_available(self, session: SessionAccess) -> None:
        with self._lock, session._lock:
            self._available(session)
            if session._active is not None:
                raise AnalysesBusy()

    @contextmanager
    def analysis(self, session: SessionAccess) -> Iterator[AnalysisPermit]:
        with self._lock, session._lock:
            self._available(session)
            if session._active is not None or not self._slots.acquire(blocking=False):
                raise AnalysesBusy()
            permit = AnalysisPermit(self, session)
            session._active = permit
        token = _CURRENT_ANALYSIS.set(permit)
        try:
            yield permit
        finally:
            _CURRENT_ANALYSIS.reset(token)
            with self._lock, session._lock:
                permit.closed = True
                session._active = None
                self._slots.release()

    def _check(self, permit: AnalysisPermit, *, start: bool) -> None:
        session = permit.session
        with self._lock, session._lock:
            if permit.closed or session._active is not permit or not session.authorized:
                raise AccessDenied()
            if not permit.started:
                self._available(session)
                if start:
                    session._analyses += 1
                    self._daily_count += 1
                    permit.started = True
                    logger.info("session_count=%d daily_count=%d", session._analyses, self._daily_count)


class AnalysisPermit:
    def __init__(self, limits: AnalysisLimits, session: SessionAccess):
        self.limits, self.session = limits, session
        self.started = False
        self.closed = False

    def check(self, *, start: bool = False) -> None:
        self.limits._check(self, start=start)


_CURRENT_ANALYSIS: ContextVar[AnalysisPermit | None] = ContextVar("web_analysis", default=None)


def check_api_access() -> None:
    """Prüfung vor jedem logischen API-Aufruf; die CLI hat keinen Web-Kontext."""
    permit = _CURRENT_ANALYSIS.get()
    if permit is not None:
        permit.check()


def before_api_request(_request) -> None:
    """HTTP-Hook: prüft auch SDK-Retries, zählt erst beim ersten Sendeversuch."""
    permit = _CURRENT_ANALYSIS.get()
    if permit is not None:
        permit.check(start=True)
