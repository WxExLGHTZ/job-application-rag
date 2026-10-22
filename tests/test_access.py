from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from datetime import datetime, timezone
from threading import Barrier, Event

import httpx
import pytest

from copilot import access, llm


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "synthetic-invitation")


def invited():
    session = access.SessionAccess()
    assert session.refresh("synthetic-invitation")
    return session


def start(limits, session):
    with limits.analysis(session):
        access.before_api_request(None)


@pytest.mark.parametrize("code,expected", [("synthetic-invitation", True), ("wrong", False), (None, False), ("", False)])
def test_code_verification(code, expected, caplog):
    assert access.valid_code(code) is expected
    assert "synthetic-invitation" not in caplog.text
    assert "wrong" not in caplog.text


@pytest.mark.parametrize("configured_code", [None, "", "   "])
def test_missing_configuration_always_locks(monkeypatch, configured_code):
    session = invited()
    if configured_code is None:
        monkeypatch.delenv("ACCESS_CODE")
    else:
        monkeypatch.setenv("ACCESS_CODE", configured_code)
    assert not access.valid_code("synthetic-invitation")
    assert not session.refresh()
    with pytest.raises(access.AccessDenied):
        access.AnalysisLimits().check_available(session)


def test_unicode_code_and_constant_time_comparison(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "Einladung-ä-🔐")
    original = access.hmac.compare_digest
    calls = []
    def compare(left, right):
        calls.append((type(left), type(right)))
        return original(left, right)
    monkeypatch.setattr(access.hmac, "compare_digest", compare)
    assert access.valid_code("Einladung-ä-🔐")
    assert calls == [(bytes, bytes)]


def test_session_retains_only_freigabe_and_rotation_revokes(monkeypatch):
    session = invited()
    assert session.refresh(None)
    assert "synthetic-invitation" not in repr(vars(session))
    monkeypatch.setenv("ACCESS_CODE", "new-invitation")
    assert not session.authorized
    assert not session.refresh("wrong")
    assert session.refresh("new-invitation")


def test_fifth_analysis_finishes_retries_count_once(caplog):
    caplog.set_level("INFO", logger="copilot.access")
    limits, session = access.AnalysisLimits(), invited()
    for _ in range(5):
        with limits.analysis(session):
            for _ in range(4):
                access.check_api_access()
                access.before_api_request(None)
    assert session.analyses == limits.daily_count == 5
    with pytest.raises(access.SessionLimitReached, match="Danke fürs Ausprobieren"):
        start(limits, session)
    assert "synthetic-invitation" not in caplog.text
    assert "session_count=5 daily_count=5" in caplog.text


def test_daily_limit_and_berlin_midnight_reset():
    now = [datetime(2026, 10, 22, 21, 59, tzinfo=timezone.utc)]
    limits = access.AnalysisLimits(clock=lambda: now[0])
    session = None
    for _ in range(10):
        session = invited()
        for _ in range(5):
            start(limits, session)
    assert limits.daily_count == 50
    with pytest.raises(access.DailyLimitReached):
        start(limits, invited())
    now[0] = datetime(2026, 10, 22, 22, 1, tzinfo=timezone.utc)
    assert limits.daily_count == 0
    with pytest.raises(access.SessionLimitReached):
        start(limits, session)
    start(limits, invited())
    assert limits.daily_count == 1


def test_analysis_started_before_midnight_can_finish_after_reset():
    now = [datetime(2026, 10, 22, 21, 59, tzinfo=timezone.utc)]
    limits, session = access.AnalysisLimits(clock=lambda: now[0]), invited()
    with limits.analysis(session):
        access.before_api_request(None)
        now[0] = datetime(2026, 10, 22, 22, 1, tzinfo=timezone.utc)
        access.before_api_request(None)
    assert session.analyses == 1
    assert limits.daily_count == 0


def test_atomic_daily_admission_for_two_competing_starts():
    limits = access.AnalysisLimits()
    for _ in range(49):
        start(limits, invited())
    barrier = Barrier(2)
    def compete():
        try:
            with limits.analysis(invited()):
                barrier.wait(timeout=5)
                access.before_api_request(None)
                return "started"
        except access.DailyLimitReached:
            return "blocked"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(compete) for _ in range(2)]
        assert sorted(future.result(timeout=10) for future in futures) == ["blocked", "started"]
    assert limits.daily_count == 50


def test_only_two_concurrent_analyses_and_places_reusable():
    limits, barrier, release = access.AnalysisLimits(), Barrier(3), Event()
    def hold():
        with limits.analysis(invited()):
            access.before_api_request(None)
            barrier.wait(timeout=5)
            assert release.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(hold) for _ in range(2)]
        try:
            barrier.wait(timeout=5)
            third = invited()
            with pytest.raises(access.AnalysesBusy):
                start(limits, third)
            assert third.analyses == 0
            assert limits.daily_count == 2
        finally:
            release.set()
        for future in futures:
            future.result(timeout=10)
    start(limits, third)
    assert limits.daily_count == 3


@pytest.mark.parametrize("error", [ValueError("PRIVATE"), KeyboardInterrupt()])
def test_failed_preparation_does_not_count_and_frees_place(error):
    limits, session = access.AnalysisLimits(), invited()
    with pytest.raises(type(error)):
        with limits.analysis(session):
            access.check_api_access()
            raise error
    assert session.analyses == limits.daily_count == 0
    start(limits, session)
    assert limits.daily_count == 1


def test_same_session_cannot_launch_twice():
    limits, session = access.AnalysisLimits(), invited()
    with limits.analysis(session):
        with pytest.raises(access.AnalysesBusy):
            start(limits, session)
    assert session.analyses == 0


def test_guard_rejects_released_context_and_cli_has_no_web_quota():
    limits = access.AnalysisLimits()
    with limits.analysis(invited()):
        stale_context = copy_context()
    with pytest.raises(access.AccessDenied):
        stale_context.run(access.before_api_request, None)
    Context().run(access.before_api_request, None)
    assert limits.daily_count == 0


def sdk_response(text):
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": llm.DEFAULT_EXTRACT_MODEL,
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-api-key")
    monkeypatch.setattr(llm, "_CLIENT", None)
    transports = []
    def configure(transport):
        monkeypatch.setattr(llm, "DefaultHttpxClient", lambda **kwargs: httpx.Client(
            transport=httpx.MockTransport(transport), **kwargs,
        ))
        transports.append(transport)
    yield configure
    if llm._CLIENT is not None:
        llm._CLIENT.close()


def test_real_sdk_retries_and_correction_have_one_quota_charge(sdk):
    requests = []
    def transport(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(529, headers={"retry-after-ms": "1"}, json={
                "error": {"type": "overloaded_error", "message": "Test"},
            })
        return sdk_response("kein JSON" if len(requests) == 2 else '[{"text":"SQL","typ":"muss"}]')
    sdk(transport)
    limits, session = access.AnalysisLimits(), invited()
    with limits.analysis(session):
        assert llm.extract_requirements("SQL")[0]["text"] == "SQL"
    assert len(requests) == 3
    assert session.analyses == limits.daily_count == 1


def test_sdk_retry_is_blocked_after_code_revocation(sdk, monkeypatch):
    requests = []
    def transport(request):
        requests.append(request)
        monkeypatch.delenv("ACCESS_CODE")
        return httpx.Response(529, headers={"retry-after-ms": "1"}, json={
            "error": {"type": "overloaded_error", "message": "Test"},
        })
    sdk(transport)
    limits, session = access.AnalysisLimits(), invited()
    with limits.analysis(session), pytest.raises(access.AccessDenied):
        llm.extract_requirements("PRIVATE_DOCUMENT")
    assert len(requests) == 1
    assert limits.daily_count == 1


def test_each_api_stage_rechecks_authorization(sdk, monkeypatch):
    requests = []
    def transport(request):
        requests.append(request)
        return sdk_response('[{"text":"SQL","typ":"muss"}]')
    sdk(transport)
    limits, session = access.AnalysisLimits(), invited()
    with limits.analysis(session):
        llm.extract_requirements("SQL")
        monkeypatch.setenv("ACCESS_CODE", "changed")
        with pytest.raises(access.AccessDenied):
            llm.assess_requirements([{"id": "R1", "text": "SQL", "typ": "muss", "retrieved_chunks": []}])
    assert len(requests) == 1


def test_token_budget_rejection_never_charges_quota(sdk):
    requests = []
    sdk(lambda request: requests.append(request))
    limits, session = access.AnalysisLimits(), invited()
    with limits.analysis(session), pytest.raises(llm.RequestTooLarge):
        llm.extract_requirements("x" * 100_000)
    assert not requests
    assert session.analyses == limits.daily_count == 0
