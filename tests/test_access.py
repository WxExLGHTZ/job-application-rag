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


def test_context_reset_error_still_frees_the_same_slot():
    limits, session = access.AnalysisLimits(), invited()
    original = Context()
    manager = limits.analysis(session)
    original.run(manager.__enter__)
    with pytest.raises(ValueError):
        Context().run(manager.__exit__, None, None, None)
    with limits.analysis(session), limits.analysis(invited()):
        with pytest.raises(access.AnalysesBusy):
            start(limits, invited())


def test_expired_places_are_reclaimed_and_old_finally_cannot_release_replacement():
    now = [0.0]
    limits = access.AnalysisLimits(monotonic_clock=lambda: now[0])
    session = invited()
    old_context = Context()
    old_manager = limits.analysis(session)
    old = old_context.run(old_manager.__enter__)
    now[0] = limits.lease_seconds - 0.001
    with pytest.raises(access.AnalysesBusy):
        start(limits, session)
    now[0] = limits.lease_seconds
    with limits.analysis(session) as replacement, limits.analysis(invited()):
        assert old.expired and old.closed
        with pytest.raises(access.AnalysisExpired):
            old_context.run(access.before_api_request, None)
        old_context.run(old_manager.__exit__, None, None, None)
        replacement.check()
        with pytest.raises(access.AnalysesBusy):
            start(limits, invited())
    assert limits.daily_count == session.analyses == 0


def test_two_orphaned_places_both_recovered():
    now = [0.0]
    limits = access.AnalysisLimits(monotonic_clock=lambda: now[0])
    held = []
    try:
        for _ in range(2):
            context = Context()
            manager = limits.analysis(invited())
            context.run(manager.__enter__)
            held.append((context, manager))
        with pytest.raises(access.AnalysesBusy):
            start(limits, invited())
        now[0] += limits.lease_seconds + 1
        with limits.analysis(invited()), limits.analysis(invited()):
            with pytest.raises(access.AnalysesBusy):
                start(limits, invited())
    finally:
        for context, manager in held:
            context.run(manager.__exit__, None, None, None)


def test_expiration_does_not_refund_a_started_analysis():
    now = [0.0]
    limits, session = access.AnalysisLimits(monotonic_clock=lambda: now[0]), invited()
    with limits.analysis(session) as permit:
        access.before_api_request(None)
        now[0] += limits.lease_seconds
        with pytest.raises(access.AnalysisExpired):
            permit.check()
    start(limits, session)
    assert session.analyses == limits.daily_count == 2


def test_expiration_uses_monotonic_time_not_the_day_reset_clock():
    now = [datetime(2026, 10, 22, 12, tzinfo=timezone.utc)]
    monotonic_now = [0.0]
    limits = access.AnalysisLimits(clock=lambda: now[0], monotonic_clock=lambda: monotonic_now[0])
    session = invited()
    with limits.analysis(session) as permit:
        now[0] = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
        permit.check()
        with pytest.raises(access.AnalysesBusy):
            start(limits, session)
        monotonic_now[0] = limits.lease_seconds
        with pytest.raises(access.AnalysisExpired):
            permit.check()


def test_analysis_failure_after_first_request_releases_place_without_refunding_quota():
    limits, session = access.AnalysisLimits(), invited()
    with pytest.raises(RuntimeError):
        with limits.analysis(session):
            access.before_api_request(None)
            raise RuntimeError("Test")
    start(limits, session)
    assert session.analyses == limits.daily_count == 2


def test_concurrent_double_click_on_same_session_starts_only_one_analysis():
    limits, session = access.AnalysisLimits(), invited()
    ready, finish = Event(), Event()
    def first_click():
        with limits.analysis(session):
            access.before_api_request(None)
            ready.set()
            assert finish.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(first_click)
        try:
            assert ready.wait(timeout=5)
            with pytest.raises(access.AnalysesBusy):
                start(limits, session)
            assert session.analyses == limits.daily_count == 1
        finally:
            finish.set()
        first.result(timeout=10)
    start(limits, session)
    assert session.analyses == limits.daily_count == 2


@pytest.mark.parametrize("timeout,retries,corrections,stages", [
    (60, 1, 1, 2), (90, 2, 2, 3), (15, 0, 0, 1),
])
def test_lease_budget_tracks_all_runtime_settings(monkeypatch, timeout, retries, corrections, stages):
    from copilot import runtime_config as config
    monkeypatch.setattr(config, "API_TIMEOUT_SECONDS", timeout)
    monkeypatch.setattr(config, "SDK_MAX_RETRIES", retries)
    monkeypatch.setattr(config, "RESPONSE_CORRECTION_RETRIES", corrections)
    monkeypatch.setattr(config, "ANALYSIS_API_STAGES", tuple(range(stages)))
    longest_api_budget = stages * (1 + corrections) * (
        (1 + retries) * timeout + retries * config.SDK_MAX_RETRY_WAIT_SECONDS
    )
    limits = access.AnalysisLimits()
    assert limits.lease_seconds == longest_api_budget + 30
    assert limits.lease_seconds > longest_api_budget


def test_longest_sdk_and_correction_chain_fits_lease(sdk, monkeypatch, pdf_bytes):
    import json
    import numpy as np
    from anthropic import _base_client
    from copilot import index, pipeline, runtime_config as config
    now, waits, attempts = [0.0], [], {}
    monkeypatch.setenv("MODEL_EXTRACT", llm.DEFAULT_EXTRACT_MODEL)
    monkeypatch.setenv("MODEL_ASSESS", llm.DEFAULT_ASSESS_MODEL)
    def embed(texts):
        return np.ones((len(texts), 2), dtype=np.float32)
    monkeypatch.setattr(pipeline, "embed_texts", embed)
    monkeypatch.setattr(index, "embed_texts", embed)
    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(_base_client.time, "sleep", sleep)
    def transport(request):
        assert request.extensions["timeout"]["read"] == config.API_TIMEOUT_SECONDS
        now[0] += config.API_TIMEOUT_SECONDS
        model = json.loads(request.content)["model"]
        attempts[model] = attempts.get(model, 0) + 1
        if attempts[model] % (1 + config.SDK_MAX_RETRIES):
            return httpx.Response(529, headers={"retry-after": str(config.SDK_MAX_RETRY_WAIT_SECONDS)},
                                  json={"error": {"type": "overloaded_error", "message": "Test"}})
        if attempts[model] < (1 + config.SDK_MAX_RETRIES) * (1 + config.RESPONSE_CORRECTION_RETRIES):
            return sdk_response("kein JSON")
        if model == llm.DEFAULT_EXTRACT_MODEL:
            return sdk_response('[{"text":"SQL","typ":"muss"}]')
        return sdk_response(json.dumps({"requirements": [{"id": "R1", "status": "nicht_belegt", "beleg": None,
                              "begruendung": "Kein Beleg"}], "fazit": "Nachweis fehlt."}))
    sdk(transport)
    limits, session = access.AnalysisLimits(monotonic_clock=lambda: now[0]), invited()
    with limits.analysis(session) as permit:
        pipeline.analyze(pdf_bytes(["SQL Anforderungen " * 20]), pdf_bytes(["Erfahrung im Beruf " * 20]), "job")
        permit.check()
        assert not permit.expired
    assert len(attempts) == len(config.ANALYSIS_API_STAGES)
    assert sum(attempts.values()) == len(config.ANALYSIS_API_STAGES) * (1 + config.SDK_MAX_RETRIES) * (1 + config.RESPONSE_CORRECTION_RETRIES)
    assert now[0] == limits.lease_seconds - config.ANALYSIS_BUFFER_SECONDS
    assert waits and max(waits) == config.SDK_MAX_RETRY_WAIT_SECONDS
    assert limits.lease_seconds > now[0]
    assert session.analyses == limits.daily_count == 1
