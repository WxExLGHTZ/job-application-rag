import copy
from io import BytesIO
import html
import json
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import app
from copilot import llm
from copilot.access import before_api_request
from copilot.ingest import DocumentTooLong, PdfExtractionError, PdfWithoutText


@pytest.fixture
def result():
    return {
        "mode": "job",
        "summary": {"score": 75, "muss_erfuellt": "1/1", "fazit": "Gute fachliche Übereinstimmung."},
        "requirements": [
            {"id": "R1", "text": "Python entwickeln", "typ": "muss", "status": "erfuellt",
             "beleg": {"zitat": "Python und SQL.", "seite": 2}, "begruendung": "Erfahrung ist belegt."},
            {"id": "R2", "text": "Projekte leiten", "typ": "kann", "status": "teilweise",
             "beleg": {"zitat": "Mitarbeit in Projekten.", "seite": 1}, "begruendung": "Leitung ist nicht belegt."},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@pytest.fixture
def live_access(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "test-invitation")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    clear_limits = app.get_limits.clear
    clear_limits()
    yield
    clear_limits()


def live_app():
    at = AppTest.from_string("import app\napp.main()")
    at.query_params["code"] = "test-invitation"
    return at.run()


@pytest.fixture
def ui(monkeypatch, result, live_access):
    monkeypatch.setattr(app, "load_model", lambda: object())

    def analyze(requirements, profile, mode, *, on_progress):
        for stage in app.STAGES:
            on_progress(stage)
        return {**copy.deepcopy(result), "mode": mode}

    mocked = Mock(side_effect=analyze)
    monkeypatch.setattr(app.pipeline, "analyze", mocked)
    return live_app(), mocked


def test_initial_view_and_mode_dependent_labels(ui):
    at, analyze = ui
    assert not at.exception
    assert any("Anforderungs-Check" in item.value for item in at.markdown)
    assert at.button[1].disabled
    assert [item.proto.label for item in at.get("file_uploader")] == [
        "Stellenanzeige (PDF)", "Lebenslauf (PDF)",
    ]
    assert all(list(item.proto.type) == [".pdf"] for item in at.get("file_uploader"))
    at.get("button_group")[0].set_value("tender").run()
    assert [item.proto.label for item in at.get("file_uploader")] == [
        "Ausschreibung (PDF)", "Firmenprofil (PDF)",
    ]
    at.button[0].click().run()
    assert "folgt noch" in at.info[0].value
    assert at.button[1].disabled
    analyze.assert_not_called()


def test_example_analysis_progress_download_and_rerun(ui, result):
    at, analyze = ui
    at.button[0].click().run()
    assert not at.button[1].disabled
    assert any("job_anforderungen.pdf" in item.value for item in at.markdown)
    at.button[1].click().run()
    assert not at.exception
    assert not at.error
    args = analyze.call_args.args
    assert args == (
        (app.EXAMPLES_DIR / "job_anforderungen.pdf").read_bytes(),
        (app.EXAMPLES_DIR / "job_profil.pdf").read_bytes(), "job",
    )
    assert at.session_state["result"] == result
    assert at.session_state["analysis_meta"]["duration_seconds"] >= 0
    assert at.status[0].state == "complete"
    assert [item.value for item in at.status[0].markdown] == list(app.STAGES.values())
    assert [item.proto.label for item in at.get("download_button")] == ["Markdown", "JSON"]
    assert any("Muss-Kriterien" in item.value and "1/1" in item.value for item in at.markdown)
    at.run()
    analyze.assert_called_once()
    at.get("button_group")[0].set_value("tender").run()
    assert "result" not in at.session_state
    assert "analysis_meta" not in at.session_state
    assert not at.get("download_button")
    assert at.button[1].disabled


def test_results_do_not_leak_between_sessions(ui):
    at, _ = ui
    at.button[0].click().run()
    at.button[1].click().run()
    other = live_app()
    assert "result" in at.session_state
    assert "result" not in other.session_state
    assert "analysis_meta" not in other.session_state
    assert not other.get("download_button")


@pytest.mark.parametrize("error, expected", [
    (PdfWithoutText("PRIVATE"), "Textschicht"),
    (DocumentTooLong("PRIVATE"), "15 Seiten"),
    (PdfExtractionError("PRIVATE"), "PDF konnte"),
    (llm.RequestTooLarge("PRIVATE"), "Tokenlimit"),
    (llm.OutputTokenLimit("PRIVATE"), "Ausgabelimit"),
    (llm.LlmResponseError("PRIVATE"), "keine verwertbare"),
    (llm.MissingApiKeyError("PRIVATE"), "nicht eingerichtet"),
    (llm.LlmApiError("timeout"), "zu lange"),
    (llm.LlmApiError("rate_limit"), "zu viele Anfragen"),
    (llm.LlmApiError("spending_limit"), "Guthaben"),
    (llm.LlmApiError("connection"), "unterbrochen"),
    (llm.LlmApiError("unavailable"), "überlastet"),
    (llm.LlmApiError("authentication"), "Zugangsdaten"),
    (llm.LlmApiError("request_too_large"), "zu groß"),
    (llm.LlmApiError("api"), "nicht verarbeiten"),
    (RuntimeError("PRIVATE"), "nicht abgeschlossen"),
])
def test_errors_are_friendly_and_clear_old_result(ui, caplog, error, expected):
    at, analyze = ui
    at.button[0].click().run()
    at.button[1].click().run()
    analyze.side_effect = error
    at.button[1].click().run()
    assert not at.exception
    assert expected in at.error[0].value
    assert "PRIVATE" not in at.error[0].value
    assert "PRIVATE" not in caplog.text
    assert at.status[0].state == "error"
    assert "result" not in at.session_state
    assert "analysis_meta" not in at.session_state
    assert not at.get("download_button")


def test_model_load_failure_does_not_show_exception(monkeypatch, live_access):
    monkeypatch.setattr(app, "load_model", Mock(side_effect=RuntimeError("PRIVATE")))
    at = live_app()
    assert not at.exception
    assert "Modell" in at.error[0].value
    assert "PRIVATE" not in at.error[0].value


def test_missing_example_files_are_handled(ui, monkeypatch, tmp_path):
    at, analyze = ui
    monkeypatch.setattr(app, "EXAMPLES_DIR", tmp_path)
    at.button[0].click().run()
    at.button[1].click().run()
    assert not at.exception
    assert "Beispieldateien" in at.error[0].value
    analyze.assert_not_called()


def test_uploads_use_getvalue_without_disk_access(monkeypatch):
    uploads = [BytesIO(b"PDF-A"), BytesIO(b"PDF-B")]
    for upload in uploads:
        upload.seek(0, 2)
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("Kein Dateizugriff")))
    assert app.read_documents(*uploads, False) == (b"PDF-A", b"PDF-B")


def test_upload_change_invalidates_example_and_result(monkeypatch, result):
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    state = State(result=result, analysis_meta={"duration_seconds": 10}, result_filter="Lücken",
                  use_example=True, example_notice=True)
    monkeypatch.setattr(app.st, "session_state", state)
    app.change_upload()
    assert state == {"use_example": False, "example_notice": False}


@pytest.mark.parametrize("score,color", [(0, "red"), (49, "red"), (50, "yellow"), (74, "yellow"), (75, "green"), (100, "green")])
def test_score_colors(score, color):
    assert app.score_color(score) == color


@pytest.mark.parametrize("status,typ,color", [
    ("erfuellt", "muss", "green"), ("teilweise", "muss", "yellow"),
    ("nicht_belegt", "muss", "red"), ("nicht_belegt", "kann", "gray"),
])
def test_status_colors(status, typ, color):
    assert app.status_color({"status": status, "typ": typ}) == color


def test_html_escapes_every_document_text(result):
    attack = '<script>alert("x")</script> & <img src=x onerror="x">'
    result["summary"]["fazit"] = attack
    result["summary"]["muss_erfuellt"] = attack
    item = result["requirements"][0]
    item.update(text=attack, begruendung=attack, beleg={"zitat": attack, "seite": attack})
    rendered = app.render_result_html(result)
    assert attack not in rendered
    assert rendered.count(html.escape(attack, quote=True)) == 6
    assert "<script>" not in rendered
    assert "<em>" in rendered


def test_markdown_export_contains_complete_result_and_safe_text(result):
    result["requirements"][0]["text"] = "<script> [Link](https://example.com) **Text**"
    markdown = app.result_markdown(result)
    assert "**Score: 75/100**" in markdown
    assert "**Muss erfüllt: 1/1**" in markdown
    assert "## R2: Projekte leiten" in markdown
    assert "S. 2" in markdown
    assert "<script>" not in markdown
    assert r"\[Link](https://example.com)" in markdown
    assert r"\*\*Text\*\*" in markdown
    assert "Python und SQL" in markdown


@pytest.mark.parametrize("text", [
    "Python 3.11, CI-CD (Docker).", "C# und C++", "Array [0] und (Optionen)",
    "snake_case und foo_bar_baz", "99.9 Prozent", "#ohne Leerzeichen",
    "E-Mail: test@example.com", 'Ein "Zitat" und ein Apostroph: geht\'s.',
])
def test_markdown_preserves_ordinary_punctuation(text):
    assert app._markdown_text(text, line_start=True) == text


@pytest.mark.parametrize("text,expected", [
    ("# Überschrift", r"\# Überschrift"),
    ("## Überschrift", r"\## Überschrift"),
    ("- Eintrag", r"\- Eintrag"), ("+ Eintrag", r"\+ Eintrag"),
    ("1. Eintrag", r"1\. Eintrag"), ("2) Eintrag", r"2\) Eintrag"),
    ("---", r"\---"), ("- - -", r"\- - -"),
    ("[Name]: URL", r"\[Name]: URL"),
])
def test_markdown_protects_actual_line_start_syntax(text, expected):
    assert app._markdown_text(text, line_start=True) == expected


@pytest.mark.parametrize("text,expected", [
    ("[Link](https://example.com)", r"\[Link](https://example.com)"),
    ("![Bild](bild.png)", r"!\[Bild](bild.png)"),
    ("[Link][Referenz]", r"\[Link][Referenz]"),
    ("**Text**", r"\*\*Text\*\*"), ("__Text__", r"\_\_Text\_\_"),
    ("~~Text~~", r"\~\~Text\~\~"), ("`Code`", r"\`Code\`"),
])
def test_markdown_prevents_unintended_inline_formatting(text, expected):
    assert app._markdown_text(text) == expected


def test_markdown_line_start_protection_respects_export_context(result):
    result["summary"]["fazit"] = "# Zusammenfassung (kurz)."
    result["requirements"][0]["begruendung"] = "- Gut belegt (S. 1)."
    result["requirements"][0]["text"] = "- Sicherer Umgang (Python)."
    markdown = app.result_markdown(result)
    assert r"\# Zusammenfassung (kurz)." in markdown
    assert r"\- Gut belegt (S. 1)." in markdown
    assert "## R1: - Sicherer Umgang (Python)." in markdown


def test_model_is_cached(monkeypatch):
    app.load_model.clear()
    load = Mock(return_value=object())
    monkeypatch.setattr(app, "get_embedding_model", load)
    try:
        assert app.load_model() is app.load_model()
        load.assert_called_once()
    finally:
        app.load_model.clear()


def test_upload_config_preserves_disabled_watcher():
    config = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf-8-sig"))
    assert config["server"] == {"fileWatcherType": "none", "maxUploadSize": 5}
    assert config["client"]["toolbarMode"] == "minimal"
    assert config["theme"]["base"] == "light"
    assert config["theme"]["primaryColor"] == "#1E40AF"


def test_sorting_and_filters_keep_export_order_and_content(result):
    result["requirements"].extend([
        {"id": "R3", "text": "Docker", "typ": "muss", "status": "nicht_belegt",
         "beleg": None, "begruendung": "Beleg im Dokument nicht gefunden"},
        {"id": "R4", "text": "Git", "typ": "muss", "status": "teilweise",
         "beleg": {"zitat": "Git", "seite": 1}, "begruendung": "Teilweise belegt"},
    ])
    original = copy.deepcopy(result)
    assert [item["id"] for item in app.filtered_requirements(result)] == ["R3", "R1", "R4", "R2"]
    assert [item["id"] for item in app.filtered_requirements(result, "Muss")] == ["R3", "R1", "R4"]
    assert [item["id"] for item in app.filtered_requirements(result, "Lücken")] == ["R3", "R4", "R2"]
    html_table = app.render_requirements_html(result)
    assert "nicht belegt · kritisch" in html_table
    assert "Beleg nicht gefunden" in html_table
    assert "im Dokument verifiziert · S. 2" in html_table
    assert result == original
    assert app.result_markdown(result).index("## R1") < app.result_markdown(result).index("## R3")


def test_filter_reruns_keep_full_result_and_exports(ui, result, monkeypatch):
    at, analyze = ui
    downloads = []
    original_download = app.st.download_button
    def record_download(*args, **kwargs):
        downloads.append((args[0], kwargs["data"]))
        return original_download(*args, **kwargs)
    monkeypatch.setattr(app.st, "download_button", record_download)
    at.button[0].click().run()
    at.button[1].click().run()
    at.get("button_group")[1].set_value("Lücken").run()
    assert not at.exception
    analyze.assert_called_once()
    assert at.session_state["result"] == result
    table = next(item.value for item in at.markdown if 'class="ac-table"' in item.value)
    assert "Python entwickeln" not in table
    assert "Projekte leiten" in table
    assert "## R1" in dict(downloads[-2:])["Markdown"]
    assert json.loads(dict(downloads[-2:])["JSON"]) == result


def test_cost_uses_separate_models_and_cache_categories():
    capture = llm.UsageCapture(by_model={
        "claude-haiku-5-5": llm.TokenUsage(input_tokens=1000, output_tokens=500),
        "claude-sonnet-5-5": llm.TokenUsage(input_tokens=1000, output_tokens=500,
            cache_creation_input_tokens=400, cache_creation_1h_input_tokens=100,
            cache_read_input_tokens=100),
    })
    assert app.analysis_cost_usd(capture) == "0.00851"
    capture.complete = False
    assert app.analysis_cost_usd(capture) is None
    assert app.analysis_cost_usd(llm.UsageCapture()) is None
    assert app.analysis_cost_usd(llm.UsageCapture(by_model={"unknown": llm.TokenUsage()})) is None


def test_analysis_metadata_is_measured_without_changing_result(ui, result, monkeypatch):
    at, analyze = ui
    clock = iter((100.0, 110.6))
    monkeypatch.setattr(app, "perf_counter", lambda: next(clock))
    def measured(*args, **kwargs):
        llm._capture_response_usage("claude-sonnet-5-5", SimpleNamespace(input_tokens=1000, output_tokens=500))
        return copy.deepcopy(result)
    analyze.side_effect = measured
    at.button[0].click().run()
    at.button[1].click().run()
    assert not at.exception
    assert at.session_state["result"] == result
    metadata = at.session_state["analysis_meta"]
    assert metadata["duration_seconds"] == pytest.approx(10.6)
    assert metadata["cost_usd"] == "0.007"
    dashboard = next(item.value for item in at.markdown if 'class="ac-metrics"' in item.value)
    assert "10,6 s" in dashboard
    assert "0,0070 USD" in dashboard


@pytest.mark.parametrize("metadata", [None, {}, {"cost_usd": "invalid"}, {"cost_usd": "NaN"},
                                     {"cost_usd": "Infinity"}, {"cost_usd": "-1"}])
def test_unavailable_cost_is_not_displayed_as_free(metadata):
    assert app.format_cost(metadata) == "—"


def test_filename_and_requirement_id_are_escaped(result):
    attack = '<img src=x onerror="alert(1)">'
    result["requirements"][0]["id"] = attack
    assert attack not in app.render_requirements_html(result)
    assert html.escape(attack, quote=True) in app.render_requirements_html(result)
    assert attack not in app.render_file_note(attack)
    assert app.render_file_note(attack).count(html.escape(attack, quote=True)) == 2


@pytest.mark.parametrize("code", [None, "wrong", ""])
def test_public_view_has_demo_but_no_upload_model_or_analysis(monkeypatch, live_access, code, caplog):
    model, analyze = Mock(), Mock()
    monkeypatch.setattr(app, "load_model", model)
    monkeypatch.setattr(app.pipeline, "analyze", analyze)
    at = AppTest.from_string("import app\napp.main()")
    if code is not None:
        at.query_params["code"] = code
    at.run()
    assert not at.exception
    assert not at.error
    assert at.info[0].value == app.INVITATION_NOTICE
    assert not at.get("file_uploader")
    assert at.button[0].disabled
    assert len(at.get("download_button")) == 2
    assert any("Beispielergebnis" in item.value for item in at.markdown)
    assert any("68" in item.value and 'class="ac-metrics"' in item.value for item in at.markdown)
    at.button[0].click().run()  # Auch ein simulierter Klick darf nichts starten.
    at.get("button_group")[0].set_value("Lücken").run()
    assert not at.exception
    assert at.get("button_group")[0].value == "Lücken"
    analyze.assert_not_called()
    model.assert_not_called()
    assert "test-invitation" not in caplog.text
    assert "test-api-key" not in caplog.text
    assert "result" not in at.session_state
    assert app.get_limits().daily_count == 0


def test_valid_link_cannot_unlock_without_environment_code(monkeypatch, live_access):
    monkeypatch.delenv("ACCESS_CODE")
    model = Mock()
    monkeypatch.setattr(app, "load_model", model)
    at = live_app()
    assert not at.exception
    assert at.button[0].disabled
    assert not at.get("file_uploader")
    model.assert_not_called()


def test_invitation_survives_rerun_but_revocation_clears_private_result(ui, monkeypatch):
    at, analyze = ui
    at.query_params.clear()
    at.run()
    assert len(at.get("file_uploader")) == 2
    at.button[0].click().run()
    at.button[1].click().run()
    assert "result" in at.session_state
    monkeypatch.delenv("ACCESS_CODE")
    at.run()
    assert not at.exception
    assert "result" not in at.session_state
    assert "analysis_meta" not in at.session_state
    assert not at.get("file_uploader")
    assert any("Beispielergebnis" in item.value for item in at.markdown)
    analyze.assert_called_once()


def test_five_live_analyses_then_disabled_with_exact_limit_message(ui, result):
    at, analyze = ui
    def started(*args, **kwargs):
        before_api_request(None)
        return copy.deepcopy(result)
    analyze.side_effect = started
    at.button[0].click().run()
    for _ in range(5):
        at.button[1].click().run()
        assert not at.exception
    assert at.session_state["access_session"].analyses == 5
    assert app.get_limits().daily_count == 5
    assert at.button[1].disabled
    assert any(item.value == "Das Analyse-Limit für diese Sitzung ist erreicht. Danke fürs Ausprobieren!" for item in at.info)
    at.button[1].click().run()
    assert analyze.call_count == 5
    assert at.session_state["result"] == result


def test_bad_pdf_and_missing_api_key_do_not_consume_quota(monkeypatch, live_access, caplog):
    from copilot.access import AnalysisLimits, SessionAccess
    session, limits = SessionAccess(), AnalysisLimits()
    session.refresh("test-invitation")
    monkeypatch.setattr(app, "access_session", lambda: session)
    monkeypatch.setattr(app, "get_limits", lambda: limits)
    for _ in range(3):
        with pytest.raises(PdfExtractionError):
            app.run_live_analysis(BytesIO(b"bad PDF"), BytesIO(b"bad PDF"), "job", False, on_progress=lambda _: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(llm.MissingApiKeyError):
        app.run_live_analysis(None, None, "job", True, on_progress=lambda _: None)
    assert session.analyses == limits.daily_count == 0
    assert "bad P" not in caplog.text


def test_server_gate_blocks_direct_start_without_authorization(monkeypatch, live_access):
    from copilot.access import AccessDenied, SessionAccess
    monkeypatch.setattr(app, "access_session", lambda: SessionAccess())
    reader, analyze = Mock(), Mock()
    monkeypatch.setattr(app, "read_documents", reader)
    monkeypatch.setattr(app.pipeline, "analyze", analyze)
    with pytest.raises(AccessDenied):
        app.run_live_analysis(None, None, "job", True, on_progress=lambda _: None)
    reader.assert_not_called()
    analyze.assert_not_called()


def test_example_schema_error_is_friendly_and_does_not_disclose_contents(monkeypatch, live_access, caplog):
    def invalid_example():
        return app.AnalysisResult.model_validate({"PRIVATE_FIELD": "PRIVATE_DOCUMENT"})
    monkeypatch.setattr(app, "load_example_result", invalid_example)
    at = AppTest.from_string("import app\napp.main()").run()
    assert not at.exception
    assert "Beispielergebnis" in at.error[0].value
    assert "PRIVATE" not in caplog.text
    assert "ValidationError" in caplog.text


def test_shared_limit_manager_is_cached():
    assert app.get_limits() is app.get_limits()


def test_api_failures_after_start_consume_quota(ui, caplog):
    at, analyze = ui
    def failed(*args, **kwargs):
        before_api_request(None)
        raise llm.LlmApiError("timeout")
    analyze.side_effect = failed
    at.button[0].click().run()
    for _ in range(5):
        at.button[1].click().run()
        assert not at.exception
        assert "zu lange" in at.error[0].value
    assert at.session_state["access_session"].analyses == 5
    assert app.get_limits().daily_count == 5
    assert any("Analyse-Limit für diese Sitzung" in item.value for item in at.info)
    at.run()
    assert at.button[1].disabled
    assert "test-invitation" not in caplog.text
    assert "test-api-key" not in caplog.text


def test_global_daily_limit_blocks_another_session(ui):
    from copilot.access import SessionAccess
    at, analyze = ui
    limits = app.get_limits()
    for _ in range(50):
        session = SessionAccess()
        session.refresh("test-invitation")
        with limits.analysis(session):
            before_api_request(None)
    at.button[0].click().run()
    assert at.button[1].disabled
    assert any("tägliche Analyse-Limit" in item.value for item in at.info)
    at.button[1].click().run()
    analyze.assert_not_called()
    assert at.session_state["access_session"].analyses == 0


def test_ambiguous_invitation_link_is_locked(monkeypatch, live_access):
    model = Mock()
    monkeypatch.setattr(app, "load_model", model)
    at = AppTest.from_string("import app\napp.main()")
    at.query_params["code"] = ["test-invitation", "wrong"]
    at.run()
    assert not at.exception
    assert not at.get("file_uploader")
    assert at.button[0].disabled
    model.assert_not_called()
