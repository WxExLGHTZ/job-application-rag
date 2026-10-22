import copy
import hashlib
from io import BytesIO
import json
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import app
from copilot.access import before_api_request
from copilot.ingest import extract_pages_from_pdf, build_page_chunks
from copilot.pipeline import verify_quotes, calculate_summary
from copilot.schema import AnalysisResult


@pytest.mark.parametrize("mode", ["job", "tender"])
def test_saved_example_is_valid_and_metrics_match_exactly(mode):
    result = AnalysisResult.model_validate(app.load_example_result(mode))
    metadata = app.load_example_metadata(mode)
    assert metadata and metadata["duration_seconds"] > 0
    assert float(metadata["cost_usd"]) > 0
    assert set(metadata["usage_by_model"]) == {"claude-haiku-5-5", "claude-sonnet-5-5"}
    pages = extract_pages_from_pdf(app.example_paths(mode)[1].read_bytes())
    assert verify_quotes(result.requirements, pages, build_page_chunks(pages)) == result.requirements
    assert calculate_summary(result.requirements, result.summary.fazit) == result.summary
    if mode == "tender":
        iso = [item for item in result.requirements if "27001" in item.text]
        assert len(iso) == 1 and iso[0].typ == "muss" and iso[0].status == "nicht_belegt" and iso[0].beleg is None
        for item in result.requirements:
            assert not any(word in item.text for word in ("Zuschlagskriterien", "Angebotsfrist", "Bieterfragen", "Vergabeplattform"))


def test_stale_metadata_is_not_shown(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "EXAMPLES_DIR", tmp_path)
    (tmp_path / "result_job.json").write_text('{}', encoding="utf-8")
    meta = {"mode": "job", "result_sha256": "old", "duration_seconds": 10}
    path = tmp_path / "result_job.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert app.load_example_metadata("job") is None
    meta["result_sha256"] = hashlib.sha256(b'{}').hexdigest()
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert app.load_example_metadata("job") == meta


def test_example_measurements_survive_windows_git_line_endings(tmp_path, monkeypatch):
    data = (app.EXAMPLES_DIR / "result_job.json").read_bytes()
    metadata = (app.EXAMPLES_DIR / "result_job.meta.json").read_bytes()
    (tmp_path / "result_job.json").write_bytes(data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    (tmp_path / "result_job.meta.json").write_bytes(metadata)
    monkeypatch.setattr(app, "EXAMPLES_DIR", tmp_path)
    assert app.load_example_metadata("job") is not None


def test_public_selector_downloads_original_pdfs_without_api(monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    analyze, model = Mock(), Mock()
    monkeypatch.setattr(app.pipeline, "analyze", analyze)
    monkeypatch.setattr(app, "load_model", model)
    downloads = []
    original = app.st.download_button
    def record(label, **kwargs):
        downloads.append((label, kwargs["data"]))
        return original(label, **kwargs)
    monkeypatch.setattr(app.st, "download_button", record)
    at = AppTest.from_string("import app\napp.main()").run()
    assert not at.exception and not at.error
    assert at.get("button_group")[0].value == "tender"
    assert any(app.SAVED_EXAMPLE_NOTICE in item.value for item in at.caption)
    assert dict(downloads)["Ausschreibung ansehen"] == app.example_paths("tender")[0].read_bytes()
    assert dict(downloads)["Firmenprofil ansehen"] == app.example_paths("tender")[1].read_bytes()
    assert any("gemessen beim Erstellen des Beispiels" in item.value for item in at.markdown)
    at.get("button_group")[1].set_value("Lücken").run()
    at.get("button_group")[0].set_value("job").run()
    assert not at.exception and at.get("button_group")[1].value == "Alle"
    assert dict(downloads)["Stellenanzeige ansehen"] == app.example_paths("job")[0].read_bytes()
    assert not at.get("file_uploader")
    analyze.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("mode", ["job", "tender"])
def test_load_example_starts_one_live_analysis_with_limits_and_progress(monkeypatch, mode):
    monkeypatch.setenv("ACCESS_CODE", "test-invitation")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(app, "load_model", lambda: None)
    app.get_limits.clear()
    expected = app.load_example_result(mode)
    def analyze(*args, on_progress):
        assert args == (*[path.read_bytes() for path in app.example_paths(mode)], mode)
        before_api_request(None)
        for stage in app.STAGES:
            on_progress(stage)
        return copy.deepcopy(expected)
    mocked = Mock(side_effect=analyze)
    monkeypatch.setattr(app.pipeline, "analyze", mocked)
    try:
        at = AppTest.from_string("import app\napp.main()")
        at.query_params["code"] = "test-invitation"
        at.run()
        at.get("button_group")[0].set_value(mode).run()
        at.button(key="example_button").click().run()
        assert not at.exception and not at.error
        mocked.assert_called_once()
        assert at.session_state["result"] == expected
        assert at.session_state["access_session"].analyses == app.get_limits().daily_count == 1
        assert at.status[0].state == "complete"
        assert not at.get("file_uploader")
        assert len([button for button in at.button if button.label == "Eigene Datei verwenden"]) == 2
        at.run()
        mocked.assert_called_once()  # Kein erneuter Start durch Rerun.
        at.button(key="own_document_0").click().run()
        assert at.session_state["use_example"] == (False, True)
        assert len(at.get("file_uploader")) == 1
        assert "result" not in at.session_state
        assert at.button(key="analyze_button").disabled
        # Session-Limit gilt auch für den automatischen Beispielstart.
        for _ in range(4):
            with app.get_limits().analysis(at.session_state["access_session"]):
                before_api_request(None)
        at.button(key="example_button").click().run()
        assert not at.exception
        assert any("Analyse-Limit für diese Sitzung" in info.value for info in at.info)
        mocked.assert_called_once()
    finally:
        app.get_limits.clear()


@pytest.mark.parametrize("mode", ["job", "tender"])
@pytest.mark.parametrize("side", [0, 1])
def test_mixed_example_and_uploaded_pdf(mode, side):
    sources = (side == 0, side == 1)
    documents = app.read_documents(BytesIO(b"upload-a"), BytesIO(b"upload-b"), sources, mode)
    assert documents[side] == app.example_paths(mode)[side].read_bytes()
    assert documents[1-side] == (b"upload-b" if side == 0 else b"upload-a")
