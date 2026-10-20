import builtins
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from copilot import cli, index, ingest, llm, pipeline


@pytest.fixture
def offline_embeddings(monkeypatch):
    def embed(texts):
        return np.asarray([
            [text.count("Python") + 0.01, text.count("SQL") + 0.01] for text in texts
        ], dtype=np.float32)
    monkeypatch.setattr(pipeline, "embed_texts", embed)
    monkeypatch.setattr(index, "embed_texts", embed)
    return embed


def test_retrieval_keeps_correct_page_and_id(offline_embeddings):
    chunks = ingest.build_page_chunks([
        {"page": 1, "text": "Python " * 20}, {"page": 3, "text": "SQL " * 30},
    ])
    profile_index = index.ProfileIndex(chunks, offline_embeddings([c["text"] for c in chunks]))
    hits = profile_index.retrieve("SQL", top_k=1)
    assert len(hits) == 1
    assert hits[0]["page"] == 3
    assert hits[0]["chunk_id"] == chunks[1]["chunk_id"]
    assert hits[0]["score"] > .99


@pytest.mark.parametrize("mode", ["job", "tender"])
def test_pipeline_complete_without_files_or_network(pdf_bytes, offline_embeddings, monkeypatch, mode):
    document = pdf_bytes(["Python SQL Anforderungen " * 20])
    profile = pdf_bytes(["Python Erfahrung " * 20, "SQL Kenntnisse " * 20])
    requests = []
    model_result = {
        "requirements": [
            {"id": "R1", "status": "erfuellt", "beleg": {"zitat": "Python Erfahrung", "seite": 2}, "begruendung": "Erfahrung vorhanden"},
            {"id": "R2", "status": "erfuellt", "beleg": {"zitat": "Erfundene SQL Erfahrung", "seite": 2}, "begruendung": "Kenntnisse vorhanden"},
        ], "fazit": "Kenntnisse sind vorhanden.",
    }
    def create(**kwargs):
        requests.append(kwargs)
        text = json.dumps([{"text": "Python", "typ": "muss"}, {"text": "SQL", "typ": "kann"}]) if len(requests) == 1 else json.dumps(model_result)
        return SimpleNamespace(
            content=[SimpleNamespace(text=text)], stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )
    monkeypatch.setattr(llm, "_CLIENT", SimpleNamespace(messages=SimpleNamespace(create=create)))
    def forbidden(*args, **kwargs):
        raise AssertionError("Kein Dateizugriff oder globaler Profilindex erlaubt")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(index, "retrieve", forbidden)
    monkeypatch.setattr(index, "build_index", forbidden)
    monkeypatch.setattr(ingest, "load_profile", forbidden)
    monkeypatch.setattr(index, "_DEFAULT_INDEX", None)

    result = pipeline.analyze(document, profile, mode)
    assert set(result) == {"mode", "summary", "requirements", "usage"}
    assert result["mode"] == mode
    assert result["summary"] == {"score": 67, "muss_erfuellt": "1/1", "fazit": model_result["fazit"]}
    assert result["requirements"][0]["beleg"]["seite"] == 1
    assert result["requirements"][0]["text"] == "Python"
    assert result["requirements"][1]["status"] == "nicht_belegt"
    assert result["requirements"][1]["beleg"] is None
    assert result["usage"] == {"input_tokens": 22, "output_tokens": 14}
    assert index._DEFAULT_INDEX is None
    payload = json.loads(requests[1]["messages"][0]["content"])["dokumentdaten"]
    assert {c["page"] for c in payload["chunks"]} == {1, 2}
    assert len(payload["chunks"]) == 2


def test_parallel_profiles_and_usage_are_isolated(pdf_bytes, offline_embeddings, monkeypatch):
    document = pdf_bytes(["Anforderungen " * 20])
    profiles = [pdf_bytes(["Python " * 25]), pdf_bytes(["SQL " * 40])]
    monkeypatch.setattr(pipeline, "extract_requirements", lambda *a, **kw: [{"id": "R1", "text": "SQL", "typ": "muss"}])
    def assess(requirements, mode, *, usage):
        text = requirements[0]["retrieved_chunks"][0]["text"]
        usage.add_response(SimpleNamespace(input_tokens=len(text), output_tokens=2))
        return {"fazit": text, "requirements": [{"id": "R1", "status": "nicht_belegt", "beleg": None, "begruendung": "Kein Beleg"}]}
    monkeypatch.setattr(pipeline, "assess_requirements", assess)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda profile: pipeline.analyze(document, profile, "job"), profiles))
    assert "Python" in results[0]["summary"]["fazit"] and "SQL" not in results[0]["summary"]["fazit"]
    assert "SQL" in results[1]["summary"]["fazit"] and "Python" not in results[1]["summary"]["fazit"]
    for result in results:
        assert result["usage"] == {"input_tokens": len(result["summary"]["fazit"]), "output_tokens": 2}


def test_documents_checked_before_embedding_or_api(pdf_bytes, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Nicht vor PDF-Prüfung aufrufen"))
    monkeypatch.setattr(pipeline, "embed_texts", forbidden)
    monkeypatch.setattr(pipeline, "extract_requirements", forbidden)
    with pytest.raises(ingest.PdfWithoutText):
        pipeline.analyze(pdf_bytes(["Anforderung " * 20]), pdf_bytes([""]), "job")
    with pytest.raises(ValueError, match="Modus"):
        pipeline.analyze(b"", b"", "invalid")
    forbidden.assert_not_called()


def test_progress_matches_actual_pipeline_steps(pdf_bytes, offline_embeddings, monkeypatch):
    events = []
    original_read = pipeline.extract_pages_from_pdf
    original_retrieve = index.ProfileIndex.retrieve

    def read(data):
        events.append("read")
        return original_read(data)

    def extract(*args, **kwargs):
        events.append("extract")
        return [{"id": "R1", "text": "Python", "typ": "muss"}]

    def retrieve(self, *args, **kwargs):
        events.append("retrieve")
        return original_retrieve(self, *args, **kwargs)

    def assess(*args, **kwargs):
        events.append("assess")
        return {"fazit": "Kein Beleg.", "requirements": [
            {"id": "R1", "status": "nicht_belegt", "beleg": None, "begruendung": "Kein Beleg"},
        ]}

    monkeypatch.setattr(pipeline, "extract_pages_from_pdf", read)
    monkeypatch.setattr(pipeline, "extract_requirements", extract)
    monkeypatch.setattr(index.ProfileIndex, "retrieve", retrieve)
    monkeypatch.setattr(pipeline, "assess_requirements", assess)
    document = pdf_bytes(["Python Anforderung " * 20])
    pipeline.analyze(document, document, "job", on_progress=events.append)
    assert events == ["text", "read", "read", "requirements", "extract",
                      "evidence", "retrieve", "assessment", "assess"]


def test_progress_stops_after_bad_pdf(pdf_bytes):
    events = []
    with pytest.raises(ingest.PdfWithoutText):
        pipeline.analyze(pdf_bytes([""]), pdf_bytes([""]), "job", on_progress=events.append)
    assert events == ["text"]


def test_cli_reads_bytes_and_prints_only_json(pdf_bytes, monkeypatch, capsys):
    data = pdf_bytes(["Anforderungen " * 20])
    monkeypatch.setattr(cli.Path, "read_bytes", lambda self: data)
    expected = {"mode": "tender", "summary": {"score": 0, "muss_erfuellt": "0/0", "fazit": "Kein Nachweis"},
                "requirements": [], "usage": {"input_tokens": 10, "output_tokens": 5}}
    analyze = Mock(return_value=expected)
    monkeypatch.setattr(cli, "analyze", analyze)
    assert cli.main(["requirements.pdf", "profile.pdf", "--mode", "tender"]) == 0
    analyze.assert_called_once_with(data, data, "tender")
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert captured.err == ""


def test_cli_file_errors_do_not_print_paths(monkeypatch, capsys):
    def unreadable(self):
        raise PermissionError("PRIVATER_PFAD")
    monkeypatch.setattr(cli.Path, "read_bytes", unreadable)
    assert cli.main(["requirements.pdf", "profile.pdf"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PermissionError" in captured.err
    assert "PRIVATER_PFAD" not in captured.err


def test_cli_pdf_error_and_invalid_mode(monkeypatch, capsys):
    assert cli.main(["requirements.txt", "profile.pdf"]) == 1
    assert "PDF" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        cli.main(["requirements.pdf", "profile.pdf", "--mode", "other"])
    assert error.value.code == 2


def test_cli_full_result_after_reconciliation_with_retry_usage(
    pdf_bytes, offline_embeddings, monkeypatch, capsys, caplog,
):
    document = pdf_bytes(["Python und SQL erforderlich " * 20])
    profile = pdf_bytes(["Python Erfahrung " * 20])
    monkeypatch.setattr(
        cli.Path, "read_bytes",
        lambda path: document if path.name == "requirements.pdf" else profile,
    )
    first = {"id": "R1", "status": "erfuellt",
             "beleg": {"zitat": "Python Erfahrung", "seite": 1},
             "begruendung": "Python ist belegt"}
    duplicate = {"id": "R1", "status": "nicht_belegt",
                 "beleg": None, "begruendung": "Spätere doppelte Bewertung"}
    responses = [
        [{"text": "Python", "typ": "muss"}, {"text": "SQL", "typ": "muss"}],
        {"requirements": [first], "fazit": "Ein Nachweis fehlt."},
        {"requirements": [first, duplicate, {**duplicate, "id": "R99"}],
         "fazit": "Ein Nachweis fehlt."},
    ]
    messages = Mock()
    messages.create.side_effect = [
        SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps(data))], stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  cache_creation_input_tokens=2, cache_read_input_tokens=3),
        ) for data in responses
    ]
    monkeypatch.setattr(llm, "_CLIENT", SimpleNamespace(messages=messages))
    assert cli.main(["requirements.pdf", "profile.pdf", "--mode", "job"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "mode": "job",
        "summary": {"score": 50, "muss_erfuellt": "1/2", "fazit": "Ein Nachweis fehlt."},
        "requirements": [
            {**first, "text": "Python", "typ": "muss"},
            {"id": "R2", "text": "SQL", "typ": "muss", "status": "nicht_belegt",
             "beleg": None, "begruendung": "Nicht bewertet"},
        ],
        "usage": {"input_tokens": 45, "output_tokens": 15},
    }
    assert messages.create.call_count == 3
    assert "Python Erfahrung" not in caplog.text
