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
    original_result = {
        "anforderungs_matrix": [], "framing_vorschlaege": [],
        "hervorzuhebende_projekte": [], "wahrscheinliche_interviewfragen": [],
    }
    def create(**kwargs):
        requests.append(kwargs)
        text = json.dumps([{"requirement": "Python"}, {"requirement": "SQL"}]) if len(requests) == 1 else json.dumps(original_result)
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
    assert {k: v for k, v in result.items() if k != "usage"} == original_result
    assert result["usage"]["total"]["total_tokens"] == 36
    assert result["usage"]["extract"]["response_count"] == 1
    assert index._DEFAULT_INDEX is None
    payload = json.loads(requests[1]["messages"][0]["content"])["dokumentdaten"]
    assert {c["page"] for c in payload["chunks"]} == {1, 2}
    assert len(payload["chunks"]) == 2


def test_parallel_profiles_and_usage_are_isolated(pdf_bytes, offline_embeddings, monkeypatch):
    document = pdf_bytes(["Anforderungen " * 20])
    profiles = [pdf_bytes(["Python " * 25]), pdf_bytes(["SQL " * 40])]
    monkeypatch.setattr(pipeline, "extract_requirements", lambda *a, **kw: [{"requirement": "SQL"}])
    def assess(requirements, mode, *, usage):
        text = requirements[0]["retrieved_chunks"][0]["text"]
        usage.add_response(SimpleNamespace(input_tokens=len(text), output_tokens=2))
        return {"source": text}
    monkeypatch.setattr(pipeline, "assess_requirements", assess)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda profile: pipeline.analyze(document, profile, "job"), profiles))
    assert "Python" in results[0]["source"] and "SQL" not in results[0]["source"]
    assert "SQL" in results[1]["source"] and "Python" not in results[1]["source"]
    for result in results:
        assert result["usage"]["total"]["response_count"] == 1
        assert result["usage"]["total"]["input_tokens"] == len(result["source"])


def test_documents_checked_before_embedding_or_api(pdf_bytes, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Nicht vor PDF-Prüfung aufrufen"))
    monkeypatch.setattr(pipeline, "embed_texts", forbidden)
    monkeypatch.setattr(pipeline, "extract_requirements", forbidden)
    with pytest.raises(ingest.PdfWithoutText):
        pipeline.analyze(pdf_bytes(["Anforderung " * 20]), pdf_bytes([""]), "job")
    with pytest.raises(ValueError, match="Modus"):
        pipeline.analyze(b"", b"", "invalid")
    forbidden.assert_not_called()


def test_cli_reads_bytes_and_prints_only_json(pdf_bytes, monkeypatch, capsys):
    data = pdf_bytes(["Anforderungen " * 20])
    monkeypatch.setattr(cli.Path, "read_bytes", lambda self: data)
    analyze = Mock(return_value={"anforderungs_matrix": [], "usage": {}})
    monkeypatch.setattr(cli, "analyze", analyze)
    assert cli.main(["requirements.pdf", "profile.pdf", "--mode", "tender"]) == 0
    analyze.assert_called_once_with(data, data, "tender")
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"anforderungs_matrix": [], "usage": {}}
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
