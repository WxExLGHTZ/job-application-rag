import copy

import pytest
from pydantic import ValidationError

from copilot.ingest import build_page_chunks
from copilot.pipeline import calculate_summary, normalize_quote, verify_quotes
from copilot.schema import AnalysisResult, AssessmentResponse, Requirement


def requirement(**changes):
    data = {
        "id": "R1", "text": "Erfahrung mit Python", "typ": "muss",
        "status": "erfuellt", "beleg": {"zitat": "Python, PyTorch, FastAPI", "seite": 1},
        "begruendung": "Erfahrung vorhanden",
    }
    data.update(changes)
    return data


def result():
    return {
        "mode": "job",
        "summary": {"score": 100, "muss_erfuellt": "1/1", "fazit": "Anforderung erfüllt."},
        "requirements": [requirement()],
        "usage": {"input_tokens": 18000, "output_tokens": 1900},
    }


def test_exact_result_roundtrip():
    assert AnalysisResult.model_validate(result()).model_dump() == result()
    assert set(AssessmentResponse.model_fields) == {"requirements", "fazit"}


@pytest.mark.parametrize("changes", [
    {"status": "luecke"}, {"typ": "aufgabe"}, {"id": " "}, {"text": ""},
    {"begruendung": " "}, {"status": "nicht_belegt"}, {"beleg": None},
    {"beleg": {"zitat": "Python", "seite": 0}},
    {"beleg": {"zitat": "Python", "seite": "1"}},
    {"beleg": {"zitat": "Python", "seite": True}},
    {"beleg": {"zitat": "Python", "seite": 1.5}},
    {"beleg": {"zitat": "", "seite": 1}},
    {"beleg": {"zitat": "Python ... FastAPI", "seite": 1}},
    {"beleg": {"zitat": "Python … FastAPI", "seite": 1}},
    {"score": 1},
])
def test_invalid_requirement_rejected(changes):
    with pytest.raises(ValidationError):
        Requirement.model_validate(requirement(**changes))


@pytest.mark.parametrize("section,changes", [
    ("summary", {"score": -1}), ("summary", {"score": 101}),
    ("summary", {"score": 78.5}), ("summary", {"score": True}),
    ("summary", {"muss_erfuellt": "eins"}),
    ("usage", {"input_tokens": -1}), ("usage", {"output_tokens": "5"}),
    ("usage", {"total_tokens": 10}),
])
def test_invalid_summary_and_usage_rejected(section, changes):
    data = result()
    data[section].update(changes)
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)


def test_final_schema_rejects_missing_extra_and_duplicate_fields():
    for field in result():
        data = result()
        del data[field]
        with pytest.raises(ValidationError):
            AnalysisResult.model_validate(data)
    data = result()
    data["requirements"].append(copy.deepcopy(data["requirements"][0]))
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)
    data = result()
    data["mode"] = "other"
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)
    data = result()
    data["anforderungs_matrix"] = []
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)


def check_quote(quote, page, pages, chunks=None):
    item = Requirement.model_validate(requirement(beleg={"zitat": quote, "seite": page}))
    return verify_quotes([item], pages, chunks if chunks is not None else build_page_chunks(pages))[0]


def test_quotes_normalize_case_and_whitespace():
    pages = [{"page": 1, "text": "Python,\n  PyTorch,\tFastAPI"}]
    checked = check_quote("PYTHON, PyTorch, FastAPI", 1, pages)
    assert checked.status == "erfuellt"
    assert checked.beleg.seite == 1


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_quote_dehyphenation_at_line_end(newline):
    pages = [{"page": 2, "text": f"Erfahrung mit Daten-{newline}banken und SQL"}]
    checked = check_quote("Erfahrung mit Datenbanken und SQL", 2, pages)
    assert checked.status == "erfuellt"
    assert normalize_quote(f"Daten-{newline}banken") == "datenbanken"
    assert normalize_quote("Client-Server") == "client-server"


def test_invented_quote_is_downgraded():
    checked = check_quote("Zehn Jahre Python", 1, [{"page": 1, "text": "SQL Grundlagen"}])
    assert checked.status == "nicht_belegt"
    assert checked.beleg is None
    assert checked.begruendung == "Beleg im Dokument nicht gefunden"


def test_wrong_page_corrected_and_valid_duplicate_page_retained():
    pages = [{"page": 1, "text": "SQL"}, {"page": 2, "text": "Python"},
             {"page": 3, "text": "Python"}]
    assert check_quote("Python", 1, pages).beleg.seite == 2
    assert check_quote("Python", 3, pages).beleg.seite == 3


def test_quote_cannot_span_pages_or_chunks():
    assert check_quote("Python SQL", 1, [
        {"page": 1, "text": "Python"}, {"page": 2, "text": "SQL"},
    ]).status == "nicht_belegt"
    assert check_quote("Python SQL", 1, [{"page": 1, "text": "Python SQL"}], [
        {"page": 1, "text": "Python"}, {"page": 1, "text": "SQL"},
    ]).status == "nicht_belegt"


def test_generated_chunk_prefix_is_not_original_evidence():
    checked = check_quote("Studienmodul: Python", 1, [{"page": 1, "text": "Python SS2024 1,0"}], [
        {"page": 1, "text": "Studienmodul: Python. Originalzeile: Python SS2024 1,0"},
    ])
    assert checked.status == "nicht_belegt"


@pytest.mark.parametrize("items,score,must", [
    ([], 0, "0/0"),
    ([("muss", "erfuellt")], 100, "1/1"),
    ([("kann", "nicht_belegt")], 0, "0/0"),
    ([("muss", "teilweise")], 50, "0/1"),
    ([("muss", "erfuellt"), ("kann", "nicht_belegt")], 67, "1/1"),
    ([("muss", "teilweise"), ("kann", "erfuellt")], 67, "0/1"),
    ([("muss", "erfuellt"), ("muss", "nicht_belegt"), ("kann", "teilweise")], 50, "1/2"),
    ([("kann", "teilweise")] + [("kann", "nicht_belegt")] * 3, 13, "0/0"),
])
def test_weighted_score_and_rounding(items, score, must):
    requirements = [
        Requirement.model_validate(requirement(
            id=f"R{number}", typ=typ, status=status,
            beleg=None if status == "nicht_belegt" else {"zitat": "Python", "seite": 1},
        ))
        for number, (typ, status) in enumerate(items, start=1)
    ]
    assert calculate_summary(requirements, "Fazit").model_dump() == {
        "score": score, "muss_erfuellt": must, "fazit": "Fazit",
    }
