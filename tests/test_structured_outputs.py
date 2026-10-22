import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from anthropic import transform_schema
from pydantic import TypeAdapter

from copilot import llm
from copilot.schema import AssessmentResponse, ExtractedRequirement


def evaluate(monkeypatch, data, ids=("R1", "R2")):
    messages = Mock()
    messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps(data))], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    monkeypatch.setattr(llm, "_CLIENT", SimpleNamespace(messages=messages))
    result = llm.assess_requirements([
        {"id": id, "text": "Test", "typ": "muss", "retrieved_chunks": []}
        for id in ids
    ])
    assert messages.create.call_count == 1
    AssessmentResponse.model_validate(result)
    return result, messages


def valid(id="R1"):
    return {"id": id, "status": "erfuellt", "beleg": {"zitat": "Python", "seite": 1},
            "begruendung": "Belegt"}


@pytest.mark.parametrize("status", ["erfuellt", "teilweise"])
@pytest.mark.parametrize("evidence", [
    None, {}, "Python", {"zitat": "", "seite": 1}, {"zitat": "Python", "seite": 0},
    {"zitat": "Python", "seite": "1"}, {"zitat": "Python", "seite": True},
    {"zitat": "Python ... SQL", "seite": 1}, {"zitat": "Python … SQL", "seite": 1},
    {"zitat": "Python"}, {"seite": 1},
])
def test_bad_evidence_repairs_only_affected_requirement(monkeypatch, status, evidence):
    item = {**valid(), "status": status, "beleg": evidence}
    data = {"requirements": [item, valid("R2")], "fazit": "Ergebnis"}
    original = deepcopy(data)
    result, _ = evaluate(monkeypatch, data)
    assert result["requirements"][0] == {
        "id": "R1", "status": "nicht_belegt", "beleg": None,
        "begruendung": "Beleg im Dokument nicht gefunden",
    }
    assert result["requirements"][1] == valid("R2")
    assert data == original


def test_missing_evidence_and_nonmatching_status(monkeypatch):
    first = valid()
    del first["beleg"]
    second = {**valid("R2"), "status": "nicht_belegt"}
    result, _ = evaluate(monkeypatch, {"requirements": [first, second], "fazit": "Ergebnis"})
    assert result["requirements"][0]["status"] == "nicht_belegt"
    assert result["requirements"][1] == {**second, "beleg": None}


@pytest.mark.parametrize("status", ["UNKNOWN_PRIVATE", "ERFUELLT", None, [], {}, 1])
def test_unknown_status_is_not_evidence(monkeypatch, status):
    result, _ = evaluate(monkeypatch, {
        "requirements": [{**valid(), "status": status}, valid("R2")], "fazit": "Ergebnis",
    })
    assert result["requirements"][0]["status"] == "nicht_belegt"
    assert result["requirements"][0]["beleg"] is None
    assert result["requirements"][1] == valid("R2")


@pytest.mark.parametrize("items", [None, {}, "PRIVATE", 4, [None, [], {}, {"id": []}]])
def test_malformed_entries_become_missing_assessments(monkeypatch, items):
    result, _ = evaluate(monkeypatch, {"requirements": items, "fazit": "Ergebnis"})
    assert [r["id"] for r in result["requirements"]] == ["R1", "R2"]
    assert all(r["begruendung"] == "Nicht bewertet" for r in result["requirements"])


def test_invalid_first_duplicate_is_kept_as_downgraded(monkeypatch):
    result, _ = evaluate(monkeypatch, {
        "requirements": [{**valid(), "beleg": None}, valid(), valid("R99"), valid("R2")],
        "fazit": "Ergebnis", "PRIVATE_EXTRA": "PRIVATE_VALUE",
    })
    assert result["requirements"][0]["status"] == "nicht_belegt"
    assert result["requirements"][1] == valid("R2")
    assert set(result) == {"requirements", "fazit"}


@pytest.mark.parametrize("reason", [None, "", " ", [], 123])
def test_unusable_reason_is_repaired_conservatively(monkeypatch, reason):
    result, _ = evaluate(monkeypatch, {
        "requirements": [{**valid(), "begruendung": reason}, valid("R2")], "fazit": "Ergebnis",
    })
    assert result["requirements"][0] == {
        "id": "R1", "status": "nicht_belegt", "beleg": None, "begruendung": "Nicht bewertet",
    }
    assert result["requirements"][1] == valid("R2")


def test_both_stages_send_transformed_schema_and_preserve_effort(monkeypatch):
    _, messages = evaluate(monkeypatch, {"requirements": [valid()], "fazit": "Ergebnis"}, ("R1",))
    assess = messages.create.call_args.kwargs
    messages.create.return_value.content[0].text = '[{"text":"Python","typ":"muss"}]'
    llm.extract_requirements("Python")
    extract = messages.create.call_args.kwargs
    for call, schema in (
        (assess, transform_schema(AssessmentResponse)),
        (extract, transform_schema(TypeAdapter(list[ExtractedRequirement]).json_schema())),
    ):
        assert call["output_config"] == {
            "effort": "low", "format": {"type": "json_schema", "schema": schema},
        }
        assert call["thinking"] == {"type": "between_tools" if call is assess else "disabled"}
        assert call["timeout"] == 60

        def check(node):
            if isinstance(node, dict):
                assert not ({"minLength", "maximum", "minimum"} & node.keys())
                if node.get("type") == "object":
                    assert node["additionalProperties"] is False
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)
        check(schema)


def test_schema_is_included_in_token_budget(monkeypatch):
    messages = Mock()
    monkeypatch.setattr(llm, "_CLIENT", SimpleNamespace(messages=messages))
    schema = transform_schema(AssessmentResponse)
    text = "x" * (99_999 - llm.estimate_request_tokens("", ""))
    assert llm.estimate_request_tokens("", text, schema) >= 100_000
    with pytest.raises(llm.RequestTooLarge):
        llm._send_message("", text, "test", output_schema=schema)
    messages.create.assert_not_called()
