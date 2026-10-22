import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from copilot import cli, llm


def test_cli_saves_result_and_matching_measurements(tmp_path, monkeypatch, capsys):
    requirements, profile = tmp_path / "requirements.pdf", tmp_path / "profile.pdf"
    requirements.write_bytes(b"requirements")
    profile.write_bytes(b"profile")
    result_path, meta_path = tmp_path / "result.json", tmp_path / "result.meta.json"
    result = {"mode": "tender", "requirements": [], "usage": {"input_tokens": 30, "output_tokens": 7}}

    def analyze(*args):
        assert args == (b"requirements", b"profile", "tender")
        # Einschließlich einer Korrekturantwort, mit getrennten Modellpreisen.
        for model, inputs, outputs in [("claude-haiku-5-5", 10, 2), ("claude-sonnet-5-5", 10, 3), ("claude-sonnet-5-5", 10, 2)]:
            llm._capture_response_usage(model, SimpleNamespace(input_tokens=inputs, output_tokens=outputs))
        return result

    monkeypatch.setattr(cli, "analyze", analyze)
    clock = iter([100, 112.5])
    monkeypatch.setattr(cli, "perf_counter", lambda: next(clock))
    assert cli.main([str(requirements), str(profile), "--mode", "tender", "--output", str(result_path), "--metadata", str(meta_path)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(result_path.read_bytes()) == result
    meta = json.loads(meta_path.read_bytes())
    assert meta["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
    assert meta["duration_seconds"] == 12.5
    assert meta["cost_usd"] == "0.000092"
    assert meta["usage_by_model"]["claude-sonnet-5-5"]["response_count"] == 2
    assert "requirements" not in meta


@pytest.mark.parametrize("target", ["requirements.pdf", "same.json", "missing/result.json"])
def test_invalid_output_targets_rejected_before_analysis(tmp_path, monkeypatch, target):
    analyze = Mock()
    monkeypatch.setattr(cli, "analyze", analyze)
    argv = [str(tmp_path / "requirements.pdf"), str(tmp_path / "profile.pdf"), "--output", str(tmp_path / target), "--metadata", str(tmp_path / "same.json")]
    if target == "missing/result.json":
        assert cli.main(argv) == 1
    else:
        with pytest.raises(SystemExit):
            cli.main(argv)
    analyze.assert_not_called()


def test_tender_extraction_excludes_award_criteria():
    scope = llm._extraction_scope("tender")
    assert "Zuschlagskriterien" in scope
    assert "Gewichtungen" in scope
