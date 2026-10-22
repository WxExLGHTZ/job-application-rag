from decimal import Decimal
from types import SimpleNamespace

import pytest

from copilot.llm import TokenUsage
from copilot.pricing import calculate_cost_usd


@pytest.mark.parametrize("model,expected", [
    ("claude-haiku-5-5", "0.00166"),
    ("claude-sonnet-5-5", "0.03315"),
    ("claude-sonnet-5-5-20261001", "0.03315"),
])
def test_cost_with_separate_input_output_and_cache_categories(model, expected):
    usage = TokenUsage()
    usage.add_response(SimpleNamespace(
        input_tokens=10_000, output_tokens=1_000,
        cache_creation_input_tokens=1_000, cache_read_input_tokens=500,
        cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=600,
                                       ephemeral_1h_input_tokens=400),
    ))
    assert usage.to_dict()["total_tokens"] == 12_500
    assert calculate_cost_usd(model, usage.to_dict()) == Decimal(expected)


def test_cache_creation_without_breakdown_uses_standard_5m_rate():
    assert calculate_cost_usd("claude-haiku-5-5", {
        "cache_creation_input_tokens": 1_000,
    }) == Decimal("0.000125")


def test_response_corrections_are_included_and_cache_totals_are_not_doubled():
    first = SimpleNamespace(input_tokens=10, output_tokens=5)
    usage = TokenUsage()
    usage.add_response(first)
    usage.add_response(first)
    assert calculate_cost_usd("claude-haiku-5-5", usage.to_dict()) == Decimal("0.000007")


def test_previous_mixed_analysis_is_priced_by_stage_not_by_total_tokens():
    extraction = calculate_cost_usd("claude-haiku-5-5", {
        "input_tokens": 2614, "output_tokens": 741,
    })
    assessment = calculate_cost_usd("claude-sonnet-5-5", {
        "input_tokens": 6358, "output_tokens": 1692,
    })
    assert extraction == Decimal("0.0006319")
    assert assessment == Decimal("0.029636")
    assert extraction + assessment == Decimal("0.0302679")


def test_unknown_model_has_no_silent_price_fallback():
    with pytest.raises(ValueError) as error:
        calculate_cost_usd("PRIVATE_MODEL", {"input_tokens": 1})
    assert "PRIVATE_MODEL" not in str(error.value)


@pytest.mark.parametrize("count", [-1, 1.5, "10", True, None])
def test_invalid_token_counts_are_rejected(count):
    with pytest.raises(ValueError):
        calculate_cost_usd("claude-haiku-5-5", {"input_tokens": count})


def test_inconsistent_cache_breakdown_is_rejected():
    with pytest.raises(ValueError):
        calculate_cost_usd("claude-sonnet-5-5", {
            "cache_creation_input_tokens": 2, "cache_creation_1h_input_tokens": 3,
        })


def test_empty_usage_costs_zero():
    assert calculate_cost_usd("claude-haiku-5-5", {}) == Decimal(0)
