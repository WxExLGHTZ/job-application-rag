"""USD-Listenpreise der Claude API; keine Accountdaten oder Dokumentinhalte.

Haiku-Tarif für einzelne Anfragen bis 100.000 Eingabetokens. Die Pipeline bleibt
einschließlich Antwortreserve darunter. Summierte Wiederholungen dürfen mehr
Tokens enthalten: entscheidend für den Tarif ist jede einzelne Anfrage.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
import re


PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICING_CHECKED_ON = "2026-10-22"
TOKENS_PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class TokenPrices:
    """USD pro Million Tokens, Standard-API ohne regionale Zuschläge."""

    input: Decimal
    output: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal


MODEL_PRICES = {
    "claude-haiku-5-5": TokenPrices(
        Decimal("0.10"), Decimal("0.50"), Decimal("0.125"), Decimal("0.20"), Decimal("0.01"),
    ),
    "claude-sonnet-5-5": TokenPrices(
        Decimal("2.00"), Decimal("10.00"), Decimal("2.50"), Decimal("4.00"), Decimal("0.10"),
    ),
}


def calculate_cost_usd(model: str, usage: Mapping[str, int]) -> Decimal:
    """Berechne ungerundete Kosten aus separat gemeldeten Tokenkategorien.

`input_tokens` enthält die normalen Eingaben ohne Cache-Reads/-Writes, wie in
Anthropics API-Antwort. Deshalb TokenUsage.to_dict() je Stufe verwenden, nicht
die zusammengefassten Eingabetokens des öffentlichen Analyse-Ergebnisses.
Cache-Writes ohne explizite 1h-Aufteilung werden zum Standard-5m-Tarif gerechnet.
Die Kosten enthalten alle in usage bereits summierten Antwortkorrekturen.
"""
    prices = next((prices for name, prices in MODEL_PRICES.items()
                   if re.fullmatch(re.escape(name) + r"(?:-\d{8})?", model)), None)
    if prices is None:
        raise ValueError("Für dieses Modell ist kein Preis hinterlegt.")
    fields = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
              "cache_creation_1h_input_tokens", "cache_read_input_tokens")
    counts = {field: usage.get(field, 0) for field in fields}
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("Tokenzahlen müssen nichtnegative ganze Zahlen sein.")
    write_1h = counts["cache_creation_1h_input_tokens"]
    write_5m = counts["cache_creation_input_tokens"] - write_1h
    if write_5m < 0:
        raise ValueError("1h-Cache-Tokens überschreiten die gesamten Cache-Schreibzugriffe.")
    return (
        counts["input_tokens"] * prices.input
        + counts["output_tokens"] * prices.output
        + counts["cache_read_input_tokens"] * prices.cache_read
        + write_5m * prices.cache_write_5m
        + write_1h * prices.cache_write_1h
    ) / TOKENS_PER_MILLION
