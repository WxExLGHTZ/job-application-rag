"""Kommandozeilen-Einstieg für die Analyse zweier PDF-Dateien."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

from copilot.ingest import PdfExtractionError
from copilot.llm import LlmError, capture_token_usage
from copilot.pipeline import analyze
from copilot.pricing import calculate_cost_usd, PRICING_CHECKED_ON, PRICING_SOURCE


def main(argv: list[str] | None = None) -> int:
    """Lese zwei PDFs, analysiere sie und schreibe JSON bzw. einen sicheren Fehler."""
    parser = argparse.ArgumentParser(description="Prüft Anforderungen gegen ein Profil.")
    parser.add_argument("requirements_pdf", type=Path, help="PDF mit Stellenanzeige oder Ausschreibung.")
    parser.add_argument("profile_pdf", type=Path, help="PDF mit Lebenslauf oder Firmenprofil.")
    parser.add_argument("--mode", choices=("job", "tender"), default="job")
    parser.add_argument("--output", type=Path, help="Ergebnis zusätzlich als JSON speichern.")
    parser.add_argument("--metadata", type=Path, help="Messwerte zum gespeicherten Ergebnis ablegen.")
    args = parser.parse_args(argv)

    try:
        if args.metadata and not args.output:
            parser.error("--metadata erfordert --output.")
        destinations = [path.resolve() for path in (args.output, args.metadata) if path]
        inputs = {args.requirements_pdf.resolve(), args.profile_pdf.resolve()}
        if len(set(destinations)) != len(destinations) or inputs.intersection(destinations):
            parser.error("Ausgabedateien müssen voneinander und von den PDFs verschieden sein.")
        if any(not path.parent.is_dir() or path.is_dir() for path in destinations):
            raise OSError("Ungültiges Ausgabeziel.")
        documents = []
        for path in (args.requirements_pdf, args.profile_pdf):
            if path.suffix.lower() != ".pdf":
                raise PdfExtractionError("Bitte zwei PDF-Dateien übergeben.")
            documents.append(path.read_bytes())
        started = perf_counter()
        with capture_token_usage() as capture:
            result = analyze(documents[0], documents[1], args.mode)
        duration = perf_counter() - started
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.metadata:
            if not capture.complete or not capture.by_model:
                raise ValueError("Keine vollständigen Verbrauchsdaten vorhanden.")
            by_model = {model: usage.to_dict() for model, usage in capture.by_model.items()}
            cost = sum((calculate_cost_usd(model, usage) for model, usage in by_model.items()), Decimal(0))
            metadata = {
                "mode": args.mode,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "result_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                "duration_seconds": duration,
                "duration_scope": "analyze inklusive PDF-Lesen und gegebenenfalls Modellinitialisierung",
                "cost_usd": str(cost),
                "usage_by_model": by_model,
                "pricing_checked_on": PRICING_CHECKED_ON,
                "pricing_source": PRICING_SOURCE,
            }
        if args.output:
            args.output.write_bytes(serialized.encode("utf-8"))
        if args.metadata:
            args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"Fehler: Datei konnte nicht gelesen oder geschrieben werden ({type(error).__name__}).", file=sys.stderr)
        return 1
    except (PdfExtractionError, LlmError) as error:
        print(f"Fehler: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Fehler: Analyse fehlgeschlagen ({type(error).__name__}).", file=sys.stderr)
        return 1

    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
