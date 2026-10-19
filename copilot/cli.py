"""Kommandozeilen-Einstieg für die Analyse zweier PDF-Dateien."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from copilot.ingest import PdfExtractionError
from copilot.llm import LlmError
from copilot.pipeline import analyze


def main(argv: list[str] | None = None) -> int:
    """Lese zwei PDFs, analysiere sie und schreibe JSON bzw. einen sicheren Fehler."""
    parser = argparse.ArgumentParser(description="Prüft Anforderungen gegen ein Profil.")
    parser.add_argument("requirements_pdf", type=Path, help="PDF mit Stellenanzeige oder Ausschreibung.")
    parser.add_argument("profile_pdf", type=Path, help="PDF mit Lebenslauf oder Firmenprofil.")
    parser.add_argument("--mode", choices=("job", "tender"), default="job")
    args = parser.parse_args(argv)

    try:
        documents = []
        for path in (args.requirements_pdf, args.profile_pdf):
            if path.suffix.lower() != ".pdf":
                raise PdfExtractionError("Bitte zwei PDF-Dateien übergeben.")
            documents.append(path.read_bytes())
        result = analyze(documents[0], documents[1], args.mode)
    except OSError as error:
        print(f"Fehler: PDF-Datei konnte nicht gelesen werden ({type(error).__name__}).", file=sys.stderr)
        return 1
    except (PdfExtractionError, LlmError) as error:
        print(f"Fehler: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Fehler: Analyse fehlgeschlagen ({type(error).__name__}).", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
