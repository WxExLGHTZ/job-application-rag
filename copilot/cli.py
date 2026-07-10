"""Command line entry point for analyzing a job ad."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from copilot.ingest import PdfExtractionError, extract_text_from_pdf
from copilot.llm import LlmError, LlmResponseError, MissingApiKeyError
from copilot.pipeline import analyze_job_ad


def main(argv: list[str] | None = None) -> int:
    """Run the job-ad analysis CLI and print formatted JSON."""
    parser = argparse.ArgumentParser(
        description="Analysiert eine Stellenanzeige gegen das lokale Profil."
    )
    parser.add_argument(
        "job_ad_path",
        help="Pfad zu einer .txt- oder .pdf-Datei mit der Stellenanzeige.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Anzahl Profil-Chunks pro Anforderung (Standard: 5).",
    )
    args = parser.parse_args(argv)

    try:
        job_ad_text = _load_job_ad(Path(args.job_ad_path))
        result = analyze_job_ad(job_ad_text, top_k=args.top_k)
    except (
        FileNotFoundError,
        ValueError,
        PdfExtractionError,
        MissingApiKeyError,
        LlmResponseError,
        LlmError,
    ) as error:
        print(f"Fehler: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _load_job_ad(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    if not path.is_file():
        raise ValueError(f"Pfad ist keine Datei: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Textdatei ist nicht als UTF-8 lesbar: {path}"
            ) from error
    elif suffix == ".pdf":
        text = extract_text_from_pdf(path)
    else:
        raise ValueError("Bitte eine .txt- oder .pdf-Datei uebergeben.")

    if not text.strip():
        raise ValueError("Die Stellenanzeige enthaelt keinen Text.")

    return text


if __name__ == "__main__":
    raise SystemExit(main())
