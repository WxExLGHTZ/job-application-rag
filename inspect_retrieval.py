"""Inspect semantic retrieval results for fixed German test questions."""

import sys

from copilot.index import build_index


TOP_K = 5
PREVIEW_CHARS = 200
TEST_QUESTIONS = [
    "Wer hat FastAPI-Erfahrung?",
    "Welche Datenbank-Kenntnisse?",
    "Erfahrung mit Computer Vision und 3D?",
    "Welche Abschlussnote im Master?",
]


def preview_text(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    """Shorten chunk text for terminal diagnostics."""
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[:max_chars] + "..."


def configure_output_encoding() -> None:
    """Use UTF-8 output so PDF text previews print on Windows terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    """Build the profile index and print retrieval diagnostics."""
    configure_output_encoding()
    index = build_index()
    print(f"Gesamtzahl Chunks: {len(index.chunks)}")

    if not index.chunks:
        print("Keine Chunks gefunden. Liegen lesbare PDFs in profile/?")
        return

    for question in TEST_QUESTIONS:
        print("=" * 80)
        print(f"Frage: {question}")
        print("-" * 80)

        results = index.retrieve(question, top_k=TOP_K)
        for rank, result in enumerate(results, start=1):
            print(f"Treffer {rank} | Score: {result['score']:.4f} | Datei: {result['filename']}")
            print(preview_text(result["text"]))
            print()


if __name__ == "__main__":
    main()
