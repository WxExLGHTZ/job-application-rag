"""Show a quick preview of text extracted from profile PDFs."""

from copilot.ingest import load_profile


PREVIEW_CHARS = 800


def main() -> None:
    """Load profile PDFs and print extraction diagnostics."""
    documents = load_profile()

    if not documents:
        print("Keine lesbaren PDF-Dateien in profile/ gefunden.")
        return

    for index, document in enumerate(documents, start=1):
        print("=" * 80)
        print(f"PDF {index}: {document['filename']}")
        print(f"Seiten: {document['num_pages']}")
        print(f"Zeichen: {document['num_chars']}")
        print("-" * 80)
        print(document["text"][:PREVIEW_CHARS])
        print()


if __name__ == "__main__":
    main()
