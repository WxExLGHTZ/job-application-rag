import logging

import pytest

from copilot.ingest import (
    DocumentTooLong, PdfExtractionError, PdfWithoutText,
    build_page_chunks, extract_pages_from_pdf,
)


@pytest.mark.parametrize("texts", [[""], ["a" * 99], ["a" * 100, "", "", ""]])
def test_too_little_text(pdf_bytes, texts):
    with pytest.raises(PdfWithoutText, match="Text"):
        extract_pages_from_pdf(pdf_bytes(texts))


@pytest.mark.parametrize("texts", [["a" * 100], ["a" * 120, "", "", ""]])
def test_minimum_text_boundaries_are_inclusive(pdf_bytes, texts):
    assert len(extract_pages_from_pdf(pdf_bytes(texts))) == len(texts)


@pytest.mark.parametrize("texts", [["a" * 100] * 16, ["a" * 60_001]])
def test_document_limits(pdf_bytes, texts):
    with pytest.raises(DocumentTooLong):
        extract_pages_from_pdf(pdf_bytes(texts))


@pytest.mark.parametrize("texts", [["a" * 100] * 15, ["a" * 60_000]])
def test_document_limits_are_inclusive(pdf_bytes, texts):
    pages = extract_pages_from_pdf(pdf_bytes(texts))
    assert len(pages) == len(texts)


def test_blank_pages_do_not_renumber_chunks(pdf_bytes):
    pages = extract_pages_from_pdf(pdf_bytes(["Alpha " * 200, "", "Beta " * 200]))
    chunks = build_page_chunks(pages)
    assert {chunk["page"] for chunk in chunks} == {1, 3}
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all("Beta" not in c["text"] for c in chunks if c["page"] == 1)
    assert all("Alpha" not in c["text"] for c in chunks if c["page"] == 3)


def test_module_chunking_is_reused():
    lines = "\n".join(f"Modul Datenbanken {i} WS24/25 1,0 BE / 1 5 WP" for i in range(3))
    chunks = build_page_chunks([{"page": 4, "text": lines}])
    assert sum(c["kind"] == "module" for c in chunks) == 3
    assert all(c["page"] == 4 for c in chunks)


def test_broken_pdf_does_not_disclose_contents(caplog):
    private_text = "VERTRAULICHER_DOKUMENTINHALT"
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PdfExtractionError) as error:
            extract_pages_from_pdf(private_text.encode())
    assert private_text not in str(error.value)
    assert private_text not in caplog.text


def test_empty_document(pdf_bytes):
    with pytest.raises(PdfWithoutText):
        extract_pages_from_pdf(pdf_bytes([]))


def test_legacy_file_extraction_and_chunking_preserve_pages(pdf_bytes, monkeypatch):
    from io import BytesIO
    from pathlib import Path
    from copilot import ingest

    data = pdf_bytes(["Alpha " * 30, "", "Beta " * 30])
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [Path("synthetic.pdf")])
    monkeypatch.setattr(Path, "open", lambda *args: BytesIO(data))
    chunks = ingest.build_chunks("unused-directory")
    assert {c["page"] for c in chunks} == {1, 3}
    assert {c["filename"] for c in chunks} == {"synthetic.pdf"}
    assert "Alpha" in ingest.extract_text_from_pdf("synthetic.pdf")
