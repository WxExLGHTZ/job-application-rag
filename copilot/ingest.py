"""PDF loading and text extraction for the profile knowledge base."""

import re
import warnings
from pathlib import Path
from typing import Any

from pypdf import PdfReader


SEMESTER_RE = re.compile(r"\b(?:WS|SS)\d{2}(?:/\d{2})?\b")
GRADE_RE = re.compile(r"\b(?:[1-4],[037]|mit Erfolg|mE)\b", re.IGNORECASE)
MODULE_STATUS_RE = re.compile(r"\bBE\b.*\b(?:PF|WP)\b|\b(?:PF|WP)\b")
CREDITS_RE = re.compile(r"/\s*\d+\s+(?P<credits>\d+)\s+(?P<kind>PF|WP)\b")
MIN_MODULE_LINES_FOR_TABLE = 3


class PdfExtractionError(ValueError):
    """Raised when a PDF cannot be read or does not contain extractable text."""


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract text from one PDF file.

    Args:
        path: Path to a single PDF file.

    Returns:
        The extracted text as one string. Pages without text are skipped.

    Raises:
        PdfExtractionError: If the PDF is missing, unreadable, broken, empty, or
            contains no extractable text.
    """
    text, _num_pages = _extract_pdf_text_and_page_count(path)
    return text


def load_profile(profile_dir: str | Path = "profile") -> list[dict[str, Any]]:
    """Load all PDF files from a profile directory.

    Args:
        profile_dir: Directory that contains profile PDFs.

    Returns:
        A list of dictionaries, one per successfully read PDF, with filename,
        text, num_pages, and num_chars. Broken or unreadable PDFs are skipped
        with a warning.
    """
    directory = Path(profile_dir)
    if not directory.exists():
        warnings.warn(f"Profilordner nicht gefunden: {directory}", stacklevel=2)
        return []
    if not directory.is_dir():
        warnings.warn(f"Profilpfad ist kein Ordner: {directory}", stacklevel=2)
        return []

    documents: list[dict[str, Any]] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        try:
            text, num_pages = _extract_pdf_text_and_page_count(pdf_path)
        except PdfExtractionError as error:
            warnings.warn(f"Ueberspringe {pdf_path.name}: {error}", stacklevel=2)
            continue

        documents.append(
            {
                "filename": pdf_path.name,
                "text": text,
                "num_pages": num_pages,
                "num_chars": len(text),
            }
        )

    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split flowing text into overlapping chunks at natural boundaries.

    Args:
        text: The full text of one document.
        chunk_size: Target number of characters per chunk.
        overlap: Number of characters repeated from the previous chunk.

    Returns:
        A list of non-empty text chunks.

    Raises:
        ValueError: If chunk_size or overlap would make chunking impossible.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size muss groesser als 0 sein.")
    if overlap < 0:
        raise ValueError("overlap darf nicht negativ sein.")
    if overlap >= chunk_size:
        raise ValueError("overlap muss kleiner als chunk_size sein.")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        if len(cleaned_text) - start <= chunk_size:
            chunk = cleaned_text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        end = _find_chunk_end(cleaned_text, start, chunk_size)
        if end <= start:
            end = min(start + chunk_size, len(cleaned_text))

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Schnitt-Strategie: erst Absatzgrenzen, dann Zeilenumbrueche, dann
        # erst als Notloesung ein harter Schnitt. So bleiben Sinnabschnitte
        # eher zusammen; die Ueberlappung schuetzt Informationen an Kanten.
        start = _find_overlap_start(cleaned_text, start, end, overlap)

    return chunks


def build_chunks(profile_dir: str | Path = "profile") -> list[dict[str, Any]]:
    """Build retrievable chunks for all profile PDFs.

    Args:
        profile_dir: Directory that contains profile PDFs.

    Returns:
        A list of dictionaries with chunk_id, filename, text, and kind.
    """
    documents = load_profile(profile_dir)
    chunks: list[dict[str, Any]] = []

    for document in documents:
        for chunk, kind in _chunk_document(document["text"]):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "filename": document["filename"],
                    "text": chunk,
                    "kind": kind,
                }
            )

    return chunks


def _chunk_document(text: str) -> list[tuple[str, str]]:
    module_chunks, remaining_text = _extract_module_chunks(text)
    text_chunks = [(chunk, "text") for chunk in chunk_text(remaining_text)]
    return text_chunks + [(chunk, "module") for chunk in module_chunks]


def _extract_module_chunks(text: str) -> tuple[list[str], str]:
    lines = text.splitlines()
    module_chunks: list[str] = []
    removed_line_indices: set[int] = set()

    for index, line in enumerate(lines):
        if not _looks_like_module_grade_line(line):
            continue

        module_line = _normalize_space(line)
        previous_index = _previous_non_empty_line_index(lines, index)

        if previous_index is not None and _should_join_previous_module_line(
            lines[previous_index], module_line
        ):
            module_line = _normalize_space(f"{lines[previous_index]} {module_line}")
            removed_line_indices.add(previous_index)

        module_chunks.append(_format_module_chunk(module_line))
        removed_line_indices.add(index)

    # Die Sonderbehandlung greift nur, wenn mehrere Zeilen gleichzeitig nach
    # Notentabelle aussehen. Einzelne Semester-/Noten-Erwaehnungen in CV-Prosa
    # oder Paper-Text laufen weiter durch das normale grenzbasierte Chunking.
    if len(module_chunks) < MIN_MODULE_LINES_FOR_TABLE:
        return [], text

    remaining_text = "\n".join(
        line for index, line in enumerate(lines) if index not in removed_line_indices
    )
    return module_chunks, remaining_text


def _looks_like_module_grade_line(line: str) -> bool:
    normalized = _normalize_space(line)
    if len(normalized) < 20:
        return False

    semester_match = SEMESTER_RE.search(normalized)
    grade_match = GRADE_RE.search(normalized)
    if not semester_match or not grade_match:
        return False

    module_name = normalized[: semester_match.start()].strip(" -:\u00a0")
    if len(module_name) < 2:
        return False

    # Heuristik fuer echte Notentabellen: Semester + Bewertung reichen allein
    # nicht; zusaetzlich erwarten wir typische Tabellenkuerzel wie BE/PF/WP
    # oder die kompakte LP-Spalte aus dem PDF (z. B. "/ 1 5 WP").
    return bool(MODULE_STATUS_RE.search(normalized) or CREDITS_RE.search(normalized))


def _format_module_chunk(line: str) -> str:
    semester_match = SEMESTER_RE.search(line)
    grade_match = GRADE_RE.search(line)
    credits_match = CREDITS_RE.search(line)

    semester = semester_match.group(0) if semester_match else "unbekanntes Semester"
    module_name = (
        line[: semester_match.start()].strip(" -:\u00a0") if semester_match else line
    )
    grade = grade_match.group(0) if grade_match else "unbekannte Bewertung"

    parts = [f"Studienmodul: {module_name}", f"Semester {semester}"]
    if grade.lower() in {"mit erfolg", "me"}:
        parts.append(f"Bewertung {grade}")
    else:
        parts.append(f"Note {grade}")

    if credits_match:
        parts.append(f"{credits_match.group('credits')} Leistungspunkte")
        parts.append(credits_match.group("kind"))

    return f"{'; '.join(parts)}. Originalzeile: {line}"


def _find_chunk_end(text: str, start: int, chunk_size: int) -> int:
    target_end = min(start + chunk_size, len(text))
    min_clean_end = start + max(1, chunk_size // 2)

    paragraph_end = _last_boundary(text, start, target_end, min_clean_end, r"\n\s*\n")
    if paragraph_end is not None:
        return paragraph_end

    line_end = _last_boundary(text, start, target_end, min_clean_end, r"\n")
    if line_end is not None:
        return line_end

    whitespace_end = _last_whitespace(text, start, target_end, min_clean_end)
    if whitespace_end is not None:
        return whitespace_end

    return target_end


def _last_boundary(
    text: str, start: int, target_end: int, min_clean_end: int, pattern: str
) -> int | None:
    segment = text[start:target_end]
    matches = [
        start + match.end()
        for match in re.finditer(pattern, segment)
        if start + match.end() >= min_clean_end
    ]
    return matches[-1] if matches else None


def _last_whitespace(
    text: str, start: int, target_end: int, min_clean_end: int
) -> int | None:
    for index in range(target_end - 1, min_clean_end - 1, -1):
        if text[index].isspace():
            return index + 1
    return None


def _find_overlap_start(text: str, chunk_start: int, chunk_end: int, overlap: int) -> int:
    if chunk_end >= len(text):
        return len(text)

    next_start = max(chunk_start + 1, chunk_end - overlap)
    while next_start > chunk_start + 1 and not text[next_start - 1].isspace():
        next_start -= 1
    while next_start < len(text) and text[next_start].isspace():
        next_start += 1
    return next_start


def _normalize_space(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def _previous_non_empty_line_index(lines: list[str], index: int) -> int | None:
    for previous_index in range(index - 1, -1, -1):
        if lines[previous_index].strip():
            return previous_index
    return None


def _should_join_previous_module_line(previous_line: str, current_line: str) -> bool:
    previous = _normalize_space(previous_line)
    if not previous or SEMESTER_RE.search(previous):
        return False

    semester_match = SEMESTER_RE.search(current_line)
    current_module_name = (
        current_line[: semester_match.start()].strip(" -:\u00a0")
        if semester_match
        else current_line
    )

    return (
        current_line.startswith("(")
        or len(current_module_name) <= 16
        or previous.endswith(":")
        or previous.endswith(",")
    )


def _extract_pdf_text_and_page_count(path: str | Path) -> tuple[str, int]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise PdfExtractionError(f"Datei nicht gefunden: {pdf_path}")
    if not pdf_path.is_file():
        raise PdfExtractionError(f"Pfad ist keine Datei: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as error:
        raise PdfExtractionError(
            f"PDF konnte nicht gelesen werden ({type(error).__name__}: {error})"
        ) from error

    num_pages = len(reader.pages)
    if num_pages == 0:
        raise PdfExtractionError("PDF enthaelt keine Seiten.")

    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
        except Exception as error:
            warnings.warn(
                f"Seite {page_number} in {pdf_path.name} konnte nicht gelesen "
                f"werden und wird uebersprungen ({type(error).__name__}: {error}).",
                stacklevel=2,
            )
            continue

        if page_text:
            page_texts.append(page_text.strip())

    text = "\n\n".join(page_text for page_text in page_texts if page_text)
    if not text:
        raise PdfExtractionError("PDF enthaelt keinen extrahierbaren Text.")

    return text, num_pages
