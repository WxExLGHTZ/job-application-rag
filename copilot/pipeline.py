"""RAG-Analyse zweier PDF-Uploads, ausschließlich im Arbeitsspeicher."""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

from copilot.index import ProfileIndex, embed_texts
from copilot.ingest import build_page_chunks, extract_pages_from_pdf
from copilot.llm import TokenUsage, assess_requirements, extract_requirements
from copilot.schema import AnalysisResult, Requirement, Summary


TOP_K = 5
ProgressStage = Literal["text", "requirements", "evidence", "assessment"]


def normalize_quote(text: str) -> str:
    """Entferne Zeilen-Silbentrennung, vereinheitliche Großschreibung und Leerraum."""
    text = re.sub(r"-(?:\r\n|\n|\r)", "", text)
    return " ".join(text.casefold().split())


def verify_quotes(
    requirements: list[Requirement],
    pages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[Requirement]:
    """Prüfe jeden Beleg in einer Originalseite UND einem einzelnen Chunk."""
    normalized_pages = {page["page"]: normalize_quote(page["text"]) for page in pages}
    normalized_chunks = [
        (chunk["page"], normalize_quote(chunk["text"])) for chunk in chunks
    ]
    checked = []
    for requirement in requirements:
        data = requirement.model_dump()
        if requirement.beleg is not None:
            quote = normalize_quote(requirement.beleg.zitat)
            matching_pages = sorted({
                page for page, text in normalized_chunks
                if quote and quote in text and quote in normalized_pages.get(page, "")
            })
            if not matching_pages:
                data.update(
                    status="nicht_belegt", beleg=None,
                    begruendung="Beleg im Dokument nicht gefunden",
                )
            elif requirement.beleg.seite not in matching_pages:
                data["beleg"]["seite"] = matching_pages[0]
        checked.append(Requirement.model_validate(data))
    return checked


def calculate_summary(requirements: list[Requirement], fazit: str) -> Summary:
    """Berechne gewichtete Prozentpunkte, kaufmännisch auf ganze Zahlen gerundet."""
    values = {"erfuellt": 2, "teilweise": 1, "nicht_belegt": 0}
    total_weight = sum(2 if item.typ == "muss" else 1 for item in requirements)
    earned = sum(
        (2 if item.typ == "muss" else 1) * values[item.status] for item in requirements
    )
    score = (100 * earned + total_weight) // (2 * total_weight) if total_weight else 0
    mandatory = [item for item in requirements if item.typ == "muss"]
    fulfilled = sum(item.status == "erfuellt" for item in mandatory)
    return Summary(score=score, muss_erfuellt=f"{fulfilled}/{len(mandatory)}", fazit=fazit)


def analyze(
    doc_requirements: bytes,
    doc_profile: bytes,
    mode: Literal["job", "tender"],
    *,
    on_progress: Callable[[ProgressStage], None] | None = None,
) -> dict[str, Any]:
    """Prüfe Anforderungs-PDF gegen Profil-PDF; liefere Bewertung und API-Verbrauch.

    Beide Dokumente werden vor Embeddings/API-Aufrufen geprüft. Der Profilindex
    und die Verbrauchszähler gehören nur zu diesem Aufruf. Nur das CPU-Modell
    wird prozessweit wiederverwendet; Dokumente werden nicht gespeichert.
    """
    if mode not in ("job", "tender"):
        raise ValueError("Der Modus muss 'job' oder 'tender' sein.")

    if on_progress is not None:
        on_progress("text")
    requirement_pages = extract_pages_from_pdf(doc_requirements)
    profile_pages = extract_pages_from_pdf(doc_profile)
    chunks = build_page_chunks(profile_pages)
    index = ProfileIndex(chunks, embed_texts([chunk["text"] for chunk in chunks]))
    requirement_text = "\n\n".join(page["text"] for page in requirement_pages)

    extraction_usage = TokenUsage()
    assessment_usage = TokenUsage()
    if on_progress is not None:
        on_progress("requirements")
    requirements = extract_requirements(requirement_text, mode, usage=extraction_usage)
    if on_progress is not None:
        on_progress("evidence")
    requirements_with_context = [
        {
            **requirement,
            "retrieved_chunks": index.retrieve(requirement["text"], top_k=TOP_K),
        }
        for requirement in requirements
    ]
    if on_progress is not None:
        on_progress("assessment")
    assessment = assess_requirements(requirements_with_context, mode, usage=assessment_usage)
    canonical = {item["id"]: item for item in requirements}
    evaluated = [
        Requirement.model_validate({**item, **canonical[item["id"]]})
        for item in assessment["requirements"]
    ]
    checked = verify_quotes(evaluated, profile_pages, chunks)
    usage = (extraction_usage, assessment_usage)
    result = AnalysisResult(
        mode=mode,
        summary=calculate_summary(checked, assessment["fazit"]),
        requirements=checked,
        usage={
            # Anthropic meldet gecachte Eingabetokens separat; hier alle Eingaben summieren.
            "input_tokens": sum(
                part.input_tokens + part.cache_creation_input_tokens + part.cache_read_input_tokens
                for part in usage
            ),
            "output_tokens": sum(part.output_tokens for part in usage),
        },
    )
    return result.model_dump()
