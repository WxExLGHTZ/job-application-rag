"""RAG-Analyse zweier PDF-Uploads, ausschließlich im Arbeitsspeicher."""

from __future__ import annotations

from typing import Any, Literal

from copilot.index import ProfileIndex, embed_texts
from copilot.ingest import build_page_chunks, extract_pages_from_pdf
from copilot.llm import TokenUsage, assess_requirements, extract_requirements


TOP_K = 5


def analyze(
    doc_requirements: bytes,
    doc_profile: bytes,
    mode: Literal["job", "tender"],
) -> dict[str, Any]:
    """Prüfe Anforderungs-PDF gegen Profil-PDF; liefere Bewertung und API-Verbrauch.

    Beide Dokumente werden vor Embeddings/API-Aufrufen geprüft. Der Profilindex
    und die Verbrauchszähler gehören nur zu diesem Aufruf. Nur das CPU-Modell
    wird prozessweit wiederverwendet; Dokumente werden nicht gespeichert.
    """
    if mode not in ("job", "tender"):
        raise ValueError("Der Modus muss 'job' oder 'tender' sein.")

    requirement_pages = extract_pages_from_pdf(doc_requirements)
    profile_pages = extract_pages_from_pdf(doc_profile)
    chunks = build_page_chunks(profile_pages)
    index = ProfileIndex(chunks, embed_texts([chunk["text"] for chunk in chunks]))
    requirement_text = "\n\n".join(page["text"] for page in requirement_pages)

    extraction_usage = TokenUsage()
    assessment_usage = TokenUsage()
    requirements = extract_requirements(requirement_text, mode, usage=extraction_usage)
    requirements_with_context = [
        {
            **requirement,
            "retrieved_chunks": index.retrieve(requirement["requirement"], top_k=TOP_K),
        }
        for requirement in requirements
    ]
    result = assess_requirements(requirements_with_context, mode, usage=assessment_usage)

    extraction = extraction_usage.to_dict()
    assessment = assessment_usage.to_dict()
    result["usage"] = {
        "extract": extraction,
        "assess": assessment,
        "total": {key: extraction[key] + assessment[key] for key in extraction},
    }
    return result
