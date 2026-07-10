"""RAG pipeline that connects requirement extraction, retrieval, and assessment."""

from __future__ import annotations

from typing import Any

from copilot.index import retrieve
from copilot.llm import assess_requirements, extract_requirements


def analyze_job_ad(job_ad_text: str, top_k: int = 5) -> dict[str, Any]:
    """Analyze one job ad against the profile knowledge base.

    Args:
        job_ad_text: Full text of the job ad.
        top_k: Number of profile chunks to retrieve per requirement.

    Returns:
        The final Claude assessment as a JSON-compatible dictionary.
    """
    if not job_ad_text.strip():
        raise ValueError("Die Stellenanzeige ist leer.")
    if top_k <= 0:
        raise ValueError("top_k muss groesser als 0 sein.")

    requirements = extract_requirements(job_ad_text)
    requirements_with_context: list[dict[str, Any]] = []

    for requirement in requirements:
        requirement_text = str(requirement["requirement"]).strip()
        matching_chunks = retrieve(requirement_text, top_k=top_k)
        requirements_with_context.append(
            {
                **requirement,
                "retrieved_chunks": matching_chunks,
            }
        )

    return assess_requirements(requirements_with_context)
