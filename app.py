"""Streamlit UI for the Bewerbungs-Copilot.

Annahme: Die Analysefunktion in copilot.pipeline ist die einzige Quelle fuer
Analyse-, Retrieval- und LLM-Logik. Diese App rendert nur Eingabe, Fehler und
das Ergebnis-Dict.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from copilot.llm import LlmError, LlmResponseError, MissingApiKeyError
from copilot.pipeline import analyze_job_ad


STATUS_MARKERS = {
    "erfuellt": ("🟢", "erfüllt"),
    "teilweise": ("🟡", "teilweise"),
    "luecke": ("🔴", "Lücke"),
}


def main() -> None:
    """Render the Streamlit app and call the existing pipeline on button click."""
    st.set_page_config(
        page_title="Bewerbungs-Copilot",
        page_icon="📄",
        layout="wide",
    )

    st.title("Bewerbungs-Copilot")
    st.write(
        "Füge den Text einer Stellenanzeige ein. Beim Klick auf "
        "„Analysieren“ wird die Anzeige gegen das lokale Profil geprüft."
    )

    job_ad_text = st.text_area(
        "Stellenanzeige",
        height=320,
        placeholder="Text der Stellenanzeige hier einfügen ...",
    )

    if st.button("Analysieren", type="primary"):
        if not job_ad_text.strip():
            st.info("Bitte füge zuerst den Text einer Stellenanzeige ein.")
            return

        try:
            with st.spinner("Analyse läuft ..."):
                result = analyze_job_ad(job_ad_text)
        except (MissingApiKeyError, LlmResponseError, LlmError, ValueError) as error:
            st.error(f"Analyse fehlgeschlagen: {error}")
            return
        except Exception as error:  # Schutznetz, damit die UI nicht abstürzt.
            st.error(
                "Analyse fehlgeschlagen: Es ist ein unerwarteter Fehler "
                f"aufgetreten ({type(error).__name__}: {error})."
            )
            return

        render_result(result)


def render_result(result: dict[str, Any]) -> None:
    """Render all expected result sections from the pipeline output."""
    st.success("Analyse abgeschlossen.")

    render_requirement_matrix(result.get("anforderungs_matrix"))
    render_framing_suggestions(result.get("framing_vorschlaege"))
    render_highlighted_projects(result.get("hervorzuhebende_projekte"))
    render_interview_questions(result.get("wahrscheinliche_interviewfragen"))

    with st.expander("Rohes JSON anzeigen"):
        st.json(result)


def render_requirement_matrix(matrix: Any) -> None:
    """Render the requirement matrix with traffic-light status markers."""
    st.subheader("Anforderungs-Matrix")
    rows = as_list(matrix)
    if not rows:
        st.info("Keine Anforderungen in der Analyse enthalten.")
        return

    for item in rows:
        if not isinstance(item, dict):
            st.write(item)
            continue

        requirement = text_or_fallback(item.get("requirement"), "Ohne Anforderungstext")
        status_key = str(item.get("status", "")).strip().lower()
        marker, label = STATUS_MARKERS.get(status_key, ("⚪", status_key or "unbekannt"))
        reason = text_or_fallback(item.get("begruendung"), "Keine Begründung angegeben.")
        evidence = format_evidence(item.get("beleg"))

        status_col, content_col = st.columns([1, 5], vertical_alignment="top")
        status_col.markdown(f"**{marker} {label}**")
        content_col.markdown(f"**{requirement}**")
        content_col.write(reason)
        content_col.caption(evidence["summary"])

        if evidence["text"]:
            with content_col.expander("Belegtext anzeigen"):
                st.write(evidence["text"])


def render_framing_suggestions(items: Any) -> None:
    """Render honest framing suggestions for gaps."""
    st.subheader("Framing-Vorschläge für Lücken")
    suggestions = as_list(items)
    if not suggestions:
        st.info("Keine Framing-Vorschläge enthalten.")
        return

    for suggestion in suggestions:
        if isinstance(suggestion, dict):
            gap = text_or_fallback(suggestion.get("luecke"), "Nicht benannte Lücke")
            framing = text_or_fallback(
                suggestion.get("ehrliches_framing"),
                "Keine Formulierung angegeben.",
            )
            st.markdown(f"**{gap}**")
            st.write(framing)
        else:
            st.write(suggestion)


def render_highlighted_projects(items: Any) -> None:
    """Render projects or experiences that should be highlighted."""
    st.subheader("Hervorzuhebende Projekte")
    projects = as_list(items)
    if not projects:
        st.info("Keine hervorzuhebenden Projekte enthalten.")
        return

    for project in projects:
        if isinstance(project, dict):
            title = text_or_fallback(project.get("projekt"), "Unbenanntes Projekt")
            reason = text_or_fallback(
                project.get("warum_relevant"),
                "Keine Begründung angegeben.",
            )
            filename = project.get("beleg_filename")
            source = (
                f"Beleg: {filename}"
                if isinstance(filename, str) and filename.strip()
                else "Kein Beleg im Profil gefunden."
            )
            st.markdown(f"**{title}**")
            st.write(reason)
            st.caption(source)
        else:
            st.write(project)


def render_interview_questions(items: Any) -> None:
    """Render likely interview questions."""
    st.subheader("Wahrscheinliche Interviewfragen")
    questions = as_list(items)
    if not questions:
        st.info("Keine Interviewfragen enthalten.")
        return

    for question in questions:
        if isinstance(question, dict):
            text = text_or_fallback(question.get("frage"), "Ohne Fragetext")
            reason = text_or_fallback(question.get("grund"), "Keine Begründung angegeben.")
            st.markdown(f"**{text}**")
            st.write(reason)
        else:
            st.write(question)


def format_evidence(beleg: Any) -> dict[str, str | None]:
    """Return readable evidence text while tolerating missing fields."""
    if not isinstance(beleg, dict):
        return {
            "summary": "Kein Beleg im Profil gefunden.",
            "text": None,
        }

    filename = beleg.get("filename")
    score = beleg.get("score")
    evidence_text = beleg.get("text")

    if isinstance(filename, str) and filename.strip():
        summary = f"Beleg: {filename}"
        if isinstance(score, int | float):
            summary += f" (Score: {float(score):.3f})"
    else:
        summary = "Kein Beleg im Profil gefunden."

    return {
        "summary": summary,
        "text": evidence_text if isinstance(evidence_text, str) and evidence_text.strip() else None,
    }


def as_list(value: Any) -> list[Any]:
    """Treat missing values as empty lists and single values as one-item lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text_or_fallback(value: Any, fallback: str) -> str:
    """Return stripped text for string-like values or a readable fallback."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


if __name__ == "__main__":
    main()
