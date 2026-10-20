"""Streamlit-Oberfläche für die Analyse zweier PDFs im Arbeitsspeicher."""

from __future__ import annotations

import html
from pathlib import Path
import re
from typing import Any

import streamlit as st

from copilot import pipeline
from copilot.index import get_embedding_model
from copilot.ingest import DocumentTooLong, PdfExtractionError, PdfWithoutText
from copilot.llm import (
    LlmApiError, LlmError, LlmResponseError, MissingApiKeyError,
    OutputTokenLimit, RequestTooLarge,
)


TITLE = "Anforderungs-Check"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
MODES = {"job": "Stellenanzeige und Lebenslauf", "tender": "Ausschreibung und Firmenprofil"}
UPLOAD_LABELS = {"job": ("Stellenanzeige", "Lebenslauf"), "tender": ("Ausschreibung", "Firmenprofil")}
STAGES = {
    "text": "Text lesen", "requirements": "Anforderungen finden",
    "evidence": "Belege suchen", "assessment": "Bewerten",
}
STATUS_LABELS = {"erfuellt": "erfüllt", "teilweise": "teilweise", "nicht_belegt": "nicht belegt"}

RESULT_CSS = """
<style>
.ac-result { color: inherit; line-height: 1.65; overflow-wrap: anywhere; }
.ac-result * { box-sizing: border-box; }
.ac-result .ac-summary, .ac-result .ac-card {
    border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
    border-radius: 16px; padding: 1.35rem; margin-bottom: 1rem;
    background: color-mix(in srgb, currentColor 3%, transparent);
}
.ac-result .ac-summary { display: flex; gap: 1.4rem; align-items: center; flex-wrap: wrap; }
.ac-result .ac-overview { flex: 1; min-width: 12rem; }
.ac-result .ac-score { border-radius: 14px; padding: .65rem 1rem; text-align: center; }
.ac-result .ac-score strong { display: block; font-size: 3rem; line-height: 1.15; }
.ac-result .ac-score span { font-size: .85rem; }
.ac-result .ac-must { font-size: 1.15rem; font-weight: 700; margin: 0 0 .4rem; }
.ac-result .ac-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .8rem; }
.ac-result .ac-badge { border-radius: 999px; padding: .15rem .7rem; font-size: .85rem; font-weight: 650; }
.ac-result .ac-green { background: #dcfce7; color: #166534; }
.ac-result .ac-yellow { background: #fef3c7; color: #854d0e; }
.ac-result .ac-red { background: #fee2e2; color: #991b1b; }
.ac-result .ac-gray { background: #e5e7eb; color: #374151; }
.ac-result .ac-type { border: 1px solid currentColor; }
.ac-result h3 { color: inherit; font-size: 1.12rem; line-height: 1.5; margin: 0 0 .8rem; padding: 0; }
.ac-result p { margin: .5rem 0 0; white-space: pre-wrap; }
.ac-result blockquote {
    color: inherit; opacity: 1; margin: 1rem 0; padding: .1rem 0 .1rem 1rem;
    border-left: 3px solid color-mix(in srgb, currentColor 40%, transparent);
    white-space: pre-wrap;
}
.ac-result cite { display: block; font-style: normal; font-size: .85rem; margin-top: .35rem; }
@media (max-width: 480px) {
    .ac-result .ac-summary, .ac-result .ac-card { padding: 1rem; }
    .ac-result .ac-summary { gap: 1rem; }
    .ac-result .ac-score strong { font-size: 2.5rem; }
}
</style>
"""


@st.cache_resource(show_spinner="Embedding-Modell wird geladen …")
def load_model():
    """Nur das CPU-Modell wird sitzungsübergreifend wiederverwendet."""
    return get_embedding_model()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def score_color(score: int) -> str:
    return "green" if score >= 75 else "yellow" if score >= 50 else "red"


def status_color(requirement: dict[str, Any]) -> str:
    if requirement["status"] == "erfuellt":
        return "green"
    if requirement["status"] == "teilweise":
        return "yellow"
    return "red" if requirement["typ"] == "muss" else "gray"


def render_result_html(result: dict[str, Any]) -> str:
    """CSS-Klassen kommen nur aus festen Zuordnungen, sämtliche Texte sind escaped."""
    summary = result["summary"]
    parts = [
        RESULT_CSS,
        '<section class="ac-result" aria-label="Analyseergebnis">',
        '<div class="ac-summary">',
        f'<div class="ac-score ac-{score_color(summary["score"])}">'
        f'<strong>{_escape(summary["score"])}</strong><span>von 100 Punkten</span></div>',
        '<div class="ac-overview">',
        f'<p class="ac-must">Muss erfüllt {_escape(summary["muss_erfuellt"])}</p>',
        f'<p>{_escape(summary["fazit"])}</p></div></div>',
    ]
    for item in result["requirements"]:
        parts.extend([
            '<article class="ac-card">',
            '<div class="ac-badges">',
            f'<span class="ac-badge ac-type">{_escape("Muss" if item["typ"] == "muss" else "Kann")}</span>',
            f'<span class="ac-badge ac-{status_color(item)}">{_escape(STATUS_LABELS[item["status"]])}</span>',
            '</div>',
            f'<h3>{_escape(item["text"])}</h3>',
        ])
        if item["beleg"] is not None:
            parts.append(
                f'<blockquote><em>„{_escape(item["beleg"]["zitat"])}“</em>'
                f'<cite>S. {_escape(item["beleg"]["seite"])}</cite></blockquote>'
            )
        parts.append(f'<p>{_escape(item["begruendung"])}</p></article>')
    parts.append("</section>")
    return "".join(parts)


def _markdown_text(value: Any) -> str:
    # Dokumenttexte sollen im Export weder HTML noch Links/Markdown-Steuerzeichen ausführen.
    text = " ".join(str(value).split())
    text = re.sub(r"([\\`*_{}\[\]()#+.!|~>-])", r"\\\1", text)
    return html.escape(text, quote=True)


def result_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        f"# {TITLE}", "", f"Modus: {MODES[result['mode']]}", "",
        f"**Score: {summary['score']}/100**",
        f"**Muss erfüllt: {_markdown_text(summary['muss_erfuellt'])}**", "",
        _markdown_text(summary["fazit"]), "",
    ]
    for item in result["requirements"]:
        lines.extend([
            f"## {_markdown_text(item['id'])}: {_markdown_text(item['text'])}", "",
            f"**{'Muss' if item['typ'] == 'muss' else 'Kann'} · {STATUS_LABELS[item['status']]}**", "",
        ])
        if item["beleg"] is not None:
            lines.extend([
                f"> *„{_markdown_text(item['beleg']['zitat'])}“*",
                f"> S. {_markdown_text(item['beleg']['seite'])}", "",
            ])
        lines.extend([_markdown_text(item["begruendung"]), ""])
    return "\n".join(lines)


def error_message(error: Exception) -> str:
    """Feste deutsche Meldungen statt fremder Exception-Texte oder Stacktraces."""
    if isinstance(error, PdfWithoutText):
        return "Im PDF wurde zu wenig lesbarer Text gefunden. Bitte verwende ein PDF mit Textschicht; reine Scans können nicht ausgewertet werden."
    if isinstance(error, DocumentTooLong):
        return "Ein Dokument ist zu lang. Bitte verwende pro PDF höchstens 15 Seiten und 60.000 Zeichen."
    if isinstance(error, PdfExtractionError):
        return "Das PDF konnte nicht gelesen werden. Bitte prüfe, ob es eine gültige, unverschlüsselte PDF-Datei ist."
    if isinstance(error, MissingApiKeyError):
        return "Die Verbindung zu Anthropic ist noch nicht eingerichtet. Bitte informiere den Betreiber der App."
    if isinstance(error, RequestTooLarge):
        return "Die Dokumente ergeben eine zu große Anfrage für das Tokenlimit. Bitte kürze die Dokumente und versuche es erneut."
    if isinstance(error, OutputTokenLimit):
        return "Die Antwort hat das Ausgabelimit der API erreicht. Bitte verwende ein kürzeres Anforderungsdokument."
    if isinstance(error, LlmApiError):
        return {
            "timeout": "Die Auswertung hat zu lange gedauert. Bitte versuche es später erneut.",
            "connection": "Die Verbindung zu Anthropic wurde unterbrochen. Bitte versuche es später erneut.",
            "rate_limit": "Die API erhält gerade zu viele Anfragen. Bitte warte einen Moment und versuche es erneut.",
            "spending_limit": "Das API-Ausgabenlimit oder Guthaben ist ausgeschöpft. Bitte informiere den Betreiber der App.",
            "authentication": "Die API-Zugangsdaten konnten nicht bestätigt werden. Bitte informiere den Betreiber der App.",
            "request_too_large": "Die Anfrage ist für die API zu groß. Bitte verwende kürzere Dokumente.",
            "unavailable": "Anthropic ist momentan überlastet oder nicht erreichbar. Bitte versuche es später erneut.",
        }.get(error.reason, "Die API konnte die Anfrage nicht verarbeiten. Bitte versuche es später erneut.")
    if isinstance(error, TimeoutError):
        return "Die Auswertung hat zu lange gedauert. Bitte versuche es später erneut."
    if isinstance(error, LlmResponseError):
        return "Die API hat keine verwertbare Bewertung geliefert. Bitte versuche es erneut."
    if isinstance(error, LlmError):
        return "Bei der Auswertung mit Anthropic ist ein Fehler aufgetreten. Bitte versuche es später erneut."
    if isinstance(error, OSError):
        return "Die Beispieldateien konnten nicht gelesen werden. Bitte lade stattdessen zwei eigene PDFs hoch."
    return "Die Analyse konnte nicht abgeschlossen werden. Bitte versuche es erneut."


def clear_result() -> None:
    st.session_state.pop("result", None)


def change_mode() -> None:
    clear_result()
    st.session_state.use_example = False
    st.session_state.example_notice = False
    st.session_state.upload_generation += 1


def change_upload() -> None:
    clear_result()
    st.session_state.use_example = False
    st.session_state.example_notice = False


def select_example() -> None:
    if st.session_state.mode == "tender":
        st.session_state.example_notice = True
        return
    clear_result()
    st.session_state.use_example = True
    st.session_state.example_notice = False
    # Neue Widget-Keys leeren alte Uploads vor dem nächsten Rendern.
    st.session_state.upload_generation += 1


def read_documents(requirements_upload, profile_upload, use_example: bool) -> tuple[bytes, bytes]:
    if use_example:
        return (
            (EXAMPLES_DIR / "job_anforderungen.pdf").read_bytes(),
            (EXAMPLES_DIR / "job_profil.pdf").read_bytes(),
        )
    return requirements_upload.getvalue(), profile_upload.getvalue()


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="centered")
    st.title(TITLE)
    st.write("Lade ein Anforderungsdokument und ein Profil hoch. Die App findet jede Anforderung und prüft sie mit einem wörtlichen Beleg aus dem Profil.")
    st.caption("Deine Dateien werden nur für diese Analyse im Arbeitsspeicher verarbeitet und nicht gespeichert. Für die Auswertung wird der Text an die Claude API von Anthropic übermittelt.")

    for key, value in (("upload_generation", 0), ("use_example", False), ("example_notice", False)):
        st.session_state.setdefault(key, value)
    try:
        load_model()
    except Exception:
        st.error("Das Embedding-Modell konnte nicht geladen werden. Bitte versuche es später erneut.")
        return

    mode = st.radio("Was möchtest du prüfen?", options=list(MODES), format_func=MODES.get,
                    key="mode", on_change=change_mode)
    labels = UPLOAD_LABELS[mode]
    generation = st.session_state.upload_generation
    requirements_upload = st.file_uploader(
        f"{labels[0]} (PDF)", type=["pdf"], key=f"requirements_{generation}",
        on_change=change_upload, help="Eine PDF-Datei, maximal 5 MB.",
    )
    profile_upload = st.file_uploader(
        f"{labels[1]} (PDF)", type=["pdf"], key=f"profile_{generation}",
        on_change=change_upload, help="Eine PDF-Datei, maximal 5 MB.",
    )
    st.button("Beispiel laden", on_click=select_example)
    if st.session_state.example_notice:
        st.info("Das Beispiel für Ausschreibung und Firmenprofil folgt noch.")
    if st.session_state.use_example:
        st.info("Beispiel geladen: job_anforderungen.pdf und job_profil.pdf. Klicke auf „Analysieren“, um die Auswertung zu starten.")

    ready = st.session_state.use_example or (
        requirements_upload is not None and profile_upload is not None
    )
    if st.button("Analysieren", type="primary", disabled=not ready):
        clear_result()
        with st.status("Analyse wird vorbereitet …", expanded=True) as status:
            try:
                documents = read_documents(requirements_upload, profile_upload, st.session_state.use_example)

                def on_progress(stage: pipeline.ProgressStage) -> None:
                    label = STAGES[stage]
                    status.update(label=f"{label} …")
                    status.write(label)

                result = pipeline.analyze(*documents, mode, on_progress=on_progress)
            except Exception as error:
                status.update(label="Analyse nicht abgeschlossen", state="error", expanded=False)
                st.error(error_message(error))
            else:
                st.session_state.result = result
                status.update(label="Analyse abgeschlossen", state="complete", expanded=False)

    if st.session_state.get("result") is not None:
        result = st.session_state.result
        st.subheader("Dein Ergebnis")
        st.markdown(render_result_html(result), unsafe_allow_html=True)
        st.download_button(
            "Als Markdown herunterladen", data=result_markdown(result),
            file_name="anforderungs-check.md", mime="text/markdown; charset=utf-8",
            on_click="ignore",
        )


if __name__ == "__main__":
    main()
