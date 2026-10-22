"""Streamlit-Oberfläche für die Analyse zweier PDFs im Arbeitsspeicher."""

from __future__ import annotations

import base64
import html
import json
import math
import logging
import os
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import streamlit as st

from copilot import pipeline
from copilot.access import AccessError, AnalysisLimits, SessionAccess
from copilot.index import get_embedding_model
from copilot.ingest import DocumentTooLong, PdfExtractionError, PdfWithoutText
from copilot.llm import (
    LlmApiError, LlmError, LlmResponseError, MissingApiKeyError,
    OutputTokenLimit, RequestTooLarge, UsageCapture, capture_token_usage,
)
from copilot.pricing import calculate_cost_usd
from copilot.schema import AnalysisResult


TITLE = "Anforderungs-Check"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
MODES = {"job": "Stellenanzeige und Lebenslauf", "tender": "Ausschreibung und Firmenprofil"}
UPLOAD_LABELS = {"job": ("Stellenanzeige", "Lebenslauf"), "tender": ("Ausschreibung", "Firmenprofil")}
STAGES = {
    "text": "Text lesen", "requirements": "Anforderungen finden",
    "evidence": "Belege suchen", "assessment": "Bewerten",
}
STATUS_LABELS = {"erfuellt": "erfüllt", "teilweise": "teilweise", "nicht_belegt": "nicht belegt"}
INVITATION_NOTICE = "Die Live-Analyse ist auf eingeladene Besucher begrenzt. Unten siehst du ein vollständiges Beispielergebnis."
logger = logging.getLogger(__name__)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
REPO_URL = "https://github.com/WxExLGHTZ/job-application-rag"
FILTERS = ("Alle", "Muss", "Lücken")
LOGO_SVG = '<svg class="ac-logo" viewBox="0 0 36 36" fill="none" aria-hidden="true"><rect width="36" height="36" rx="9" fill="#1E40AF"/><path d="M12 9.5h8l5 5V27H12V9.5Z" stroke="white" stroke-width="1.5" stroke-linejoin="round"/><path d="M20 9.5V15h5M15.5 20l2 2 4.5-4.5" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
LOCK_SVG = '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><rect x="3.5" y="7" width="9" height="7" rx="1.5" stroke="currentColor" stroke-width="1.2"/><path d="M5.5 7V4.5a2.5 2.5 0 0 1 5 0V7M8 10v1" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>'
FILE_SVG = '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4 2h5l3 3v9H4V2Z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/><path d="M9 2v3h3M6 8h4M6 10.5h3" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>'


@lru_cache(maxsize=1)
def app_styles() -> str:
    """Nur statische Styles und die lokale Schriftdatei werden gemeinsam gecacht."""
    font = base64.b64encode((ASSETS_DIR / "fonts" / "InterVariable.woff2").read_bytes()).decode("ascii")
    css = (ASSETS_DIR / "app.css").read_text(encoding="utf-8")
    return "<style>" + css.replace("__INTER_FONT__", "data:font/woff2;base64," + font) + "</style>"


@st.cache_resource(show_spinner="Embedding-Modell wird geladen …")
def load_model():
    """Nur das CPU-Modell wird sitzungsübergreifend wiederverwendet."""
    return get_embedding_model()


@st.cache_resource
def get_limits() -> AnalysisLimits:
    """Flüchtige, prozessweite Zähler; ein Serverneustart setzt sie zurück."""
    return AnalysisLimits()


def access_session() -> SessionAccess:
    session = st.session_state.setdefault("access_session", SessionAccess())
    codes = st.query_params.get_all("code")
    # Mehrdeutige Links mit mehreren Codes werden ebenfalls abgelehnt.
    session.refresh(codes[0] if len(codes) == 1 else "" if codes else None)
    return session


def load_example_result() -> dict[str, Any]:
    text = (EXAMPLES_DIR / "result_job.json").read_text(encoding="utf-8-sig")
    return AnalysisResult.model_validate_json(text).model_dump()


def run_live_analysis(requirements_upload, profile_upload, mode, use_example, *, on_progress):
    """Alle Startwege durch dieselbe serverseitige Zulassung führen."""
    session = access_session()
    with get_limits().analysis(session) as permit:
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise MissingApiKeyError("ANTHROPIC_API_KEY ist nicht gesetzt.")
        documents = read_documents(requirements_upload, profile_upload, use_example)
        try:
            result = pipeline.analyze(*documents, mode, on_progress=on_progress)
        except Exception:
            permit.check()
            raise
        permit.check()
        return result


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



def analysis_cost_usd(capture: UsageCapture) -> str | None:
    """Messdaten dürfen ein erfolgreiches Analyseergebnis niemals verhindern."""
    if not capture.complete or not capture.by_model:
        return None
    try:
        return str(sum(
            (calculate_cost_usd(model, usage.to_dict())
             for model, usage in capture.by_model.items()), Decimal(0),
        ))
    except Exception:
        return None


def format_duration(metadata: dict[str, Any] | None) -> str:
    value = (metadata or {}).get("duration_seconds")
    if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
        return f"{value:.1f}".replace(".", ",") + " s"
    return "—"


def format_cost(metadata: dict[str, Any] | None) -> str:
    value = (metadata or {}).get("cost_usd")
    if value is not None:
        try:
            amount = Decimal(str(value))
            if amount.is_finite() and amount >= 0:
                return f"{amount:.4f}".replace(".", ",") + " USD"
        except (InvalidOperation, ValueError):
            pass
    return "—"


def filtered_requirements(result: dict[str, Any], filter_name: str = "Alle") -> list[dict[str, Any]]:
    """Nur die Darstellung sortieren/filtern; Ergebnis und Exporte bleiben vollständig."""
    items = [
        item for item in result["requirements"]
        if (filter_name != "Muss" or item["typ"] == "muss")
        and (filter_name != "Lücken" or item["status"] != "erfuellt")
    ]
    return sorted(items, key=lambda item:
                  0 if item["typ"] == "muss" and item["status"] == "nicht_belegt"
                  else 1 if item["typ"] == "muss" else 2)


def status_badge(item: dict[str, Any]) -> str:
    color = status_color(item)
    label = STATUS_LABELS[item["status"]]
    if color == "red":
        label += " · kritisch"
    return (f'<span class="ac-status ac-{color}"><span class="ac-dot" aria-hidden="true"></span>'
            f'{_escape(label)}</span>')


def render_dashboard_html(result: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    summary = result["summary"]
    score = max(0, min(100, int(summary["score"])))
    ring_color = {"green": "#339265", "yellow": "#c4932e", "red": "#c4635c"}[score_color(score)]
    counts = {status: sum(item["status"] == status for item in result["requirements"])
              for status in STATUS_LABELS}
    total = max(1, len(result["requirements"]))
    distributions = "".join(
        f'<span style="width:{_escape(str(counts[status] * 100 / total))}%;background:{color}"></span>'
        for status, color in (("erfuellt", "#6ea884"), ("teilweise", "#dbb65d"),
                              ("nicht_belegt", "#c9cfd9"))
    )
    legend = "".join(
        f'<span><i class="ac-dot" style="color:{color}" aria-hidden="true"></i>'
        f'{_escape(STATUS_LABELS[status])}<b>{_escape(counts[status])}</b></span>'
        for status, color in (("erfuellt", "#6ea884"), ("teilweise", "#d4a94d"),
                              ("nicht_belegt", "#aeb8c7"))
    )
    return (
        '<section class="ac-result" aria-label="Zusammenfassung der Analyse"><div class="ac-metrics">'
        '<div class="ac-metric"><div class="ac-metric-label">Gesamtscore</div>'
        '<div class="ac-score-row">'
        f'<div class="ac-score-ring" style="--ac-ring:{ring_color};--ac-progress:{_escape(score)}%" '
        f'role="img" aria-label="Gesamtscore: {_escape(score)} von 100">'
        f'<div class="ac-score-inner">{_escape(score)}</div></div>'
        '<div class="ac-score-caption"><strong>von 100</strong>Gewichtete Passung</div></div></div>'
        '<div class="ac-metric"><div class="ac-metric-label">Muss-Kriterien</div>'
        f'<div class="ac-metric-value">{_escape(summary["muss_erfuellt"])}</div>'
        '<div class="ac-metric-note">vollständig erfüllt</div></div>'
        '<div class="ac-metric"><div class="ac-metric-label">Bewertungen im Überblick</div>'
        f'<div class="ac-distribution" aria-hidden="true">{distributions}</div>'
        f'<div class="ac-distribution-key">{legend}</div></div>'
        '<div class="ac-metric"><div class="ac-metric-label">Dauer &amp; Kosten</div>'
        f'<div class="ac-time">{_escape(format_duration(metadata))}</div>'
        f'<div class="ac-cost"><span>{_escape(format_cost(metadata))}</span><small>API-Kosten</small></div>'
        '<div class="ac-metric-note">Für diese Analyse</div></div></div>'
        '<div class="ac-verdict"><div class="ac-verdict-label">Fazit</div>'
        f'<p>{_escape(summary["fazit"])}</p></div></section>'
    )


def render_requirements_html(result: dict[str, Any], filter_name: str = "Alle") -> str:
    items = filtered_requirements(result, filter_name)
    if not items:
        return '<div class="ac-table-wrap ac-no-matches">Keine Anforderungen für diesen Filter.</div>'
    rows = []
    for item in items:
        evidence = '<span style="color:#98a0ae">Kein Beleg vorhanden</span>'
        if item["beleg"] is not None:
            evidence = (
                f'<blockquote><em>„{_escape(item["beleg"]["zitat"])}“</em></blockquote>'
                f'<span class="ac-verified">im Dokument verifiziert · S. {_escape(item["beleg"]["seite"])}</span>'
            )
        elif item["begruendung"] == "Beleg im Dokument nicht gefunden":
            evidence = '<span class="ac-evidence-missing">Beleg nicht gefunden</span>'
        rows.append(
            f'<tr><td class="ac-requirement" data-label="Anforderung">'
            f'<span class="ac-row-id">{_escape(item["id"])}</span>{_escape(item["text"])}</td>'
            f'<td data-label="Typ"><span class="ac-type">{_escape("Muss" if item["typ"] == "muss" else "Kann")}</span></td>'
            f'<td data-label="Status">{status_badge(item)}</td>'
            f'<td data-label="Beleg">{evidence}</td>'
            f'<td data-label="Begründung">{_escape(item["begruendung"])}</td></tr>'
        )
    return (
        '<div class="ac-result ac-table-wrap"><table class="ac-table" aria-label="Anforderungen und Belege">'
        '<colgroup><col style="width:25%"><col style="width:7%"><col style="width:16%">'
        '<col style="width:29%"><col style="width:23%"></colgroup>'
        '<thead><tr><th scope="col">Anforderung</th><th scope="col">Typ</th>'
        '<th scope="col">Status</th><th scope="col">Beleg</th><th scope="col">Begründung</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table></div>'
    )


def render_result_html(result: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    return render_dashboard_html(result, metadata) + render_requirements_html(result)


def render_file_note(name: str) -> str:
    return f'<div class="ac-file">{FILE_SVG}<span title="{_escape(name)}">{_escape(name)}</span></div>'


def render_step(number: int, title: str) -> None:
    st.markdown(f'<div class="ac-step"><span class="ac-step-number">{_escape(number)}</span>'
                f'<span>{_escape(title)}</span></div>', unsafe_allow_html=True)


def _markdown_text(value: Any, *, line_start: bool = False) -> str:
    """Erhalte Satzzeichen; schütze nur wirksame Markdown-Syntax und HTML."""
    text = " ".join(str(value).split())
    # Backslashes nur vor Zeichen verdoppeln, die Markdown sonst entschärft.
    text = re.sub(r"\\(?=[\\`*_{}\[\]()#+.!|~>\-])", r"\\\\", text)
    text = re.sub(r"([`*])|(?<!\w)(_+)|(_+)(?!\w)|(~~+)",
                  lambda match: "".join("\\" + char for char in match[0]), text)
    # Eine geschützte öffnende Klammer reicht gegen Links und Bilder.
    text = re.sub(r"\[(?=[^\]]*\]\s*(?:\(|\[))", r"\\[", text)
    if line_start:
        text = re.sub(r"^(#{1,6}(?=\s|$)|[-+](?=\s)|\[(?=[^\]]*\]:))",
                      r"\\\1", text)
        text = re.sub(r"^(\d{1,9})([.)])(?=\s)", r"\1\\\2", text)
        if re.fullmatch(r"(?:-\s*){3,}", text):
            text = "\\" + text
    return html.escape(text, quote=False)


def result_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        f"# {TITLE}", "", f"Modus: {MODES[result['mode']]}", "",
        f"**Score: {summary['score']}/100**",
        f"**Muss erfüllt: {_markdown_text(summary['muss_erfuellt'])}**", "",
        _markdown_text(summary["fazit"], line_start=True), "",
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
        lines.extend([_markdown_text(item["begruendung"], line_start=True), ""])
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
    st.session_state.pop("analysis_meta", None)
    st.session_state.pop("result_filter", None)


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



def render_result_panel(result: dict[str, Any], metadata=None, *, example: bool = False) -> None:
    heading = "Beispielergebnis" if example else "Deine Auswertung"
    st.markdown(f'<div class="ac-section-title"><h2>{_escape(heading)}</h2>'
                f'<span>{_escape(len(result["requirements"]))} Anforderungen geprüft</span></div>',
                unsafe_allow_html=True)
    st.markdown(render_dashboard_html(result, metadata), unsafe_allow_html=True)
    st.markdown('<div class="ac-section-title"><h2>Anforderungen im Detail</h2>'
                '<span>Nach Priorität sortiert</span></div>', unsafe_allow_html=True)
    selected_filter = st.segmented_control(
        "Anforderungen filtern", FILTERS, default="Alle", required=True, key="result_filter",
        label_visibility="collapsed", width=264,
    )
    st.markdown(render_requirements_html(result, selected_filter), unsafe_allow_html=True)
    with st.container(key="export_actions", horizontal=True, gap="small"):
        st.download_button(
            "Markdown", data=result_markdown(result), file_name="anforderungs-check.md",
            mime="text/markdown; charset=utf-8", on_click="ignore",
        )
        st.download_button(
            "JSON", data=json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            file_name="anforderungs-check.json", mime="application/json", on_click="ignore",
        )


def render_footer() -> None:
    st.markdown(
        '<div class="ac-footer"><span>Gebaut von WxExLGHTZ</span>'
        f'<a href="{REPO_URL}" target="_blank" rel="noopener noreferrer">Projekt auf GitHub</a></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(app_styles(), unsafe_allow_html=True)
    st.markdown(
        f'<header class="ac-topbar"><div class="ac-brand">{LOGO_SVG}'
        f'<span>{_escape(TITLE)}</span></div><span class="ac-demo">Demo</span></header>'
        '<section class="ac-intro"><div class="ac-eyebrow">Dokumente vergleichen. Belege verstehen.</div>'
        '<h1>Anforderungen klar prüfen.</h1>'
        '<p class="ac-lead">Vergleiche Stellenanzeigen oder Ausschreibungen mit einem Profil – '
        'Anforderung für Anforderung, mit wörtlichen Belegen.</p>'
        f'<div class="ac-privacy">{LOCK_SVG}<span>'
        'Deine Dateien werden nur für diese Analyse im Arbeitsspeicher verarbeitet und nicht gespeichert. '
        'Für die Auswertung wird der Text an die Claude API von Anthropic übermittelt.'
        '</span></div></section>', unsafe_allow_html=True,
    )

    for key, value in (("upload_generation", 0), ("use_example", False), ("example_notice", False)):
        st.session_state.setdefault(key, value)
    session = access_session()
    if not session.authorized:
        if st.session_state.pop("live_authorized", False) or st.session_state.get("result") is not None:
            clear_result()
        st.session_state.use_example = False
        st.session_state.example_notice = False
        for key in list(st.session_state):
            if key.startswith(("requirements_", "profile_")):
                st.session_state.pop(key, None)
        with st.container(key="input_card"):
            st.info(INVITATION_NOTICE)
            st.button("Analysieren", key="analyze_button", type="primary", disabled=True, width="stretch")
        try:
            example_result = load_example_result()
        except Exception as error:
            logger.warning("error_type=%s", type(error).__name__)
            st.error("Das Beispielergebnis konnte nicht geladen werden. Bitte versuche es später erneut.")
        else:
            render_result_panel(example_result, example=True)
        render_footer()
        return

    st.session_state.live_authorized = True
    try:
        load_model()
    except Exception:
        st.error("Das Embedding-Modell konnte nicht geladen werden. Bitte versuche es später erneut.")
        return

    with st.container(key="input_card"):
        st.markdown('<div class="ac-card-title"><h2>Analyse vorbereiten</h2>'
                    '<span>Zwei Dokumente. Ein klarer Überblick.</span></div>', unsafe_allow_html=True)
        with st.container(key="step_mode"):
            render_step(1, "Vergleich auswählen")
            mode = st.segmented_control(
                "Was möchtest du prüfen?", options=list(MODES), format_func=MODES.get,
                default="job", required=True, key="mode", on_change=change_mode,
                label_visibility="collapsed", width="stretch",
            )
        labels = UPLOAD_LABELS[mode]
        generation = st.session_state.upload_generation
        with st.container(key="step_documents"):
            render_step(2, "Dokumente hinzufügen")
            with st.container(key="document_uploads"):
                left, right = st.columns(2, gap="medium")
                with left:
                    requirements_upload = st.file_uploader(
                        f"{labels[0]} (PDF)", type=["pdf"], key=f"requirements_{generation}",
                        on_change=change_upload, help="Eine PDF-Datei, maximal 5 MB.",
                    )
                    if st.session_state.use_example:
                        st.markdown(render_file_note("job_anforderungen.pdf"), unsafe_allow_html=True)
                    elif requirements_upload is not None:
                        st.markdown(render_file_note(requirements_upload.name), unsafe_allow_html=True)
                with right:
                    profile_upload = st.file_uploader(
                        f"{labels[1]} (PDF)", type=["pdf"], key=f"profile_{generation}",
                        on_change=change_upload, help="Eine PDF-Datei, maximal 5 MB.",
                    )
                    if st.session_state.use_example:
                        st.markdown(render_file_note("job_profil.pdf"), unsafe_allow_html=True)
                    elif profile_upload is not None:
                        st.markdown(render_file_note(profile_upload.name), unsafe_allow_html=True)
            st.button("Beispiel laden", key="example_button", type="secondary", on_click=select_example)
            if st.session_state.example_notice:
                st.info("Das Beispiel für Ausschreibung und Firmenprofil folgt noch.")
            if st.session_state.use_example:
                st.caption("Fiktive Beispieldateien ausgewählt. Du kannst die Analyse jetzt starten.")

        with st.container(key="step_analyze"):
            render_step(3, "Anforderungen prüfen")
            ready = st.session_state.use_example or (
                requirements_upload is not None and profile_upload is not None
            )
            clicked = st.button("Analysieren", key="analyze_button", type="primary", disabled=not ready, width="stretch")
            if clicked and ready:
                with st.status("Analyse wird vorbereitet …", expanded=True) as status:
                    started = perf_counter()
                    try:
                        def on_progress(stage: pipeline.ProgressStage) -> None:
                            label = STAGES[stage]
                            status.update(label=f"{label} …")
                            status.write(label)

                        with capture_token_usage() as capture:
                            result = run_live_analysis(
                                requirements_upload, profile_upload, mode, st.session_state.use_example,
                                on_progress=on_progress,
                            )
                    except AccessError as error:
                        logger.warning("error_type=%s", type(error).__name__)
                        status.update(label="Analyse nicht abgeschlossen", state="error", expanded=False)
                        st.info(str(error))
                    except Exception as error:
                        clear_result()
                        logger.warning("error_type=%s", type(error).__name__)
                        status.update(label="Analyse nicht abgeschlossen", state="error", expanded=False)
                        st.error(error_message(error))
                    else:
                        clear_result()
                        st.session_state.result = result
                        st.session_state.analysis_meta = {
                            "duration_seconds": perf_counter() - started,
                            "cost_usd": analysis_cost_usd(capture),
                        }
                        status.update(label="Analyse abgeschlossen", state="complete", expanded=False)
            st.markdown('<p class="ac-input-hint">Jede Bewertung wird mit Belegen aus deinem Profil geprüft.</p>',
                        unsafe_allow_html=True)

    if st.session_state.get("result") is not None:
        render_result_panel(st.session_state.result, st.session_state.get("analysis_meta"))
    else:
        st.markdown(
            '<section class="ac-empty"><h2>Von Anforderungen zu einer fundierten Entscheidung.</h2>'
            '<p>Lade zwei PDFs hoch oder starte mit „Beispiel laden”. Hier erscheint deine Auswertung.</p>'
            '<div class="ac-empty-statuses">'
            '<span><i class="ac-dot" style="color:#5f9e76" aria-hidden="true"></i>Erfüllt · vollständig belegt</span>'
            '<span><i class="ac-dot" style="color:#cda34d" aria-hidden="true"></i>Teilweise · zum Teil belegt</span>'
            '<span><i class="ac-dot" style="color:#aeb8c7" aria-hidden="true"></i>Nicht belegt · Nachweis fehlt</span>'
            '</div></section>', unsafe_allow_html=True,
        )
    render_footer()


if __name__ == "__main__":
    main()
