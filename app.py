"""Minimaler Gradio-Deploy-Test ohne Profil-Retrieval oder Claude-Aufrufe."""

from __future__ import annotations

import os
from time import perf_counter

import gradio as gr

from copilot.index import embed_texts, get_embedding_model


TITLE = "Anforderungs-Check (Deploy-Test)"


def test_embedding(text: str) -> str:
    """Erzeuge ein Embedding und liefere Dimension und Rechenzeit in Millisekunden."""
    if not text.strip():
        return "Bitte gib zuerst einen Text ein."

    # Das Modell ist beim App-Start bereits geladen; gemessen wird das Embedding.
    start = perf_counter()
    try:
        vectors = embed_texts([text.strip()])
    except Exception as error:
        # Keine Eingabetexte oder fremden Exception-Nachrichten ausgeben.
        return f"Embedding fehlgeschlagen ({type(error).__name__})."
    duration_ms = (perf_counter() - start) * 1000
    return f"Vektordimension: {vectors.shape[1]} | Dauer: {duration_ms:.1f} ms"


def build_app() -> gr.Blocks:
    """Lade das CPU-Modell einmal und erstelle die Oberfläche für den Deploy-Test."""
    get_embedding_model()
    secret_status = " | ".join(
        f"{name} gesetzt: {'ja' if os.environ.get(name) else 'nein'}"
        for name in ("ANTHROPIC_API_KEY", "ACCESS_CODE")
    )

    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(secret_status)
        text = gr.Textbox(label="Text", placeholder="Text für den Embedding-Test", lines=3)
        button = gr.Button("Embedding testen", variant="primary")
        result = gr.Textbox(label="Ergebnis", interactive=False)
        button.click(fn=test_embedding, inputs=text, outputs=result)

    return demo


def main() -> None:
    """Starte den Gradio-Webserver mit vorab geladenem Embedding-Modell."""
    demo = build_app()
    demo.launch(share=False, run_history=False, footer_links=[])


if __name__ == "__main__":
    main()
