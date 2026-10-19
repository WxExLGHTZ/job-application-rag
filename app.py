"""Minimaler Streamlit-Deploy-Test ohne Profil-Retrieval oder Claude-Aufrufe."""

from __future__ import annotations

import os
from time import perf_counter

import psutil
import streamlit as st
from sentence_transformers import SentenceTransformer

from copilot.index import get_embedding_model


TITLE = "Anforderungs-Check (Deploy-Test)"


@st.cache_resource(show_spinner="Embedding-Modell wird geladen ...")
def load_model() -> SentenceTransformer:
    """Lade das CPU-Modell einmal und verwende es bei Streamlit-Reruns wieder."""
    return get_embedding_model()


def main() -> None:
    """Zeige Embedding-Test, Secret-Status und aktuellen Prozessspeicher an."""
    st.set_page_config(page_title=TITLE)
    st.title(TITLE)
    st.caption(" | ".join(
        f"{name} gesetzt: {'ja' if os.environ.get(name) else 'nein'}"
        for name in ("ANTHROPIC_API_KEY", "ACCESS_CODE")
    ))

    try:
        model = load_model()
    except Exception as error:
        st.error(f"Modell konnte nicht geladen werden ({type(error).__name__}).")
        return

    text = st.text_area("Text", placeholder="Text für den Embedding-Test")
    if st.button("Embedding testen", type="primary"):
        if not text.strip():
            st.info("Bitte gib zuerst einen Text ein.")
        else:
            # Das Modell ist schon geladen; gemessen wird nur die Berechnung.
            start = perf_counter()
            try:
                vector = model.encode(text.strip(), show_progress_bar=False)
            except Exception as error:
                # Keine Dokumentinhalte oder fremden Exception-Texte ausgeben.
                st.error(f"Embedding fehlgeschlagen ({type(error).__name__}).")
            else:
                duration_ms = (perf_counter() - start) * 1000
                st.write(f"Vektordimension: {vector.shape[0]} | Dauer: {duration_ms:.1f} ms")

    # RSS ist der aktuell im RAM belegte Speicher des gesamten Python-Prozesses.
    memory_mb = psutil.Process().memory_info().rss / 1_000_000
    st.metric("Arbeitsspeicher des Prozesses (RSS)", f"{memory_mb:.1f} MB")


if __name__ == "__main__":
    main()
