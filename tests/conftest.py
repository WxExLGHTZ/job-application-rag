"""Synthetische PDFs und Offline-Schutz für die Pipeline-Tests."""

from io import BytesIO
import socket

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Tests dürfen keine Netzwerkverbindung öffnen.")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


@pytest.fixture
def pdf_bytes():
    """Erzeuge echte, einfache Text-PDFs ausschließlich im Speicher."""
    def create(texts):
        writer = PdfWriter()
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        for text in texts:
            page = writer.add_blank_page(width=600, height=800)
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 10 Tf 10 700 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = stream
        output = BytesIO()
        writer.write(output)
        return output.getvalue()
    return create
