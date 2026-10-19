# Bestandsaufnahme des Bewerbungs-Copiloten

Stand: 19.10.2026. Grundlage ist der gelesene Anwendungscode, ergänzt durch
`README.md`, `PLAN.md` und `requirements.txt`. Es wurden keine Profil-PDFs oder
Secrets gelesen, keine Claude-Aufrufe ausgeführt und keine Modelle geladen.
Diese Bestandsaufnahme verändert keine Anwendungslogik.

Die neue `AGENTS.md` beschreibt das Ziel mit Gradio. Der Code ist noch eine
lokale Streamlit-Anwendung; `PLAN.md` und `README.md` beschreiben ebenfalls den
bisherigen Stand. Die folgenden Abschnitte trennen vorhandene Funktionen von
erst erforderlichen Änderungen.

## 1. Module, Funktionen und Datenfluss

Mit „Stufe 1“ und „Stufe 2“ sind hier die beiden LLM-Schritte gemeint, nicht die
Entwicklungsstufen des alten `PLAN.md`: Anforderungen werden mit einem
Claude-Aufruf extrahiert und nach dem Retrieval mit einem zweiten bewertet.

```text
Stellenanzeige als Text (UI) oder TXT/PDF (CLI)
  → extract_requirements()                         [Stufe 1 / Schritt A]
  → pro Anforderung retrieve(..., top_k=5)         [Retrieval / Schritt B]
      ← profile/*.pdf → Text → Chunks → Embeddings → NumPy-Index
  → assess_requirements(requirements_with_context) [Stufe 2 / Schritt C]
  → Ergebnis-Dict → Streamlit oder JSON auf stdout
```

### PDF-Text und Chunking: `copilot/ingest.py`

| Funktion | Aktuelle Aufgabe |
| --- | --- |
| `extract_text_from_pdf(path)` | Liest eine einzelne PDF und liefert ihren Text. Wird vom CLI für eine Stellenanzeige verwendet. |
| `_extract_pdf_text_and_page_count(path)` | Prüft den Dateipfad, öffnet ihn mit `pypdf.PdfReader`, ruft `page.extract_text()` auf und verbindet die Seitentexte mit Leerzeilen. Liefert Text und Seitenzahl. |
| `load_profile(profile_dir="profile")` | Liest sortiert `*.pdf` direkt aus dem Ordner, nicht rekursiv. Liefert je Dokument `filename`, `text`, `num_pages`, `num_chars`. Unlesbare PDFs werden mit Warnung übersprungen; fehlender/falscher Ordner ergibt eine Warnung und eine leere Liste. |
| `chunk_text(text, chunk_size=800, overlap=150)` | Zerlegt Fließtext in überlappende Abschnitte. Größen sind **Zeichen**, keine Tokens. Leerer Text ergibt keine Chunks; ungültige Größen lösen `ValueError` aus. |
| `_find_chunk_end()`, `_find_overlap_start()` | Bevorzugen Absatz-, dann Zeilen-, dann Leerraumgrenzen; harter Schnitt als Rückfall. Die Überlappung wird an Wortgrenzen angepasst und ist daher nicht immer exakt 150 Zeichen. |
| `_chunk_document(text)` | Kombiniert normales Chunking des Resttexts mit besonderen Modul-Chunks. |
| `_extract_module_chunks()`, `_looks_like_module_grade_line()`, `_format_module_chunk()` | Erkennen Notentabellen heuristisch anhand Semester, Note/Bewertung und Tabellenkürzeln. Erst ab drei passenden Zeilen greift die Sonderbehandlung. Pro erkanntem Modul entsteht ein normalisierter Chunk inklusive Originalzeile. |
| `build_chunks(profile_dir="profile")` | Verbindet Ordnerlesen und Chunking. Liefert Dictionaries mit `chunk_id`, `filename`, `text`, `kind` (`text` oder `module`). |

Weitere Hilfsfunktionen suchen Textgrenzen, normalisieren Leerraum und verbinden
gegebenenfalls eine vorherige Zeile mit einem Modulnamen. Die Modulheuristik ist
kein allgemeiner Parser für beliebige Tabellen oder Ausschreibungen.

`PdfExtractionError` meldet fehlende, unlesbare, seitenlose oder textlose PDFs.
Einzelne fehlerhafte Seiten werden mit Warnung übersprungen. OCR ist nicht
vorhanden: reine Scan-PDFs ohne Textschicht können nicht ausgewertet werden.
Seitenzahlen werden gezählt, aber nicht als Herkunft pro Chunk erhalten.

### Embeddings und Retrieval: `copilot/index.py`

| Funktion/Klasse | Aktuelle Aufgabe |
| --- | --- |
| `get_embedding_model()` | Lädt das SentenceTransformer-Modell; `lru_cache(maxsize=1)` hält es einmal pro Prozess vor. Ein Gerät wird nicht explizit angegeben. |
| `embed_texts(texts)` | Ruft `model.encode(..., convert_to_numpy=True, show_progress_bar=False)` auf und liefert `float32`-Vektoren. |
| `normalize_vectors(vectors)` | Normalisiert jede Zeile auf Länge 1; schützt vor Division durch null. |
| `ProfileIndex(chunks, embeddings)` | Hält Chunks und normalisierte NumPy-Matrix im Arbeitsspeicher. Der Konstruktor kann bereits mit eigens aufgebauten Chunks verwendet werden. |
| `ProfileIndex.from_profile(profile_dir)` / `build_index(profile_dir)` | Bauen einen neuen Index aus den PDFs eines Ordners. |
| `ProfileIndex.retrieve(query, top_k=5)` | Embeddet die Anfrage, normalisiert sie, berechnet `self.embeddings @ query_vector` und sortiert absteigend. Liefert höchstens `top_k` Treffer mit `filename`, `text`, `score`. |
| Modulweite Funktion `retrieve(query, top_k=5)` | Erstellt beim ersten Aufruf `_DEFAULT_INDEX` aus `profile/` und verwendet diesen globalen Index danach weiter. |

Ein Embedding ist eine numerische Darstellung von Text. Bei normierten Vektoren
entspricht das Skalarprodukt der Cosine Similarity. Top-k wählt die k ähnlichsten
Chunks; der Score ist kein Beweis und keine Erfüllungswahrscheinlichkeit.

Es gibt keine Vektor-Datenbank, keinen auf Platte gespeicherten Profilindex,
keinen Reranker und keine Mindestscore-Grenze. `chunk_id` und `kind` werden nicht
in den Retrieval-Ergebnissen weitergereicht. Ein leerer Index liefert keine
Treffer, wird von der Pipeline aber nicht gesondert als fehlendes Profil abgelehnt.
Dateiänderungen in `profile/` aktualisieren den globalen Index nicht automatisch.

### Stufe 1 und Stufe 2: `copilot/llm.py`

| Funktion | Aktuelle Aufgabe |
| --- | --- |
| `extract_requirements(job_ad_text)` | Schickt den gesamten Anzeigentext an Claude. Erwartet eine JSON-Liste mit Anforderungen. Prüft Listentyp, Objekte und nichtleeres Textfeld `requirement`. |
| `assess_requirements(requirements_with_context)` | Schickt alle Anforderungen samt jeweils gefundenen Chunks gemeinsam an Claude. Erwartet die Gesamtbewertung als JSON-Objekt. |
| `_get_client()` | Liest `ANTHROPIC_API_KEY` ausschließlich über `os.environ` und hält einen `Anthropic`-Client im Prozess vor. |
| `_send_message(system_prompt, user_prompt)` | Verwendet `client.messages.create()`, sammelt Textblöcke und erkennt leere bzw. durch `max_tokens` abgeschnittene Antworten. |
| `_send_and_parse_with_retry(...)` | Wiederholt bei `LlmResponseError` einmal mit demselben Prompt. Bei erfolgreichem Ablauf zwei Generierungsaufrufe insgesamt; mit diesen Wiederholungen bis zu vier. Mögliche SDK-interne Netzwerk-Retries sind davon unabhängig. |
| `_parse_json_response()`, `_remove_outer_code_fence()`, `_find_json_end()` | Entfernen äußere Markdown-Fences und suchen ein parsebares JSON-Objekt bzw. eine Liste auch in Begleittext. Berücksichtigen verschachtelte Klammern, Strings und Escape-Zeichen. |

Fehlerklassen: `LlmError`, `MissingApiKeyError` und `LlmResponseError`.
Das gewünschte Schema wird im Prompt beschrieben, aber nicht vollständig im Code
validiert: Stufe 1 prüft weder Kategorien noch `reason`; Stufe 2 prüft nur, ob
das Ergebnis ein Dictionary ist. Statuswerte, vier Hauptteile, Vollständigkeit
der Anforderungen und Übereinstimmung von Belegen mit gelieferten Chunks werden
nicht programmatisch abgesichert. „Belegbasiert“ ist derzeit vor allem eine
Prompt-Regel.

### Ablauf, CLI und Oberfläche

- `copilot/pipeline.py`: `analyze_job_ad(job_ad_text, top_k=5)` prüft Eingabetext
  und Top-k, extrahiert Anforderungen, ruft je Anforderung das globale Retrieval
  auf und ergänzt `retrieved_chunks`. Anschließend erfolgt eine gemeinsame
  Bewertung. Die Funktion akzeptiert weder ein hochgeladenes Profil noch einen
  eigenen Index oder einen Modus.
- `copilot/cli.py`: `main()` nimmt einen Dateipfad und optional `--top-k` an.
  `_load_job_ad()` liest UTF-8-TXT oder PDF. Ergebnis: formatiertes JSON auf
  stdout; erwartete Fehler: stderr und Exitcode 1.
- `app.py`: `main()` baut eine Streamlit-Oberfläche mit Textfeld und
  „Analysieren“-Button. `render_result()` ruft `render_requirement_matrix()`,
  `render_framing_suggestions()`, `render_highlighted_projects()` und
  `render_interview_questions()` auf. `format_evidence()`, `as_list()` und
  `text_or_fallback()` helfen bei toleranter Darstellung. Zusätzlich ist das
  rohe JSON aufklappbar. Es gibt keinen PDF-Upload und keine Moduswahl.
- `inspect_extraction.py`: `main()` gibt Dateinamen, Zähler und bis zu 800
  Zeichen Dokumenttext pro Profil-PDF aus.
- `inspect_retrieval.py`: `main()` baut einen Index und zeigt zu vier festen
  Testfragen jeweils bis zu fünf Treffer samt Dateiname, Score und Textauszug;
  `preview_text()` und `configure_output_encoding()` unterstützen die Ausgabe.
- `copilot/__init__.py` markiert das Paket. Eine automatisierte Testsuite wurde
  im untersuchten Projektbestand nicht gefunden.

## 2. Modelle, Prompts und API-Konfiguration

| Zweck | Konfiguration und Ort |
| --- | --- |
| Embeddings | `MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"` in `copilot/index.py`; benutzt durch `get_embedding_model()` und `embed_texts()`. |
| Extraktion und Bewertung | Beide verwenden `MODEL = "claude-sonnet-5"` in `copilot/llm.py`. Das ist der hinterlegte Modellname; seine Verfügbarkeit wurde nicht durch einen API-Aufruf geprüft. |
| Ausgabelimit | `MAX_TOKENS = 16000` in `copilot/llm.py`, weitergereicht als `max_tokens`. Das begrenzt die Antwort, nicht den gesamten Eingabeprompt. |
| Weitere Parameter | Keine explizite Temperatur, kein Tokenzähler, kein Eingabelimit und keine technische Begrenzung auf unter 100.000 Tokens pro Anfrage. |
| Secrets | Nur `ANTHROPIC_API_KEY` wird bisher verwendet. `ACCESS_CODE` und automatisches Laden einer `.env` sind noch nicht implementiert. |

Die Prompts liegen direkt als lokale Strings in `copilot/llm.py`; es gibt keine
separaten Prompt-Dateien.

**Stufe 1 – `extract_requirements()`:** Der System-Prompt beginnt mit
„Du bist ein praeziser Assistent fuer Bewerbungsanalysen.“ Er verlangt konkrete,
gegen ein Bewerberprofil prüfbare Anforderungen, Aufgaben oder Erwartungen und
ausschließlich JSON. Das Schema je Listeneintrag lautet:

```json
{
  "requirement": "kurzer, konkreter Anforderungstext",
  "category": "muss | kann | aufgabe | soft_skill | sonstiges",
  "reason": "kurze Begruendung, warum dies eine relevante Anforderung ist"
}
```

Der User-Prompt besteht aus `Stellenanzeige:` und dem vollständigen Eingabetext.

**Stufe 2 – `assess_requirements()`:** Der System-Prompt beginnt mit
„Du bist ein ehrlicher Bewerbungs-Copilot.“ Er erlaubt ausschließlich die
mitgelieferten Profil-Chunks als Belege, verbietet erfundene Erfahrungen und
verlangt eine klare Lücke bei fehlenden oder zu schwachen Nachweisen. Er gibt
diese vier Hauptteile und Felder vor:

| Hauptteil | Felder je Eintrag |
| --- | --- |
| `anforderungs_matrix` | `requirement`, `status` (`erfuellt`, `teilweise`, `luecke`), `beleg` mit `filename`, `text`, `score`, außerdem `begruendung` |
| `framing_vorschlaege` | `luecke`, `ehrliches_framing` |
| `hervorzuhebende_projekte` | `projekt`, `warum_relevant`, `beleg_filename` |
| `wahrscheinliche_interviewfragen` | `frage`, `grund` |

Der User-Prompt ist ein mit `json.dumps(..., ensure_ascii=False, indent=2)`
serialisiertes Objekt aus `requirements_with_context` und dem Hinweis, nur
diese Chunks zu nutzen. Pro Anforderung werden Text, Kategorie/Begründung und
Retrieval-Treffer mit Dateiname, Text und Score mitgeschickt. Der vollständige
Anzeigentext wird in Stufe 2 nicht erneut gesendet. Wiederkehrende Chunks werden
pro Anforderung erneut übertragen; eine Deduplizierung fehlt.

## 3. Aktueller Startweg

Aus dem Repository-Verzeichnis, mit Python 3.11+:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mit der vorhandenen virtuellen Umgebung, ohne Aktivierung:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Voraussetzungen sind lesbare Profil-PDFs in `profile/` relativ zum
Arbeitsverzeichnis und ein in der Prozessumgebung gesetzter
`ANTHROPIC_API_KEY`. Das Embedding-Modell muss lokal verfügbar sein oder beim
ersten Laden heruntergeladen werden können. Die Oberfläche nimmt ausschließlich
den eingefügten Anzeigentext entgegen.

Alternativ über die Kommandozeile:

```powershell
python -m copilot.cli "pfad/zur/anzeige.txt" --top-k 5
python -m copilot.cli "pfad/zur/anzeige.pdf" --top-k 5
```

`python app.py` aus der neuen `AGENTS.md` ist der **Zielstart für Gradio**.
Es startet derzeit keinen regulären Streamlit-Webserver; bis zur Umstellung
bleibt `python -m streamlit run app.py` der passende Startweg.

`requirements.txt` enthält unversioniert `pypdf`, `sentence-transformers`,
`anthropic` und `streamlit`. `numpy` wird direkt importiert, ist aber nicht
explizit aufgeführt. `gradio` und `pytest` fehlen. Die auskommentierten Einträge
für `fastapi` und `uvicorn` sind keine installierten Projektanforderungen.

Die Diagnoseprogramme lassen sich mit `python inspect_extraction.py`
und `python inspect_retrieval.py` starten, geben allerdings Dokumentinhalte aus.
Sie wurden für diese Bestandsaufnahme nicht ausgeführt und entsprechen in ihrer
gegenwärtigen Form nicht der neuen Regel für inhaltsfreie Logs. Auch der neue
Testbefehl `python -m pytest -q` setzt erst noch Tests und die Abhängigkeit voraus.

## 4. Erforderliche Änderungen für Gradio auf Hugging Face Spaces

Noch **nicht umgesetzt**. Annahme: Jede Analyse erhält genau zwei PDFs, ein
Anforderungsdokument und ein Profil. „Keine Speicherung“ bedeutet gemäß neuer
`AGENTS.md`: keine dauerhafte Speicherung von Uploads, Texten, Embeddings oder
Ergebnissen; technisch anfallende Upload-Dateien werden unmittelbar nach dem
Einlesen gelöscht. Ein gemeinsamer Cache der Modellgewichte enthält keine
Nutzerdokumente und kann bestehen bleiben.

### Zielablauf und Modi

```text
Modus + Zugangscode + Anforderungs-PDF + Profil-PDF
  → serverseitige Zugangs- und Eingabeprüfung
  → beide PDFs extrahieren; temporäre Upload-Dateien unmittelbar bereinigen
  → Profiltext chunken und eigenen Index für diese Anfrage aufbauen
  → Stufe 1 mit Modus → Retrieval in diesem Index → Stufe 2 mit Modus
  → deutsche Ergebnisansicht; anfragebezogene Daten anschließend freigeben
```

| Modus | Anforderungsdokument | Profil | Benötigte Semantik |
| --- | --- | --- | --- |
| `job` | Stellenanzeige | Lebenslauf/Bewerberprofil | Bisherige Bewerbungsanalyse, Belege und Interviewfragen. |
| `tender` | Ausschreibung | Firmenprofil | Anforderungen an Leistungen und Eignung gegen belegte Fähigkeiten, Referenzen und gegebenenfalls Zertifizierungen prüfen. Fehlende Nachweise als Lücken kennzeichnen. |

Für `tender` müssen beide Prompts angepasst werden; eine andere UI-Beschriftung
allein reicht nicht. Die belegbasierte Matrix kann gemeinsam bleiben.
Als einfache Annahme für die zusätzlichen Ergebnisbereiche: ehrliche Aussagen
zu Nachweislücken, passende Firmenreferenzen und Klärungsfragen zur Ausschreibung
statt Bewerbungsformulierungen, Studienmodulen und Interviewfragen. Die konkreten
Feldnamen sind bei der Umsetzung konsistent zwischen Schema, Parser und UI zu
definieren. Keine zusätzlichen Analyse-Stufen sind dafür erforderlich.

### Änderungen nach Baustein

| Baustein | Erforderliche Änderung / Wiederverwendung |
| --- | --- |
| `app.py` | Streamlit durch Gradio ersetzen; zwei PDF-Eingaben, Moduswahl mit internen Werten `job`/`tender`, verdecktes Zugangscodefeld und deutsche Ergebnisdarstellung. Start über `python app.py` mit Gradio-Launch. |
| `copilot/ingest.py` | Einen Zugang für einzelne hochgeladene PDFs bzw. Bytes/Dateiobjekte schaffen. `pypdf`, Textvalidierung und bestehendes Chunking weiterverwenden. `load_profile()`/`build_chunks()` mit Ordnerpflicht sollen nicht im Demo-Pfad liegen. Beide PDFs auf Format, Größe und extrahierbaren Text prüfen; keine stille Analyse eines leeren Profils. |
| `copilot/index.py` | Chunks aus dem einen Profil mit `embed_texts()` und `ProfileIndex(chunks, embeddings)` indexieren. `SentenceTransformer(..., device="cpu")` explizit verwenden. Nur Modellgewichte global cachen, keine Nutzerprofile. |
| `copilot/pipeline.py` | Profil/Index und Modus explizit entgegennehmen. `index.retrieve()` auf dem Index der aktuellen Anfrage verwenden; den globalen `_DEFAULT_INDEX` im Web-Pfad vermeiden. Sonst könnten Nutzer das falsche Profil verwenden und alte Daten im Prozess verbleiben. |
| `copilot/llm.py` | Modusabhängige System-Prompts für beide Stufen, gemeinsame JSON-Verarbeitung und belegbasierte Bewertung weiterverwenden. JSON-Felder, erlaubte Statuswerte und tatsächliche Herkunft der Belege validieren. Dokumenttext als Daten behandeln, nicht als Anweisung zur Änderung der Bewertungsregeln. |
| Zugang | `ACCESS_CODE` wie den API-Key ausschließlich aus der Umgebung lesen. Serverseitig vor Analyse/API-Aufruf prüfen, auch bei direktem Endpoint-Aufruf. Vorschlag: bei fehlender Konfiguration Analyse sperren, statt unbeabsichtigt öffentlichen Zugriff zu erlauben. |
| Abhängigkeiten | Für die Ziel-App Gradio statt Streamlit sowie `numpy` und `pytest` explizit aufführen; kompatible Versionen für reproduzierbaren Betrieb festlegen. Keine zusätzliche Vektor-Datenbank und kein separates Webframework nötig. |
| Dokumentation | README auf Uploads, Modi, Umgebungsvariablen, neuen Start und Spaces umstellen; den widersprechenden lokalen Zielstand in `PLAN.md` bei der Umsetzung aktualisieren bzw. als historisch kennzeichnen. |

### Keine Speicherung und keine Inhaltslogs

Gradio legt Uploads im temporären Cache ab. Eine Dateieingabe oder ein
Bytes-Rückgabewert ist deshalb allein keine Garantie für rein speicherinterne
Verarbeitung. Nach dem Einlesen müssen die konkret zur Anfrage gehörenden
temporären Dateien auch bei Fehlern sofort gelöscht werden, etwa über
`try/finally`; offene Dateihandles sind davor zu schließen. Auch Uploads ohne
anschließende Analyse, abgebrochene Anfragen und frameworkseitige Kopien brauchen
einen geprüften Bereinigungspfad. Eine bloß periodische Bereinigung genügt der
neuen Sofort-Löschregel nicht. Das genaue Verhalten ist gegen die gewählte
Gradio-Version zu testen. Siehe [Gradio: File Access](https://gradio.app/guides/file-access).

Keine Profile oder Ergebnisse in globalen Variablen, langlebigem Session-State,
Beispiel-Caches, Dateien oder einer Datenbank halten. Nur die für die laufende
Anzeige notwendigen Daten verwenden und bei Ende/Zurücksetzen freigeben.

Aktuelle Stellen mit Änderungsbedarf sind die Textausgaben beider
`inspect_*.py`-Skripte, Rohantwort-Ausschnitte in `parse_assessment()` und
`_parse_json_response()` sowie durchgereichte Exception-Texte in PDF-/API-Fehlern
und UI. Für Betriebslogs nur Zähler und Fehlertypen verwenden, keine Dokumenttexte,
Dateinamen, vollständigen Prompts, API-Antworten oder Secrets.

Die Anwendung sendet weiterhin das Anforderungsdokument als extrahierten Text
und ausgewählte Profilauszüge an die Claude API. „Keine Speicherung“ in der Demo
ist keine Zusicherung über die Datenverarbeitung oder Aufbewahrung beim externen
API-Anbieter; die Oberfläche muss die Übermittlung verständlich benennen.

### Token-Grenze unter 100.000 pro Anfrage

`MAX_TOKENS = 16000` reicht nicht aus, um diese neue Regel einzuhalten. Vor
jedem Generierungsaufruf ist der vollständig zusammengesetzte Prompt inklusive
Systemtext, JSON und aller Profil-Chunks zu prüfen. Als konservative Auslegung
sollten Eingabetokens plus reservierte Ausgabetokens mit Sicherheitsabstand
unter 100.000 bleiben. Auch Wiederholungen müssen dieselbe Grenze einhalten.

Für die Tokenprüfung bietet Anthropic eine Zähl-API; die Dokumentation bezeichnet
den Zählwert als Schätzung, daher ist zusätzlicher Abstand sinnvoll. Vor einer
externen Zählanfrage sollten lokale Größenlimits bereits übergroße Eingaben
ablehnen. Zu große Anforderungen oder Kontexte verständlich ablehnen oder
kontrolliert verkleinern, statt sie ungeprüft zu senden. Wiederholte Chunks und
Top-k sind besonders für Stufe 2 relevant. Siehe
[Anthropic: Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting).

### Spaces-Konfiguration

Im YAML-Kopf der Space-`README.md` sind für die spätere Bereitstellung Gradio als
SDK, eine geprüfte SDK-Version, Python 3.11 und `app.py` als Einstiegsdatei
festzulegen, mit den Feldern `sdk: gradio`, `sdk_version`,
`python_version: "3.11"` und `app_file: app.py`. Die Konfiguration ist im
Repository noch nicht vorhanden. Siehe
[Hugging Face: Spaces Configuration Reference](https://huggingface.co/docs/hub/en/spaces-config-reference).

`ANTHROPIC_API_KEY` und `ACCESS_CODE` als Space-Secrets hinterlegen und nur als
Umgebungsvariablen lesen. Keine echten Werte in `.env.example`, README oder Code
eintragen. Siehe [Hugging Face: Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview).
Keinen persistenten Speicher für Nutzerdokumente einrichten. CPU-Modellstart und
begrenzte parallele Analysen bei der Umsetzung prüfen.

### Gezielte Prüfungen für die spätere Umsetzung

- Zwei synthetische PDFs für jeden Modus: richtige Anforderungsextraktion,
  erwartete Profil-Chunks, belegte Bewertung und passende deutsche Darstellung.
- Zwei aufeinanderfolgende bzw. parallele Profile: keine Vermischung der Indizes
  und keine Wiederverwendung vorheriger Nutzerdaten.
- Leere, beschädigte, zu große oder textlose PDFs: verständlicher Abbruch und
  sofortige Bereinigung, auch bei API-Fehlern und abgebrochenem Upload.
- Zugangscode fehlt/ist falsch: kein Analyseaufruf, auch über einen API-Endpunkt.
- Token-Grenzen vor beiden Stufen und Retries; synthetische übergroße Kontexte
  werden vor dem Generierungsaufruf abgewiesen.
- Ungültiges JSON, falsche Statuswerte und nicht gelieferte Belege: kontrollierte
  Fehler; keine Antwort- oder Dokumentauszüge in Logs.

Diese Tests sind Vorschläge für die Umsetzung, keine bereits ausgeführten
Prüfungen. Externe API-Aufrufe sollten in Unit-Tests durch Test-Doubles ersetzt
werden. Zielbefehl gemäß neuer `AGENTS.md`: `python -m pytest -q`.

## 5. Begleitdateien dieser Aufgabe

- `AGENTS.md` wurde vollständig durch den vorgegebenen neuen Inhalt ersetzt.
- `.env.example` enthält ausschließlich die leeren Platzhalter
  `ANTHROPIC_API_KEY=` und `ACCESS_CODE=`. Die Datei lädt selbst keine Variablen.
- `.env` steht bereits explizit in `.gitignore` (Zeile 5); die Ignore-Datei
  musste nicht geändert werden. `.env` ist nicht als versionierte Datei gelistet.
- Hinweis zum Git-Stand: Auch `AGENTS.md` und `PLAN.md` sind bisher ignoriert;
  die neue `AGENTS.md` ist daher lokal ersetzt, wird aber nicht automatisch
  versioniert. Diese bestehende Ignore-Regel wurde nicht verändert.
- Anwendungscode, Abhängigkeiten, README und alter PLAN bleiben unverändert.
