# Modusregeln, Extraktionsstabilität und Analysekosten

Stand: 22.10.2026. Gemessen wurden die vorhandenen fiktiven Dateien
`examples/job_anforderungen.pdf` und `examples/job_profil.pdf` im Modus `job`.

## Ergebnis

Die endgültigen fünf reinen Extraktionsläufe liefern **je 13 Anforderungen**,
identisch in Wortlaut, Muss-/Kann-Typ und Reihenfolge. Die sieben Muss- und sechs
Kann-Kriterien umfassen ausschließlich Anforderungen an die Person. Stellenaufgaben
sind in allen fünf Läufen ausgeschlossen. Die Oder-Gruppe bleibt eine Anforderung;
Git/Docker, Deutsch/Englisch und TypeScript/React werden getrennt.

Die drei vollständigen Analysen verwenden jeweils dieselbe extrahierte Liste und
liefern **Score 68, Muss erfüllt 6/7**. Die kleine Stichprobe belegt dieses Verhalten
für die Beispieldateien; sie ist keine Garantie für beliebige Dokumente.

## Implementierung

- `job`: Nur Qualifikationen, Kenntnisse, Erfahrungen, Abschlüsse und Sprachen.
  Stellenaufgaben wie „Deine Aufgaben” oder „Du entwickelst …” sind ausgeschlossen;
  daraus werden keine zusätzlichen fachlichen Anforderungen abgeleitet.
- `tender`: Eignungskriterien (Referenzen, Zertifikate, Umsatz, Personal) und fachliche
  Leistungsanforderungen. Abgabefristen und Angebotsformalien sind ausgeschlossen.
  Geforderte Liefer-/Leistungsfristen, Leistungsorte und Servicezeiten bleiben relevant.
- Jeder strukturelle Listenpunkt bzw. eigenständige Anforderungsabsatz im relevanten
  Abschnitt wird in Dokumentreihenfolge vollständig berücksichtigt. Verschiedene
  Punkte werden nicht zusammengefasst. Innerhalb eines Punkts gelten die bisherigen
  Und-/Oder- und Niveauregeln; ein Komma erzeugt keinen neuen strukturellen Listenpunkt.
- Separate Standards in `copilot/llm.py`: Extraktion `claude-haiku-5-5`, Bewertung
  `claude-sonnet-5-5`. `MODEL_EXTRACT` und `MODEL_ASSESS` überschreiben diese weiterhin.
- `copilot/pricing.py` enthält die USD-Tarife, Quelle, Prüfdatum und eine Kostenfunktion
  mit `Decimal`. Unbekannte Modelle erhalten keinen stillschweigenden Ersatzpreis.
  Normale Eingaben, Ausgaben sowie Cache-Reads und 5m-/1h-Writes werden getrennt bewertet.
  Die Verbrauchszähler erkennen dafür die 1h-Unterkategorie der SDK-Antwort.

## Preise und Berechnung

Standard-API-Listenpreise in **USD je Million Tokens**, geprüft am 22.10.2026:

| Modell | Eingabe | Ausgabe | Cache schreiben 5m | Cache schreiben 1h | Cache lesen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Haiku 5.5, bis 100.000 Eingabetokens je Anfrage | 0,10 | 0,50 | 0,125 | 0,20 | 0,01 |
| Sonnet 5.5 | 2,00 | 10,00 | 2,50 | 4,00 | 0,10 |

Quelle: [Anthropic: Pricing](https://platform.claude.com/docs/en/about-claude/pricing).
Die Anwendung begrenzt jede Anfrage einschließlich Antwortreserve auf unter 100.000
Tokens; der höhere Haiku-Tarif für längere Einzelanfragen kommt hier nicht zum Einsatz.

Kosten je Stufe = (normale Eingabetokens × Eingabepreis + Ausgabetokens × Ausgabepreis
+ Cache-Tokens × jeweiliger Cache-Preis) / 1.000.000. Die Analyse-Kosten sind die Summe
beider Stufen mit ihrem jeweils tatsächlich verwendeten Modell. Die bereits summierten
Tokens von Antwortkorrekturen zählen mit. Preisentscheidend ist die einzelne Anfrage,
nicht die Token-Summe mehrerer Wiederholungen.

Das öffentliche Ergebnis summiert alle Eingabetokens; für die Kosten verwendet die
Funktion dagegen die getrennten Kategorien von `TokenUsage.to_dict()` je Stufe.
Dies vermeidet eine Doppelzählung gecachter Eingaben. Cache-Writes ohne explizite
1h-Unterkategorie werden zum Standard-5m-Tarif gerechnet. In den gemessenen Läufen
waren alle Cache-Zähler null. Angegeben sind Listenpreiskosten ohne Steuern oder
individuelle Rabatte, keine Abrechnung aus dem Anthropic-Konto.

## Fünf reine Extraktionsläufe

Fünf unabhängige API-Anfragen, ohne Bewertung oder Wiederverwendung einer früheren
Extraktionsantwort. Die PDF wurde mit der vorhandenen Textfunktion im Speicher gelesen.
Verglichen werden die Paare `(text, typ)` nach Vereinheitlichung von Leerraum.
Wortwahl und Groß-/Kleinschreibung werden nicht normalisiert. Die Reihenfolge wird
zusätzlich verglichen; IDs entstehen danach im Code.

| Lauf | Anzahl | Wortlaut + Typ wie E1 | Reihenfolge wie E1 | Dauer | Tokens ein/aus | Kosten USD |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| E1 | 13 | ja | ja | 2,57 s | 3.449 / 467 | 0,00057840 |
| E2 | 13 | ja | ja | 2,95 s | 3.449 / 467 | 0,00057840 |
| E3 | 13 | ja | ja | 2,08 s | 3.449 / 467 | 0,00057840 |
| E4 | 13 | ja | ja | 2,15 s | 3.449 / 467 | 0,00057840 |
| E5 | 13 | ja | ja | 2,16 s | 3.449 / 467 | 0,00057840 |

Alle fünf Läufe besitzen genau diese Liste in dieser Reihenfolge:

| ID | Anforderung (Wortlaut der Extraktion) | Typ | E1 | E2 | E3 | E4 | E5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Abgeschlossenes Studium der Informatik oder eines vergleichbaren Fachs | muss | gleich | gleich | gleich | gleich | gleich |
| R2 | Sehr gute Kenntnisse in Python | muss | gleich | gleich | gleich | gleich | gleich |
| R3 | Praktische Erfahrung mit Embeddings, semantischer Suche oder RAG-Systemen | muss | gleich | gleich | gleich | gleich | gleich |
| R4 | Sicherer Umgang mit Git | muss | gleich | gleich | gleich | gleich | gleich |
| R5 | Sicherer Umgang mit Docker | muss | gleich | gleich | gleich | gleich | gleich |
| R6 | Sehr gute Deutschkenntnisse | muss | gleich | gleich | gleich | gleich | gleich |
| R7 | Gute Englischkenntnisse | muss | gleich | gleich | gleich | gleich | gleich |
| R8 | Erfahrung mit Kubernetes | kann | gleich | gleich | gleich | gleich | gleich |
| R9 | Kenntnisse in Microsoft Azure oder Google Cloud Platform | kann | gleich | gleich | gleich | gleich | gleich |
| R10 | Erfahrung mit Frameworks wie LangChain oder LangGraph | kann | gleich | gleich | gleich | gleich | gleich |
| R11 | Grundkenntnisse in TypeScript | kann | gleich | gleich | gleich | gleich | gleich |
| R12 | Grundkenntnisse in React | kann | gleich | gleich | gleich | gleich | gleich |
| R13 | Erfahrung mit MLOps-Werkzeugen wie MLflow | kann | gleich | gleich | gleich | gleich | gleich |

Die Messung verwendet ausschließlich den Stellenanzeigenmodus. Die neuen Ausschreibungsregeln
sind implementiert; diese Messreihe enthält keine Ausschreibungs-PDF.

## Drei vollständige Analysen

Haiku-Extraktion plus Sonnet-Bewertung, ohne gesetzte Modell-Overrides im Messprozess.
Das CPU-Embedding-Modell wurde einmal vor diesen drei Läufen geladen; der einmalige
Modellstart ist in der Dauer nicht enthalten. Jede Analyse baut ihren eigenen Profilindex.
Die Zeit umfasst PDF-Lesen, Embeddings, Retrieval und beide API-Aufrufe.

| Lauf | Kriterien | Score | Muss erfüllt | Extraktion wie E1 | Dauer | Kosten Extraktion USD | Kosten Bewertung USD | Gesamt USD |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| A1 | 13 | 68 | 6/7 | ja | 10,11 s | 0,00057840 | 0,02393000 | **0,02450840** |
| A2 | 13 | 68 | 6/7 | ja | 11,43 s | 0,00057840 | 0,02450000 | **0,02507840** |
| A3 | 13 | 68 | 6/7 | ja | 10,62 s | 0,00057840 | 0,02466000 | **0,02523840** |

| Lauf | Extraktion API | Bewertung API | Extraktion Tokens ein/aus | Bewertung Tokens ein/aus | Gesamt Tokens ein/aus |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 2,04 s | 7,83 s | 3.449 / 467 | 5.425 / 1.308 | 8.874 / 1.775 |
| A2 | 3,08 s | 8,17 s | 3.449 / 467 | 5.425 / 1.365 | 8.874 / 1.832 |
| A3 | 2,16 s | 8,25 s | 3.449 / 467 | 5.425 / 1.381 | 8.874 / 1.848 |

Alle acht endgültigen Testläufe waren erfolgreich: fünf reine Extraktionsaufrufe und
sechs API-Aufrufe für drei vollständige Analysen. Es gab keine Antwortkorrekturversuche
oder API-Fehler. Listenpreiskosten der endgültigen Messreihe insgesamt: **0,07771720 USD**.

## Rückrechnung der vorherigen sieben Analysen

Grundlage sind die Tokenwerte im [vorherigen Vergleich](konsistenz-vergleich.md),
nicht erneute API-Aufrufe. H1–H5 verwendeten Haiku für beide Stufen; S1/S2 verwendeten
Haiku für die Extraktion und Sonnet für die Bewertung. Die Eingabe-/Ausgabetokens der
Extraktion ergeben sich aus Gesamttokens minus Bewertungstokens. Cache-Tokens waren
laut damaliger Messung null. Berechnet mit den oben hinterlegten aktuellen Listenpreisen:

| Vorheriger Lauf | Modelle Extraktion / Bewertung | Extraktion USD | Bewertung USD | Gesamt USD |
| --- | --- | ---: | ---: | ---: |
| H1 | Haiku / Haiku | 0,00049490 | 0,00127290 | **0,00176780** |
| H2 | Haiku / Haiku | 0,00049490 | 0,00123640 | **0,00173130** |
| H3 | Haiku / Haiku | 0,00063190 | 0,00158980 | **0,00222170** |
| H4 | Haiku / Haiku | 0,00049490 | 0,00132490 | **0,00181980** |
| H5 | Haiku / Haiku | 0,00049490 | 0,00127790 | **0,00177280** |
| S1 | Haiku / Sonnet | 0,00063190 | 0,02963600 | **0,03026790** |
| S2 | Haiku / Sonnet | 0,00049490 | 0,02428800 | **0,02478290** |

## Structured Outputs, Effort und Thinking

Sonnet 5.5 unterstützt [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
Beide Stufen verwenden weiterhin das aus Pydantic abgeleitete JSON-Schema und `effort="low"`.
Haiku erhält `thinking={"type":"disabled"}`. Sonnet 5.5 weist diesen Wert zurück;
stattdessen verwendet der Code `thinking={"type":"between_tools"}`. Ohne angebotene
Werkzeuge bedeutet dies keine Thinking-Phase vor der Antwort. Dies ist laut
[Anthropic: Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)
bei `low` erlaubt. Alle drei tatsächlichen Sonnet-Aufrufe bestätigen die Kombination
mit unserem Schema. Timeout 60 Sekunden, SDK `max_retries=1`, Zitatprüfung, Score im
Code und Tokenlimit einschließlich Schema und Antwortreserve bleiben erhalten.

## Vorläufige Messung und Präzisierung

Vor dem endgültigen Stand gab es eine erste vollständige Messreihe (ebenfalls fünf
Extraktionsläufe und drei Analysen). Die neue Anweisung „jeder Punkt mindestens einmal”
führte noch zu einer uneinheitlichen Zerlegung der Oder-Gruppe. Zwölf andere
Anforderungen waren bereits identisch. Daraufhin wurde allgemein präzisiert, dass
strukturelle Listenpunkte im Original gemeint sind und eine Oder-Gruppe innerhalb
eines Punkts diesen vollständig abdeckt. Es wurden keine Technologien oder erwarteten
Bewertungsergebnisse aus den Testdateien in den Prompt aufgenommen.

| Vorläufiger Extraktionslauf | Anzahl | Wortlaut + Typ wie erster Lauf |
| --- | ---: | --- |
| E1 | 15 | ja |
| E2 | 15 | ja |
| E3 | 14 | nein |
| E4 | 13 | nein |
| E5 | 15 | ja |

| Vorläufige Analyse | Score | Muss erfüllt | Dauer | Kosten USD |
| --- | ---: | --- | ---: | ---: |
| A1 | 65 | 7/9 | 11,26 s | 0,02702550 |
| A2 | 65 | 7/9 | 10,84 s | 0,02710550 |
| A3 | 68 | 6/7 | 11,74 s | 0,02515900 |

Diese vorläufigen Werte gehören zum älteren Promptstand und werden nicht mit der
endgültigen Messreihe vermischt. Nach der Präzisierung wurde der vollständige Block
erneut ausgeführt. Ein zuvor von der Sandbox blockierter Verbindungsversuch ist
keine Modellmessung und nicht in diesen Messreihen enthalten.

## Prüfungen und Datenverarbeitung

`python -m pytest -q`: **235 bestanden**, ohne API-Aufrufe. Die Tests prüfen unter
anderem die getrennten Modell-Standards und Overrides, Sonnet-Konfiguration, exakte
Kosten mit beiden Cache-Schreibdauern, Korrekturantworten, unbekannte Modelle und
ungültige Tokenzähler. Das vollständige Ergebnisformat bleibt kompatibel.

Die temporäre Messhilfe wurde entfernt. Technische Ausgaben enthalten ausschließlich
Zeiten, Zähler, Modellnamen und Fehlertypen. Dieser Bericht enthält nur die ausdrücklich
angeforderten Anforderungsbezeichnungen und Messwerte; Profilzitate, vollständige
Dokumenttexte, Modellbegründungen und Secrets wurden nicht protokolliert oder gespeichert.
Die Beispiele wurden nur als Bytes gelesen, alle Profilindizes waren anfragebezogen im Speicher.
