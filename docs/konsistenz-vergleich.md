# Konsistenz der Bewertung: Haiku und Sonnet

Stand und Messdatum: 22.10.2026. Verwendet wurden ausschließlich die vorhandenen, fiktiven
`examples/job_anforderungen.pdf` und `examples/job_profil.pdf` im Modus `job`.

## Änderungen

Die Extraktion unterscheidet zuerst Alternativgruppen und trennt anschließend unabhängig
prüfbare Und-Fähigkeiten. Eine Aufzählung „A, B oder C” bleibt eine Anforderung.
Jedes getrennte Kriterium behält Niveau und Muss-/Kann-Typ. Die Promptbeispiele verwenden
abstrakte Werkzeuge, Fähigkeiten und Sprachen X/Y, keine Technologien oder erwarteten
Statuswerte aus den Beispieldateien.

Die Bewertung verlangt nur das tatsächlich geforderte Niveau. Eine belegte Oder-Alternative
genügt; ein erreichtes oder höheres Niveau ergibt `erfuellt`. Die sprachunabhängige
Einordnung nach Projektvorgabe lautet B1 = Grundkenntnisse, B2 = gut, C1 = sehr gut bzw.
verhandlungssicher, C2 und Muttersprache = exzellent. `teilweise` verlangt einen vorhandenen
Beleg und eine konkret benannte, nachweisliche Unterschreitung oder Teillücke.

Der Markdown-Export erhält normale Satzzeichen, Bindestriche, Klammern und Anführungszeichen.
Wirksame Überschriften-/Listenanfänge, Linksyntax und Inlineformatierungen werden gezielt
geschützt; die HTML-Absicherung bleibt aktiv.

## Messaufbau

Fünf vollständige Analysen mit `MODEL_ASSESS=claude-haiku-5-5`, danach zwei mit
`MODEL_ASSESS=claude-sonnet-5-5`. Extraktion in allen sieben Läufen mit
`claude-haiku-5-5`; alle Läufe verwenden denselben endgültigen Promptstand.
Die Umgebungsvariablen wurden nur im Messprozess gesetzt. Standardmodell bleibt Haiku.

Beide Modelle verwenden `effort="low"` und Structured Outputs. Haiku erhält
`thinking={"type":"disabled"}`. Für Sonnet 5.5 wird die kompatible niedrigste Einstellung
`thinking={"type":"between_tools"}` verwendet; es werden keine Werkzeuge angeboten.
Timeout 60 Sekunden, SDK `max_retries=1`, eigener Antwortkorrekturversuch und Tokenbudget
unter 100.000 einschließlich Schema und Antwortreserve bleiben aktiv.

Das CPU-Embedding-Modell wurde einmal vor der Messung geladen. Jede Analyse baut ihren
eigenen Profilindex im Arbeitsspeicher. Die Gesamtdauer umfasst die vollständige Analyse
mit PDF-Lesen, Embeddings, Retrieval und beiden API-Stufen, ohne den einmaligen Modellstart.
API-Zeiten sind separat ausgewiesen. Tokenzahlen stammen aus den API-Antworten.
Technische Ausgaben enthalten nur Zähler, Zeiten und Fehlertypen. Der Bericht enthält die
ausdrücklich angeforderten kurzen Anforderungsbezeichnungen, keine Profilzitate, vollständigen
Dokumenttexte, Modellbegründungen oder Secrets. Dokumente/Indizes wurden nicht gespeichert.

## Endgültige Messreihe

| Lauf | Score | Muss erfüllt | Kriterien | Gesamtzeit | Bewertung API | Tokens gesamt ein/aus | Bewertung Tokens ein/aus |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 65 | 5/7 | 13 | 8,71 s | 5,56 s | 8.053 / 1.925 | 5.439 / 1.458 |
| H2 | 63 | 5/7 | 13 | 8,25 s | 5,50 s | 8.053 / 1.852 | 5.439 / 1.385 |
| H3 | 66 | 6/7 | 18 | 10,95 s | 7,37 s | 8.972 / 2.649 | 6.358 / 1.908 |
| H4 | 73 | 6/7 | 13 | 8,50 s | 5,64 s | 8.053 / 2.029 | 5.439 / 1.562 |
| H5 | 63 | 5/7 | 13 | 7,90 s | 5,50 s | 8.053 / 1.935 | 5.439 / 1.468 |
| S1 | 62 | 6/7 | 18 | 12,28 s | 9,12 s | 8.972 / 2.433 | 6.358 / 1.692 |
| S2 | 68 | 6/7 | 13 | 10,32 s | 8,06 s | 8.053 / 1.808 | 5.439 / 1.341 |

H = Haiku-Bewertung, S = Sonnet-Bewertung. Alle sieben Analysen waren erfolgreich.
Es gab keine Antwortkorrekturversuche, keine API-Fehler und keine Herabstufungen durch
die Zitatprüfung. Unterschiedliche Statuswerte kommen somit aus der Modellbewertung.

| Modell | Score-Spanne | Gesamtzeit Mittel | Bewertung API Mittel | Ausgabe-Tokens gesamt Mittel |
| --- | --- | ---: | ---: | ---: |
| Haiku | 63–73 | 8,86 s | 5,91 s | 2078,00 |
| Sonnet | 62–68 | 11,30 s | 8,59 s | 2120,50 |

## Status pro Anforderung

E = `erfuellt`, T = `teilweise`, N = `nicht_belegt`, — = in diesem Lauf nicht extrahiert.
Die Zuordnung erfolgt nach Bedeutung, nicht nach den wechselnden R-IDs. Unterschiede bei
Niveaus oder bei der logischen Verknüpfung würden nicht als dieselbe Anforderung gewertet.
Alle hier zusammengeführten Bezeichnungen besitzen in den endgültigen Läufen denselben
Typ, dasselbe Niveau und dieselbe Verknüpfung.

„Wechsel” zählt Statusänderungen zwischen unmittelbar aufeinanderfolgenden Läufen, nur wenn
die Anforderung in beiden vorhanden ist. Die Angabe lautet Änderungen/vergleichbare Paare
(maximal 4 bei Haiku, 1 bei Sonnet). `n. v.` bedeutet: kein vergleichbares Paar.
Fehlende Extraktionen werden nicht künstlich als `nicht_belegt` bewertet.

| Anforderung | Typ | H1 | H2 | H3 | H4 | H5 | Wechsel H | S1 | S2 | Wechsel S |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: |
| Einschlägiger Studienabschluss | muss | E | E | E | E | E | 0/4 | E | E | 0/1 |
| Sehr gute Pythonkenntnisse | muss | E | E | E | E | E | 0/4 | E | E | 0/1 |
| Embeddings, semantische Suche oder RAG | muss | E | E | E | E | E | 0/4 | E | E | 0/1 |
| Sicherer Umgang mit Git | muss | E | E | E | E | E | 0/4 | E | E | 0/1 |
| Sicherer Umgang mit Docker | muss | T | T | T | T | T | 0/4 | T | T | 0/1 |
| Sehr gute Deutschkenntnisse | muss | T | N | E | E | N | 3/4 | E | E | 0/1 |
| Gute Englischkenntnisse | muss | E | E | E | E | E | 0/4 | E | E | 0/1 |
| Kubernetes | kann | N | N | N | N | N | 0/4 | N | N | 0/1 |
| Azure oder Google Cloud Platform | kann | N | T | T | T | T | 1/4 | T | T | 0/1 |
| LangChain oder LangGraph | kann | N | N | N | N | N | 0/4 | N | N | 0/1 |
| Grundkenntnisse TypeScript | kann | E | E | E | E | E | 0/4 | N | N | 0/1 |
| Grundkenntnisse React | kann | N | N | N | N | N | 0/4 | N | N | 0/1 |
| MLOps / MLflow | kann | N | N | N | N | N | 0/4 | N | N | 0/1 |
| LLM-Anwendungen entwickeln (inkl. RAG) | kann | — | — | N | — | — | n. v. | N | — | n. v. |
| LLM-APIs über REST anbinden | kann | — | — | N | — | — | n. v. | N | — | n. v. |
| PDF-Pipelines aufbauen und pflegen | kann | — | — | E | — | — | n. v. | E | — | n. v. |
| Cloud: Deployment, Monitoring, Kostenkontrolle | kann | — | — | N | — | — | n. v. | N | — | n. v. |
| Antwortqualität evaluieren | kann | — | — | E | — | — | n. v. | E | — | n. v. |

Über alle vergleichbaren Paare: Haiku **4/52 Statuswechsel**, Sonnet **0/13**.
Bei Haiku betreffen drei Wechsel die Deutschkenntnisse, einer Azure/GCP. Für die fünf
nur in H3 vorhandenen Aufgaben ist keine Stabilitätsaussage möglich.

Die konkret beanstandete Oder-Anforderung und gute Englischkenntnisse sind im endgültigen
Stand in **allen sieben Läufen erfüllt**. Git/Docker, Deutsch/Englisch und TypeScript/React
werden in allen sieben Läufen getrennt, die Oder-Gruppe jeweils als ein Kriterium extrahiert.
Dadurch beträgt der Muss-Nenner einheitlich 7. Docker bleibt durchgehend `teilweise`;
die Zuordnung zu einer Sprache wird nicht auf andere Sprachen übertragen.

## Verbleibende Schwankung bei der Extraktion

| Aufgabenblock | H1 | H2 | H3 | H4 | H5 | S1 | S2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Die fünf Kann-Aufgaben: LLM-Anwendungen, API-Anbindung, PDF-Pipelines, Cloudbetrieb, Evaluation | fehlt | fehlt | vorhanden | fehlt | fehlt | vorhanden | fehlt |

Die Zerlegung der Kernkriterien ist konsistent, die Vollständigkeit der Extraktion noch
nicht: Der Haiku-Extraktor lässt fünf Aufgaben in vier der fünf Haiku-Läufe und einem der
zwei Sonnet-Läufe aus. Entsprechend gibt es 13 oder 18 Kriterien. Zwischen den Haiku-Läufen
kommen fünf Aufgaben hinzu und verschwinden wieder (10 Änderungen der Listenzugehörigkeit);
zwischen S1 und S2 verschwinden fünf. Das sind keine Statuswechsel derselben Anforderung.

Die Score-Spanne bildet daher sowohl Bewertungsunterschiede als auch unterschiedliche
Anforderungslisten ab. Sonnet bewertet die 13 in beiden Sonnet-Läufen vorhandenen Kriterien
gleich; sein Score verändert sich trotzdem durch die Haiku-Extraktion. Die Daten liefern
einen ersten Hinweis auf stabilere Einzelbewertungen mit Sonnet, aber wegen nur zwei Läufen
und unterschiedlicher Listen keinen belastbaren Nachweis einer stabileren Gesamtpipeline.
Die Deutschbewertung durch Haiku bleibt trotz allgemeiner Niveau-Regel unzuverlässig.

## Vorläufige Prüfrunden

Vor der endgültigen Messreihe gab es zwei vollständige Prüfrunden mit je fünf Haiku- und
zwei Sonnet-Läufen (14 zusätzliche Analysen). Sie deckten Verstöße gegen die allgemeinen
Zerlegungsregeln auf. Daraufhin wurden ausschließlich die allgemeinen Promptanweisungen
und abstrakten Beispiele präzisiert. Die endgültige Messreihe oben wurde erst danach
vollständig ausgeführt. Die früheren Resultate werden nicht mit ihr vermischt.

| Prüfrunde | Haiku: Scores | Haiku: Muss-Zähler | Sonnet: Scores | Sonnet: Muss-Zähler | Befund |
| --- | --- | --- | --- | --- | --- |
| 1 | 54, 54, 57, 67, 64 | 5/9, 5/9, 5/9, 7/8, 7/9 | 67, 63 | 6/7, 7/9 | Oder-Gruppe meist zerlegt; Und-Zerlegung uneinheitlich |
| 2 | 50, 50, 50, 64, 55 | 4/6, 3/6, 3/6, 5/6, 4/6 | 61, 61 | 5/6, 5/6 | Oder-Gruppe erhalten; Werkzeuge und Fachkenntnisse noch zusammengezogen |

Ein zuvor durch die Sandbox blockierter Verbindungsversuch ist keine Modellmessung und
nicht in den 21 vollständig ausgeführten Analysen enthalten.

## Validierung und Quellen

`python -m pytest -q`: **221 bestanden**, ohne API-Aufrufe. Zusätzlich wurde die tatsächliche
Markdown-Darstellung von normaler Interpunktion, formatiertem Text, Links, Listenanfängen
und Überschriften mit dem lokal vorhandenen Markdown-Parser geprüft. Der Sonnet-Override
ist für Modellnamen und datierte Varianten getestet; Structured Outputs und `effort="low"`
bleiben dabei erhalten. `git diff --check` meldet keine inhaltlichen Formatfehler.

Die Thinking-Einstellung von Sonnet folgt der offiziellen Dokumentation:
[Anthropic: Prompting Claude Sonnet 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5-5).
Die Sprachniveau-Zuordnung entspricht der projektweit vorgegebenen Bewertungsregel.
