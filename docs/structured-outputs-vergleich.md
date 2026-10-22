# Structured Outputs: Diagnose und Vergleich

Messung am 20.10.2026. Ausgangsstand: Commit `b44a45e`.

## Durchführung

Je fünf Aufrufe von `copilot.cli.main()` in einem frischen Python-Prozess der
lokalen venv, mit `examples/job_anforderungen.pdf`,
`examples/job_profil.pdf` und `--mode job`. Modell in beiden Stufen:
`claude-haiku-5-5`, `effort="low"`, Thinking deaktiviert, Timeout 60 Sekunden,
SDK `max_retries=1`. Installiertes Anthropic-SDK: `0.116.0`.

Die Dauer umfasst Imports, Laden der lokal vorhandenen Embedding-Gewichte auf
der CPU und die vollständige CLI-Analyse. Downloads waren deaktiviert.
Die CLI-Ergebnisausgabe wurde im Arbeitsspeicher verworfen. Es wurden keine
Dokumenttexte, Prompts, API-Antworten oder Secrets protokolliert oder gespeichert.
Validierungsdiagnosen enthalten ausschließlich Fehlertyp, bereinigten Feldpfad,
Stufe und Versuchszähler.

Ein erster, durch die Sandbox blockierter Verbindungsversuch zählt nicht zur
Messreihe. Die zehn Vergleichsläufe erfolgten anschließend mit Netzwerkfreigabe.
Korrekturversuche bezeichnen zusätzliche Modellantworten nach Antwortfehlern,
nicht die SDK-internen Wiederholungen bei Transportfehlern.

## Ergebnisse

| Lauf | Vorher: Dauer | Vorher: Abbruch | Vorher: Korrekturen | Nachher: Dauer | Nachher: Abbruch | Nachher: Korrekturen |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 26,56 s | ja | 1 | 20,02 s | nein | 0 |
| 2 | 25,13 s | ja | 1 | 18,27 s | nein | 0 |
| 3 | 24,92 s | nein | 1 | 19,60 s | nein | 0 |
| 4 | 25,69 s | ja | 1 | 17,96 s | nein | 0 |
| 5 | 18,40 s | nein | 0 | 18,36 s | nein | 0 |
| Mittelwert / Summe | **24,14 s** | **3/5** | **4** | **18,84 s** | **0/5** | **0** |

Vorher trat ausschließlich `json_invalid` am Pfad `$` (JSON-Wurzel) auf:

| Lauf | Stufe | Versuch | Fehlertyp | Feldpfad |
| --- | --- | --- | --- | --- |
| 1 | Bewertung | 1 und 2 | `json_invalid` | `$` |
| 2 | Bewertung | 1 und 2 | `json_invalid` | `$` |
| 3 | Bewertung | 1 | `json_invalid` | `$` |
| 4 | Bewertung | 1 und 2 | `json_invalid` | `$` |

Das sind sieben Validierungsfehler in vier Analysen. Drei Analysen brachen auch
nach dem Korrekturversuch mit `LlmResponseError` ab. In Stufe 1 gab es keine
Validierungsfehler. Die Diagnose belegt ungültiges JSON; welche konkreten
Syntaxstellen betroffen waren, wurde bewusst nicht inhaltlich protokolliert.

Nachher waren alle fünf Analysen erfolgreich, jeweils mit 15 Anforderungen,
ohne Validierungsfehler, ohne Korrekturversuch und ohne Reparatur einzelner
Bewertungen. Die API-Zeiten und Tokenzahlen nachher:

| Lauf | Extraktion | Bewertung | Eingabetokens gesamt | Ausgabetokens gesamt |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4,19 s | 7,11 s | 7192 | 2229 |
| 2 | 3,01 s | 6,73 s | 7192 | 2316 |
| 3 | 4,18 s | 6,79 s | 7192 | 2273 |
| 4 | 3,20 s | 6,18 s | 7209 | 2214 |
| 5 | 2,74 s | 7,21 s | 7192 | 2430 |

Lauf 1 nachher ist der erste Aufruf mit den neuen Schemas und enthält möglichen
Kompilierungsaufwand. Dieser Anteil wurde nicht separat gemessen. Fünf Läufe
sind eine kleine Funktionsprüfung, keine allgemeine Zuverlässigkeitsgarantie.

## Umsetzung und Prüfung

Beide Stufen senden `output_config.format` mit einem aus den bestehenden
Pydantic-Modellen abgeleiteten JSON-Schema. `anthropic.transform_schema()`
verschiebt nicht unterstützte Beschränkungen wie `minimum` und `minLength`
in Beschreibungen und setzt `additionalProperties: false`. Die ursprünglichen
Pydantic-Prüfungen bleiben lokal aktiv. Das Schema wird beim Tokenbudget
einschließlich Antwortreserve berücksichtigt.

In Stufe 2 wird zuerst JSON und Fazit geprüft. Einzelne Bewertungen werden
lokal repariert: unbelegte Einträge verlieren ihren Beleg, ungültige oder
fehlende Belege und unbekannte Statuswerte führen zu `nicht_belegt`.
Fehlende/unbrauchbare Begründungen werden konservativ als `nicht_belegt`
mit „Nicht bewertet” behandelt. Unbekannte IDs entfallen, bei doppelten IDs
bleibt die erste Bewertung, fehlende IDs werden ergänzt. Erst danach wird
die gesamte Bewertungsantwort streng validiert. Zitatprüfung, Score,
Verbrauchserfassung und Ergebnisformat bleiben erhalten.

`python -m pytest -q`: **193 bestanden**, ohne API-Aufrufe. Abgedeckt sind
unter anderem ungültige Belege, unbekannte Statuswerte, malformed Einträge,
fehlende/doppelte IDs, Erhalt gültiger Bewertungen, fehlendes Fazit,
bereinigte Fehlermeldungen, die beiden API-Schemas, Tokenbudget samt Schema
und die vollständige CLI-Ausgabe.

## Dokumentation

Haiku 5.5 unterstützt Structured Outputs. API-Einschränkungen und
Schema-Transformation:
[Anthropic: Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

`effort="low"` und `thinking={"type": "disabled"}` sind kompatibel:
[Anthropic: Prompting Claude Haiku 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5).
Die fünf erfolgreichen API-Läufe bestätigen die Kombination mit beiden
Schemas für diese Beispieldateien.
