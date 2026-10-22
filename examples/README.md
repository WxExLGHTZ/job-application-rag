# Fiktive Beispiele

Beide PDF-Paare und Ergebnisse sind fiktiv. Die öffentliche Ansicht startet mit
der Ausschreibung. Ohne Zugangscode zeigt sie ausschließlich gespeicherte Daten;
die Original-PDFs können zum Nachprüfen heruntergeladen werden. Mit Zugangscode
wählt „Beispiel verwenden” die beiden PDFs als Eingaben aus. Erst „Beispiel analysieren”
startet eine neue Live-Analyse, die gegen die Limits zählt.

## Erstellung am 22.10.2026

| Beispiel | Score | Muss erfüllt | Dauer | API-Kosten (USD) |
| --- | ---: | ---: | ---: | ---: |
| Stellenanzeige | 68 | 6/7 | 24,292 s | 0,0247984 |
| Ausschreibung | 50 | 4/11 | 19,906 s | 0,0299834 |

Je Beispiel ein erfolgreicher Pipeline-Aufruf: Extraktion mit Haiku 5.5,
Bewertung mit Sonnet 5.5, jeweils eine Antwort, keine Antwortkorrektur.
Der erste Netzwerkversuch für das Job-Beispiel scheiterte ohne API-Antwort.
Die Messwerte gehören zu den neu erstellten Ergebnissen; historische Messwerte
werden nicht verwendet. Kosten sind Listenpreise, aus dem tatsächlichen Verbrauch
je Modell berechnet, einschließlich gemeldeter Cache-Kategorien und Korrekturen.
Die Dauer misst `analyze` einschließlich PDF-Textextraktion, Embeddings und
gegebenenfalls Modellinitialisierung; Prozessstart und Python-Imports sind ausgenommen.

Die Begleitdateien `result_job.meta.json` und `result_tender.meta.json` enthalten
Zeitpunkt, Dauer, Kosten, Verbrauch je Modell und eine SHA-256-Prüfsumme der
zugehörigen Ergebnisdatei (UTF-8, LF-Zeilenumbrüche, auch bei Windows-Checkouts).
Bei abweichender Prüfsumme zeigt die App keine Messwerte.
Das Ergebnisschema bleibt unverändert.

## Neu erstellen

Mit gesetztem `ANTHROPIC_API_KEY` (kostenpflichtige API-Aufrufe):

```powershell
python -m copilot.cli examples/job_anforderungen.pdf examples/job_profil.pdf --mode job --output examples/result_job.json --metadata examples/result_job.meta.json
python -m copilot.cli examples/tender_anforderungen.pdf examples/tender_profil.pdf --mode tender --output examples/result_tender.json --metadata examples/result_tender.meta.json
```

Die CLI schreibt weiterhin das reine Analyse-JSON auf stdout. Die optionalen
Ausgaben speichern nur das ausdrücklich angeforderte Ergebnis und seine Messwerte.
Keine PDFs werden kopiert. Modellüberschreibungen über `MODEL_EXTRACT` und
`MODEL_ASSESS` gelten auch hier; für die obigen Messungen wurden die Standards verwendet.

Beim Tender-Beispiel wurden Zuschlagskriterien, Fristen und Einreichungsformalien
ausgeschlossen. Die geforderte gültige ISO/IEC-27001-Zertifizierung ist ein
nicht belegtes Muss-Kriterium (`beleg: null`).
