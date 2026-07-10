"""Claude calls and robust JSON parsing for the matching stage."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from anthropic import Anthropic


MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000

_CLIENT: Anthropic | None = None


class LlmError(RuntimeError):
    """Base error for understandable LLM failures."""


class MissingApiKeyError(LlmError):
    """Raised when ANTHROPIC_API_KEY is not available."""


class LlmResponseError(LlmError):
    """Raised when Claude returns unusable or invalid JSON."""


def extract_requirements(job_ad_text: str) -> list[dict[str, Any]]:
    """Extract concrete job requirements from a job ad with Claude.

    Args:
        job_ad_text: Full text of the job ad.

    Returns:
        A list of requirement dictionaries. Each item contains at least
        "requirement" and may contain fields such as "category".
    """
    if not job_ad_text.strip():
        raise ValueError("Die Stellenanzeige ist leer.")

    # System-Prompt = dauerhafte Rolle/Regeln/Output-Form.
    # User-Prompt = konkrete Eingabedaten fuer genau diesen Aufruf.
    system_prompt = """
Du bist ein praeziser Assistent fuer Bewerbungsanalysen.
Extrahiere aus einer Stellenanzeige nur konkrete Anforderungen, Aufgaben oder
Erwartungen, die spaeter gegen ein Bewerberprofil geprueft werden koennen.

Antworte ausschliesslich mit validem JSON, ohne Markdown und ohne Erklaertext.
Das JSON muss eine Liste sein. Jedes Element hat dieses Schema:
{
  "requirement": "kurzer, konkreter Anforderungstext",
  "category": "muss | kann | aufgabe | soft_skill | sonstiges",
  "reason": "kurze Begruendung, warum dies eine relevante Anforderung ist"
}
""".strip()

    user_prompt = f"""
Stellenanzeige:

{job_ad_text}
""".strip()

    def parse_requirements(response_text: str) -> list[dict[str, Any]]:
        requirements = _parse_json_response(response_text)

        if not isinstance(requirements, list):
            raise LlmResponseError(
                "Claude sollte eine JSON-Liste mit Anforderungen liefern, "
                "hat aber etwas anderes geliefert."
            )

        for index, requirement in enumerate(requirements, start=1):
            if not isinstance(requirement, dict):
                raise LlmResponseError(
                    f"Anforderung {index} ist kein JSON-Objekt."
                )
            requirement_text = requirement.get("requirement")
            if not isinstance(requirement_text, str) or not requirement_text.strip():
                raise LlmResponseError(
                    f"Anforderung {index} enthaelt kein Feld 'requirement' mit Text."
                )

        return requirements

    return _send_and_parse_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        parse_response=parse_requirements,
    )


def assess_requirements(requirements_with_context: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess requirements against retrieved profile chunks with Claude.

    Args:
        requirements_with_context: Requirement dictionaries, each augmented with
            the chunks retrieved for that requirement.

    Returns:
        A JSON-compatible dictionary with matrix, gap framing, highlighted
        projects, and likely interview questions.
    """
    if not requirements_with_context:
        raise ValueError("Es gibt keine Anforderungen fuer die Bewertung.")

    system_prompt = """
Du bist ein ehrlicher Bewerbungs-Copilot.
Bewerte Anforderungen ausschliesslich auf Basis der mitgelieferten Profil-Chunks.
Erfinde keine Erfahrung, keine Projekte und keine Nachweise. Wenn ein Beleg
fehlt oder zu schwach ist, markiere das klar als Luecke.

Antworte ausschliesslich mit validem JSON, ohne Markdown und ohne Erklaertext.
Das JSON muss genau diese vier Hauptteile enthalten:
{
  "anforderungs_matrix": [
    {
      "requirement": "Anforderung",
      "status": "erfuellt | teilweise | luecke",
      "beleg": {
        "filename": "Herkunftsdatei oder null",
        "text": "konkreter Profil-Auszug oder null",
        "score": 0.0
      },
      "begruendung": "kurze, konkrete Bewertung"
    }
  ],
  "framing_vorschlaege": [
    {
      "luecke": "fehlender oder schwacher Nachweis",
      "ehrliches_framing": "Formulierungsidee ohne Uebertreibung"
    }
  ],
  "hervorzuhebende_projekte": [
    {
      "projekt": "Projekt/Modul/Erfahrung",
      "warum_relevant": "Bezug zur Stelle",
      "beleg_filename": "Herkunftsdatei oder null"
    }
  ],
  "wahrscheinliche_interviewfragen": [
    {
      "frage": "Interviewfrage",
      "grund": "Warum diese Frage wahrscheinlich ist"
    }
  ]
}
""".strip()

    user_payload = {
        "requirements_with_context": requirements_with_context,
        "hinweis": (
            "Jeder retrieved_chunk ist ein moeglicher Beleg aus dem Profil. "
            "Nutze nur diese Chunks."
        ),
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    def parse_assessment(response_text: str) -> dict[str, Any]:
        assessment = _parse_json_response(response_text)

        if not isinstance(assessment, dict):
            raise LlmResponseError(
                "Claude sollte ein JSON-Objekt mit der Gesamtanalyse liefern, "
                "hat aber etwas anderes geliefert. "
                f"Rohantwort (erste 500 Zeichen): {response_text[:500]!r}"
            )

        return assessment

    return _send_and_parse_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        parse_response=parse_assessment,
    )


def _get_client() -> Anthropic:
    global _CLIENT

    if _CLIENT is None:
        try:
            api_key = os.environ["ANTHROPIC_API_KEY"]
        except KeyError as error:
            raise MissingApiKeyError(
                "ANTHROPIC_API_KEY ist nicht gesetzt. Bitte setze den API-Key "
                "als Umgebungsvariable und starte den Befehl erneut."
            ) from error

        _CLIENT = Anthropic(api_key=api_key)

    return _CLIENT


def _send_message(system_prompt: str, user_prompt: str) -> str:
    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except MissingApiKeyError:
        raise
    except Exception as error:
        raise LlmError(
            f"Claude API-Aufruf fehlgeschlagen ({type(error).__name__}: {error})"
        ) from error

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise LlmResponseError(
            "Antwort wurde wegen Token-Limit abgeschnitten, erhöhe MAX_TOKENS."
        )

    text_parts: list[str] = []
    for block in response.content:
        block_text = getattr(block, "text", None)
        if isinstance(block_text, str):
            text_parts.append(block_text)

    response_text = "\n".join(text_parts).strip()
    if not response_text:
        raise LlmResponseError("Claude hat keine Textantwort geliefert.")

    return response_text


def _send_and_parse_with_retry(
    system_prompt: str,
    user_prompt: str,
    parse_response: Callable[[str], Any],
) -> Any:
    for attempt in range(2):
        try:
            response_text = _send_message(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return parse_response(response_text)
        except LlmResponseError:
            if attempt == 1:
                raise

    raise RuntimeError("Unerreichbarer Retry-Zustand.")


def _parse_json_response(response_text: str) -> Any:
    cleaned_text = _remove_outer_code_fence(response_text)
    index = 0

    while index < len(cleaned_text):
        char = cleaned_text[index]
        if char not in "[{":
            index += 1
            continue

        end = _find_json_end(cleaned_text, index)
        if end is None:
            index += 1
            continue

        candidate = cleaned_text[index:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            index = end
            continue

    snippet = cleaned_text[:800].replace("\n", " ")
    raise LlmResponseError(
        "Claude hat kein parsebares JSON geliefert. "
        f"Anfang der Antwort: {snippet!r}"
    )


def _remove_outer_code_fence(text: str) -> str:
    stripped = text.strip()
    lines = stripped.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()

def _find_json_end(text: str, start: int) -> int | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escaped = False

    for index in range(start + 1, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return index + 1

    return None

