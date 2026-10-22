"""Gemeinsame Laufzeitbudgets für API-Aufrufe und die Web-Belegung."""

API_TIMEOUT_SECONDS = 60.0
SDK_MAX_RETRIES = 1
RESPONSE_CORRECTION_RETRIES = 1
ANALYSIS_API_STAGES = ("extraction", "assessment")
ANALYSIS_BUFFER_SECONDS = 30.0

# anthropic 0.116.0 akzeptiert Retry-After bis 60s; sein Backoff bleibt darunter.
# Auch diese Pausen können in jedem Korrekturversuch jeder Stufe auftreten.
SDK_MAX_RETRY_WAIT_SECONDS = 60.0


def analysis_lease_seconds() -> float:
    attempts_per_stage = 1 + RESPONSE_CORRECTION_RETRIES
    seconds_per_attempt = (
        API_TIMEOUT_SECONDS * (1 + SDK_MAX_RETRIES)
        + SDK_MAX_RETRY_WAIT_SECONDS * SDK_MAX_RETRIES
    )
    return len(ANALYSIS_API_STAGES) * attempts_per_stage * seconds_per_attempt + ANALYSIS_BUFFER_SECONDS
