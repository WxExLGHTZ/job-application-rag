"""Validierte Ein- und Ausgabeformen der beiden Analyse-Stufen."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


Mode = Literal["job", "tender"]
RequirementType = Literal["muss", "kann"]
Status = Literal["erfuellt", "teilweise", "nicht_belegt"]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class ExtractedRequirement(StrictModel):
    text: Text
    typ: RequirementType


class Evidence(StrictModel):
    zitat: Text
    seite: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def no_ellipsis(self) -> Self:
        if "..." in self.zitat or "…" in self.zitat:
            raise ValueError("Zitate dürfen keine Auslassungspunkte enthalten.")
        return self


class Assessment(StrictModel):
    id: Text
    status: Status
    beleg: Evidence | None
    begruendung: Text

    @model_validator(mode="after")
    def evidence_matches_status(self) -> Self:
        if (self.status == "nicht_belegt") != (self.beleg is None):
            raise ValueError(
                "nicht_belegt verlangt beleg=null; erfuellt/teilweise verlangen einen Beleg."
            )
        return self


class AssessmentResponse(StrictModel):
    """Nur die vom Modell zu liefernden Bewertungen und das Fazit."""

    requirements: list[Assessment]
    fazit: Text


class Requirement(Assessment):
    text: Text
    typ: RequirementType


class Summary(StrictModel):
    score: Annotated[int, Field(ge=0, le=100)]
    muss_erfuellt: Annotated[str, StringConstraints(pattern=r"^\d+/\d+$")]
    fazit: Text


class Usage(StrictModel):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt


class AnalysisResult(StrictModel):
    mode: Mode
    summary: Summary
    requirements: list[Requirement]
    usage: Usage

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [requirement.id for requirement in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("Anforderungs-IDs müssen eindeutig sein.")
        return self
