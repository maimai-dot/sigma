"""Audit Trail — parameter-level provenance tracking.

Every value in task_params gets a ProvenanceEntry recording
where it came from, which agent/tool produced it, and the evidence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ProvenanceEntry:
    """Single provenance record for one parameter value."""

    parameter: str       # e.g. "thrust_n"
    value: object        # the value (float, str, etc.)
    source_type: str     # "tool" | "agent_estimate" | "consensus" | "manual" | "decision"
    source_name: str     # tool name, agent name, or "consensus"
    round_num: int
    evidence: str = ""   # raw result or reasoning snippet
    confidence: str = ""  # "HIGH" | "MEDIUM" | "LOW"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "round_num": self.round_num,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenanceEntry":
        return cls(
            parameter=d["parameter"],
            value=d.get("value"),
            source_type=d["source_type"],
            source_name=d["source_name"],
            round_num=d["round_num"],
            evidence=d.get("evidence", ""),
            confidence=d.get("confidence", ""),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class AuditTrail:
    """Collects and queries ProvenanceEntry records."""

    entries: list[ProvenanceEntry] = field(default_factory=list)

    def add(
        self, parameter: str, value: object, source_type: str,
        source_name: str, round_num: int,
        evidence: str = "", confidence: str = "",
    ) -> ProvenanceEntry:
        entry = ProvenanceEntry(
            parameter=parameter, value=value, source_type=source_type,
            source_name=source_name, round_num=round_num,
            evidence=evidence, confidence=confidence,
        )
        self.entries.append(entry)
        return entry

    def trace(self, parameter: str) -> list[ProvenanceEntry]:
        """All provenance records for a given parameter, oldest first."""
        return [e for e in self.entries if e.parameter == parameter]

    def latest(self, parameter: str) -> ProvenanceEntry | None:
        matches = self.trace(parameter)
        return matches[-1] if matches else None

    def by_source(self, source_name: str) -> list[ProvenanceEntry]:
        return [e for e in self.entries if e.source_name == source_name]

    def by_round(self, round_num: int) -> list[ProvenanceEntry]:
        return [e for e in self.entries if e.round_num == round_num]

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> "AuditTrail":
        return cls(entries=[ProvenanceEntry.from_dict(e) for e in d.get("entries", [])])

    def __len__(self) -> int:
        return len(self.entries)
