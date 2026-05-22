"""Tests for provenance.py — ProvenanceEntry, AuditTrail."""

from sigma.provenance import ProvenanceEntry, AuditTrail


class TestProvenanceEntry:
    def test_creation(self):
        pe = ProvenanceEntry(
            parameter="thrust_n", value=1500, source_type="tool",
            source_name="thrust_calc", round_num=1,
            evidence="raw output", confidence="HIGH",
        )
        assert pe.parameter == "thrust_n"
        assert pe.value == 1500
        assert pe.source_type == "tool"
        assert pe.source_name == "thrust_calc"
        assert pe.round_num == 1
        assert pe.evidence == "raw output"
        assert pe.confidence == "HIGH"

    def test_defaults(self):
        pe = ProvenanceEntry(
            parameter="mass_kg", value=5.0, source_type="manual",
            source_name="Structures Chief", round_num=2,
        )
        assert pe.evidence == ""
        assert pe.confidence == ""
        assert len(pe.timestamp) > 0  # auto-generated

    def test_roundtrip_serialization(self):
        pe = ProvenanceEntry(
            parameter="thrust_n", value=1500, source_type="tool",
            source_name="thrust_calc", round_num=1,
            evidence="ok", confidence="HIGH",
        )
        d = pe.to_dict()
        pe2 = ProvenanceEntry.from_dict(d)
        assert pe2.parameter == pe.parameter
        assert pe2.value == pe.value
        assert pe2.source_type == pe.source_type
        assert pe2.confidence == "HIGH"


class TestAuditTrail:
    def test_empty_trail(self):
        at = AuditTrail()
        assert len(at) == 0
        assert at.trace("thrust_n") == []

    def test_add_and_trace(self):
        at = AuditTrail()
        at.add("thrust_n", 1500, "tool", "thrust_calc", 1, evidence="ok", confidence="HIGH")
        at.add("thrust_n", 1550, "agent_estimate", "Propulsion", 2, evidence="recalc", confidence="MEDIUM")
        at.add("mass_kg", 5.0, "tool", "mass_calc", 1)

        thrust_trace = at.trace("thrust_n")
        assert len(thrust_trace) == 2
        assert thrust_trace[0].value == 1500
        assert thrust_trace[1].value == 1550

    def test_latest(self):
        at = AuditTrail()
        at.add("x", 1.0, "tool", "t1", 1)
        at.add("x", 2.0, "tool", "t1", 2)
        assert at.latest("x").value == 2.0
        assert at.latest("nonexistent") is None

    def test_by_source(self):
        at = AuditTrail()
        at.add("a", 1, "tool", "t1", 1)
        at.add("b", 2, "agent_estimate", "AgentX", 1)
        at.add("c", 3, "tool", "t1", 2)

        t1 = at.by_source("t1")
        assert len(t1) == 2

        ax = at.by_source("AgentX")
        assert len(ax) == 1
        assert ax[0].parameter == "b"

    def test_by_round(self):
        at = AuditTrail()
        at.add("a", 1, "tool", "t1", 1)
        at.add("b", 2, "tool", "t1", 2)
        at.add("c", 3, "tool", "t1", 2)

        assert len(at.by_round(1)) == 1
        assert len(at.by_round(2)) == 2
        assert len(at.by_round(3)) == 0

    def test_roundtrip_serialization(self):
        at = AuditTrail()
        at.add("thrust_n", 1500, "tool", "thrust_calc", 1, "ok", "HIGH")
        at.add("mass_kg", 5.0, "consensus", "consensus", 2, "agreed", "MEDIUM")

        d = at.to_dict()
        at2 = AuditTrail.from_dict(d)
        assert len(at2) == 2
        assert at2.trace("thrust_n")[0].value == 1500
        assert at2.trace("mass_kg")[0].source_type == "consensus"
