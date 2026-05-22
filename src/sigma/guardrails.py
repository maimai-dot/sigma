"""Guardrails — pluggable output validation for Sigma and Tau pipelines.

Guardrails catch physically impossible or internally inconsistent outputs
before they pollute downstream decisions. They annotate rather than crash:
BLOCK severity prevents propagation, WARN annotates with caution, PASS is silent.

Design:
  - Each Guardrail is a callable (value, context) → GuardResult
  - GuardrailSet groups related checks with AND/OR logic
  - Built-in rules cover common physical and cross-parameter constraints
  - Domain-specific rules can be injected via SigmaConfig or TauConfig

Usage:
    from sigma.guardrails import RangeCheck, CrossParamCheck, GuardrailSet

    guards = GuardrailSet([
        RangeCheck("thrust_n", min_val=0.1, max_val=1e6),
        RangeCheck("isp_s", min_val=50, max_val=500),
        CrossParamCheck("thrust_n", "mass_kg",
                         check="ratio",
                         min_ratio=0.5, max_ratio=500,
                         description="TWR sanity"),
    ])

    result = guards.check_all({"thrust_n": 1500, "isp_s": 180, "mass_kg": 5})
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class Severity(Enum):
    BLOCK = "BLOCK"    # Prevents result from propagating
    WARN = "WARN"      # Result propagates with warning annotation
    PASS = "PASS"      # Silent — no issue


@dataclass(frozen=True)
class GuardResult:
    """Result of a single guardrail check."""
    severity: Severity
    message: str = ""
    guard_name: str = ""
    param_key: str = ""
    actual_value: float | None = None

    @property
    def ok(self) -> bool:
        return self.severity != Severity.BLOCK

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "guard_name": self.guard_name,
            "param_key": self.param_key,
            "actual_value": self.actual_value,
        }


@dataclass(frozen=True)
class GuardrailReport:
    """Aggregate result from a GuardrailSet check."""
    results: list[GuardResult] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return any(r.severity == Severity.BLOCK for r in self.results)

    @property
    def warnings(self) -> list[GuardResult]:
        return [r for r in self.results if r.severity == Severity.WARN]

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def summary(self) -> str:
        if not self.results:
            return "No guardrails defined"
        blocks = sum(1 for r in self.results if r.severity == Severity.BLOCK)
        warns = sum(1 for r in self.results if r.severity == Severity.WARN)
        passes = sum(1 for r in self.results if r.severity == Severity.PASS)
        parts = []
        if blocks:
            parts.append(f"{blocks} BLOCK")
        if warns:
            parts.append(f"{warns} WARN")
        if passes:
            parts.append(f"{passes} PASS")
        return ", ".join(parts) if parts else "All passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "blocked": self.blocked,
            "summary": self.summary,
        }


# ═══════════════════════════════════════════════════════════════════════
# Guardrail Base
# ═══════════════════════════════════════════════════════════════════════

class Guardrail:
    """Base class for guardrail checks.

    Subclasses override check().
    """

    name: str = ""

    def __call__(self, params: dict[str, float], context: dict[str, Any] | None = None) -> GuardResult:
        ctx = context or {}
        return self.check(params, ctx)

    def check(self, params: dict[str, float], context: dict[str, Any]) -> GuardResult:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# Built-in Guardrails
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RangeCheck(Guardrail):
    """Validate a param falls within [min_val, max_val]."""
    param_key: str
    min_val: float | None = None
    max_val: float | None = None
    exclude_zero: bool = False

    def __post_init__(self):
        object.__setattr__(self, "name", f"RangeCheck({self.param_key})")

    def check(self, params: dict[str, float], context: dict[str, Any]) -> GuardResult:
        val = params.get(self.param_key)
        if val is None:
            return GuardResult(Severity.PASS, guard_name=self.name, param_key=self.param_key)

        if self.exclude_zero and val == 0:
            return GuardResult(
                Severity.BLOCK, guard_name=self.name, param_key=self.param_key,
                message=f"{self.param_key}={val} — zero is physically impossible",
                actual_value=val,
            )
        if self.min_val is not None and val < self.min_val:
            return GuardResult(
                Severity.WARN if val > 0 else Severity.BLOCK,
                guard_name=self.name, param_key=self.param_key,
                message=f"{self.param_key}={val} below minimum {self.min_val}",
                actual_value=val,
            )
        if self.max_val is not None and val > self.max_val:
            return GuardResult(
                Severity.WARN, guard_name=self.name, param_key=self.param_key,
                message=f"{self.param_key}={val} above maximum {self.max_val}",
                actual_value=val,
            )
        return GuardResult(Severity.PASS, guard_name=self.name, param_key=self.param_key)


@dataclass
class CrossParamCheck(Guardrail):
    """Validate relationship between two params.

    Supported check_types:
      - "ratio": param_a / param_b should be in [min_ratio, max_ratio]
      - "less_than": param_a must be <= param_b
      - "greater_than": param_a must be >= param_b
    """
    param_a: str
    param_b: str
    check_type: str = "ratio"       # "ratio" | "less_than" | "greater_than"
    min_ratio: float | None = None
    max_ratio: float | None = None
    description: str = ""

    def __post_init__(self):
        object.__setattr__(self, "name", f"CrossParamCheck({self.param_a}, {self.param_b})")

    def check(self, params: dict[str, float], context: dict[str, Any]) -> GuardResult:
        a = params.get(self.param_a)
        b = params.get(self.param_b)
        if a is None or b is None:
            return GuardResult(Severity.PASS, guard_name=self.name)

        desc = self.description or f"{self.param_a} vs {self.param_b}"

        if self.check_type == "ratio" and b != 0:
            ratio = a / b
            if self.min_ratio is not None and ratio < self.min_ratio:
                return GuardResult(
                    Severity.WARN, guard_name=self.name,
                    message=f"{desc}: ratio {ratio:.2f} below minimum {self.min_ratio}",
                )
            if self.max_ratio is not None and ratio > self.max_ratio:
                return GuardResult(
                    Severity.WARN, guard_name=self.name,
                    message=f"{desc}: ratio {ratio:.2f} above maximum {self.max_ratio}",
                )

        elif self.check_type == "less_than" and a > b:
            return GuardResult(
                Severity.WARN, guard_name=self.name,
                message=f"{desc}: {self.param_a}={a} > {self.param_b}={b}",
            )

        elif self.check_type == "greater_than" and a < b:
            return GuardResult(
                Severity.WARN, guard_name=self.name,
                message=f"{desc}: {self.param_a}={a} < {self.param_b}={b}",
            )

        return GuardResult(Severity.PASS, guard_name=self.name)


@dataclass
class SetCheck(Guardrail):
    """Validate a param belongs to a set of valid values (or avoids invalid ones)."""
    param_key: str
    valid_set: set[float] | None = None
    invalid_set: set[float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "name", f"SetCheck({self.param_key})")

    def check(self, params: dict[str, float], context: dict[str, Any]) -> GuardResult:
        val = params.get(self.param_key)
        if val is None:
            return GuardResult(Severity.PASS, guard_name=self.name)

        if self.valid_set is not None and val not in self.valid_set:
            return GuardResult(
                Severity.WARN, guard_name=self.name, param_key=self.param_key,
                message=f"{self.param_key}={val} not in valid set {self.valid_set}",
            )
        if self.invalid_set is not None and val in self.invalid_set:
            return GuardResult(
                Severity.BLOCK, guard_name=self.name, param_key=self.param_key,
                message=f"{self.param_key}={val} is in the invalid set",
            )
        return GuardResult(Severity.PASS, guard_name=self.name)


class CustomCheck(Guardrail):
    """Arbitrary check function: (params, context) -> GuardResult."""

    def __init__(self, name: str, fn: Callable[[dict[str, float], dict[str, Any]], GuardResult]):
        self.name = name
        self._fn = fn

    def check(self, params: dict[str, float], context: dict[str, Any]) -> GuardResult:
        return self._fn(params, context)


# ═══════════════════════════════════════════════════════════════════════
# GuardrailSet
# ═══════════════════════════════════════════════════════════════════════

class GuardrailSet:
    """Collection of guardrails with AND logic (all must pass)."""

    def __init__(self, guardrails: list[Guardrail] | None = None):
        self._guardrails: list[Guardrail] = list(guardrails or [])

    def add(self, guardrail: Guardrail) -> "GuardrailSet":
        self._guardrails.append(guardrail)
        return self

    def add_range(self, param_key: str, min_val: float | None = None,
                  max_val: float | None = None, exclude_zero: bool = False) -> "GuardrailSet":
        return self.add(RangeCheck(param_key, min_val=min_val, max_val=max_val, exclude_zero=exclude_zero))

    def add_ratio(self, param_a: str, param_b: str,
                  min_ratio: float | None = None, max_ratio: float | None = None,
                  description: str = "") -> "GuardrailSet":
        return self.add(CrossParamCheck(param_a, param_b, check_type="ratio",
                                        min_ratio=min_ratio, max_ratio=max_ratio,
                                        description=description))

    def add_custom(self, name: str, fn: Callable) -> "GuardrailSet":
        return self.add(CustomCheck(name, fn))

    def check_all(self, params: dict[str, float],
                  context: dict[str, Any] | None = None) -> GuardrailReport:
        ctx = context or {}
        results = []
        for g in self._guardrails:
            try:
                r = g(params, ctx)
                results.append(r)
            except Exception as e:
                results.append(GuardResult(
                    Severity.WARN,
                    guard_name=g.name,
                    message=f"Guardrail error: {e}",
                ))
        return GuardrailReport(results=results, context=ctx)

    def __len__(self) -> int:
        return len(self._guardrails)

    def __bool__(self) -> bool:
        return len(self._guardrails) > 0


# ═══════════════════════════════════════════════════════════════════════
# Rocket Domain Presets
# ═══════════════════════════════════════════════════════════════════════

def rocket_knsb_guardrails() -> GuardrailSet:
    """Preset guardrails for KNSB solid rocket motor design."""
    return GuardrailSet([
        # Thrust: amateur KNSB typically 10-10000N
        RangeCheck("thrust_n", min_val=0.1, max_val=50000),
        RangeCheck("thrust_N", min_val=0.1, max_val=50000),
        # Isp: KNSB theoretical max ~183s at 50bar, engineering 120-170s
        RangeCheck("isp_s", min_val=80, max_val=250, exclude_zero=True),
        # Chamber pressure: amateur range 10-200 bar
        RangeCheck("chamber_pressure_bar", min_val=1, max_val=300),
        RangeCheck("chamber_pressure", min_val=1, max_val=300),
        # Mass: amateur rocket 0.1-500kg
        RangeCheck("mass_kg", min_val=0.05, max_val=1000, exclude_zero=True),
        RangeCheck("total_mass", min_val=0.05, max_val=1000, exclude_zero=True),
        # Nozzle throat diameter: 1-100mm
        RangeCheck("throat_diameter_mm", min_val=0.5, max_val=200),
        # Wall thickness: 0.5-20mm
        RangeCheck("wall_thickness_mm", min_val=0.3, max_val=30),
        # TWR sanity: thrust(N) / (mass(kg) * 9.81) should be 1-100
        CrossParamCheck("thrust_n", "mass_kg", check_type="ratio",
                        min_ratio=9.81, max_ratio=5000,
                        description="推力质量比 (TWR)"),
        CrossParamCheck("thrust_n", "chamber_pressure_bar", check_type="ratio",
                        min_ratio=1, max_ratio=5000,
                        description="推力/室压比"),
    ])
