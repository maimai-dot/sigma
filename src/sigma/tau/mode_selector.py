"""Execution mode selector — auto-detects whether to use Sigma AERC or Tau hierarchical.

Zero-LLM rule-based classifier. Mirrors the _assess_complexity() pattern:
fast, deterministic, free.

Decision logic:
  Tau (hierarchical decomposition):
    - Sequential steps: "first...then...finally", "步骤1", "Step 1"
    - Multi-part: 3+ distinct subtasks signaled by connectors
    - Cross-department: mentions 3+ different domain keywords
    - Parallelizable: "同时", "并行", "respectively"

  Sigma (collaborative AERC):
    - Single analytical question: "calculate", "analyze", "compare"
    - Deep collaborative: requires consensus, cross-review
    - Design exploration: "design", "optimize", "方案"
    - Single focused domain
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSelection:
    mode: str              # "sigma" | "tau"
    reason: str
    confidence: str        # "HIGH" | "MEDIUM" | "LOW"


# ── Detection Patterns ────────────────────────────────────────────

_SEQUENTIAL_PATTERNS = [
    # Chinese sequential markers
    r'首先.*然后', r'第一[步个].*第二[步个]', r'步骤\s*\d',
    r'先.*再.*最后', r'之后.*接着', r'接下来',
    # English sequential markers
    r'first.*then', r'step\s*\d', r'first.*second',
    r'phase\s*\d', r'stage\s*\d',
]

_PARALLEL_PATTERNS = [
    r'同时', r'并行', r'分别', r'同步',
    r'simultaneously', r'in parallel', r'respectively',
    r'concurrently', r'each.*independently',
]

_MULTI_PART_CONNECTORS = [
    r'以及', r'并且', r'另外', r'此外', r'还有',
    r'(?:^|\s)和(?:\s|$)', r'与',
    r'and also', r'as well as', r'in addition',
    r'furthermore', r'moreover',
]

# Keywords that strongly signal hierarchical decomposition
_TAU_STRONG_SIGNALS = [
    '拆解', '分配', '分派', '部门', '各负责',
    'decompose', 'assign', 'department each',
    'sub-task', 'subtask', 'work breakdown',
    '多方', '跨部门', '接口参数',
]

# Keywords that strongly signal collaborative AERC
_SIGMA_STRONG_SIGNALS = [
    '共识', '盲审', '交叉审查', '魔鬼代言人',
    'consensus', 'blind review', 'cross-review',
    '方案对比', '权衡分析', '多方案比较',
    'trade study', 'trade-off analysis',
]


def select_mode(instruction: str) -> ModeSelection:
    """Auto-detect the best execution mode for the given instruction.

    Returns ModeSelection with mode, reason, and confidence.
    """
    text = instruction.lower()
    tau_score = 0.0
    sigma_score = 0.0
    reasons: list[str] = []

    # ── 1. Strong signals (decisive) ──
    tau_strong_hits = sum(1 for kw in _TAU_STRONG_SIGNALS if kw in text)
    sigma_strong_hits = sum(1 for kw in _SIGMA_STRONG_SIGNALS if kw in text)
    tau_score += tau_strong_hits * 3.0
    sigma_score += sigma_strong_hits * 3.0
    if tau_strong_hits:
        reasons.append(f"检测到{tau_strong_hits}个Tau强信号")
    if sigma_strong_hits:
        reasons.append(f"检测到{sigma_strong_hits}个Sigma强信号")

    # ── 2. Sequential patterns ──
    seq_hits = sum(1 for pat in _SEQUENTIAL_PATTERNS if re.search(pat, text))
    tau_score += seq_hits * 2.0
    if seq_hits >= 2:
        reasons.append(f"{seq_hits}个顺序标记")
    elif seq_hits:
        reasons.append("检测到顺序步骤")

    # ── 3. Parallel/independent patterns ──
    par_hits = sum(1 for pat in _PARALLEL_PATTERNS if re.search(pat, text))
    tau_score += par_hits * 2.0
    if par_hits:
        reasons.append("检测到并行/独立子任务")

    # ── 4. Multi-part connectors (signals decomposability) ──
    connector_hits = sum(1 for pat in _MULTI_PART_CONNECTORS if re.search(pat, text))
    if connector_hits >= 3:
        tau_score += 1.5
        reasons.append(f"多项连接 ({connector_hits}个)")

    # ── 5. Task length (longer → more likely decomposable) ──
    length = len(instruction)
    if length > 400:
        tau_score += 1.0
    elif length > 200:
        tau_score += 0.5

    # ── 6. Question/query markers (single focused → Sigma) ──
    q_markers = ['什么是', '是多少', 'how much', 'what is', '？', '?']
    q_hits = sum(1 for m in q_markers if m in text)
    if q_hits:
        sigma_score += 1.5
        reasons.append("检测到单一问题")

    # ── 7. Compare/analyze/design verbs (single deep analysis → Sigma) ──
    analysis_verbs = ['分析', '评估', '比较', '对比', '权衡',
                      'analyze', 'evaluate', 'compare', 'trade-off']
    verb_hits = sum(1 for v in analysis_verbs if v in text)
    if verb_hits and seq_hits == 0 and par_hits == 0:
        sigma_score += 2.0
        reasons.append("单一分析/比较任务")

    # ── 8. Design/optimize (could go either way, lean Sigma for single-domain) ──
    design_verbs = ['设计', '优化', 'design', 'optimize']
    if any(v in text for v in design_verbs) and seq_hits == 0:
        sigma_score += 1.0

    # ── Decision ──
    if tau_score > sigma_score + 1.0:
        mode = "tau"
        confidence = "HIGH" if tau_score >= 3.0 else "MEDIUM"
    elif sigma_score > tau_score + 1.0:
        mode = "sigma"
        confidence = "HIGH" if sigma_score >= 3.0 else "MEDIUM"
    else:
        # Close scores — default to Sigma (safer, more tested)
        mode = "sigma"
        confidence = "LOW"
        reasons.append("信号不明确，默认Sigma")

    if not reasons:
        reasons.append("无明确模式信号")

    return ModeSelection(
        mode=mode,
        reason="; ".join(reasons),
        confidence=confidence,
    )
