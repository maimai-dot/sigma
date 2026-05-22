"""Sigma — Generic Multi-Agent Collaboration Framework.

Zero-crewAI, zero-domain-knowledge. Inject everything via SigmaConfig + ToolSpec.
"""

__version__ = "0.1.0"

from sigma.config import SigmaConfig
from sigma.agent import Agent, BaseTool
from sigma.llm import (
    LLMBackend, UniversalBackend, OpenAIBackend,  # OpenAIBackend kept for backward compat
    AsyncLLMBackend, AsyncUniversalBackend, AsyncOpenAIBackend,
    LLMResponse, RetryConfig,
    RateLimiter, AsyncRateLimiter, AgentLLMConfig,
    # Multimodal
    build_message, text_content, image_url, encode_image_base64,
    is_multimodal, ContentPart, ChatMessage, TextContent, ImageUrlContent,
)
from sigma.state import (
    SharedState,
    StateManager,
    RoundRecord,
    Conflict,
    Decision,
    AlarmFlag,
    ConsensusEstimate,
    ComplexityTier,
    ComplexityAssessment,
)
from sigma.convergence import ConvergenceJudge, Verdict, JudgeResult
from sigma.triggers import TriggerSystem
from sigma.replay import ReplayPlayer
from sigma.cost_tracker import CostTracker
from sigma.discovery import (
    discover_agents_from_dir,
    discover_tools_from_dir,
    load_skills_from_dir,
)
from sigma.protocol import SigmaProtocol, AgentSpec, ToolSpec
from sigma.orchestrator import SigmaOrchestrator
from sigma.hooks import HookSystem, HookPoint
from sigma.human_gate import HumanGate, GateDecision, GateAction
from sigma.code_sandbox import CodeSandbox, validate_code as validate_sandbox_code
from sigma.cache import CacheConfig, CachedLLMBackend, AsyncCachedLLMBackend
from sigma.schema_validator import validate_against_schema
from sigma.memory import MemoryStore
from sigma.provenance import AuditTrail, ProvenanceEntry
from sigma.pydantic_validator import (
    PydanticOutputParser, PydanticValidationError, validate_output,
)
from sigma.tool_cache import ToolCache
from sigma.knowledge import KnowledgeBase, Chunk, RetrievalResult
from sigma.guardrails import (
    Guardrail, GuardrailSet, GuardResult, GuardrailReport, Severity,
    RangeCheck, CrossParamCheck, SetCheck, CustomCheck,
)
from sigma.learning import (
    ExecutionRecord, LearningStore, record_from_tau_state,
)
from sigma.observability import (
    init_tracing, get_tracer, is_initialized,
    traced, tracing_span,
)
from sigma.log import get_logger, setup_logging

# Standard Tools
from sigma.tools import (
    HttpApiTool,
    FileSystemTool,
    SQLDatabaseTool,
    JsonTool,
    CsvTool,
    TxtGrepTool,
    WebScrapeTool,
    DirectorySearchTool,
    PdfTool,
    ExcelTool,
    DocxTool,
)

# Benchmark (optional — import on demand to avoid circular deps)
from sigma.benchmark.tasks import BenchmarkTask, TASKS as BENCHMARK_TASKS
from sigma.benchmark.metrics import compute_all_metrics
from sigma.benchmark.runner import run_suite_replay, BenchmarkSuiteResult
from sigma.benchmark.reporter import generate_markdown_report, generate_json_report, save_report

# Tau (hierarchical decomposition + graduated conflict resolution)
from sigma.tau import (
    SubTask, TaskGraph, SubtaskResult, InterfaceConflict, ConflictReport,
    ResolutionResult, TauState,
    TauDecomposer, IndependentExecutor, ExecutorConfig, AsyncIndependentExecutor,
    InterfaceConflictDetector, TauResolver, TauOrchestrator,
    select_mode, ModeSelection,
    AgentCapability, CapabilityRegistry,
    TauBenchmarkTask, TauBenchmarkMetrics, TAU_BENCHMARK_TASKS,
    score_decomposition_quality, score_resolution_effectiveness,
    score_convergence_efficiency, compute_tau_composite,
    run_tau_benchmark_replay, run_tau_suite_replay, TauSuiteResult,
)

__all__ = [
    # Config
    "SigmaConfig",
    # Agent
    "Agent",
    "BaseTool",
    # LLM
    "LLMBackend",
    "UniversalBackend",
    "OpenAIBackend",         # deprecated alias
    "AsyncLLMBackend",
    "AsyncUniversalBackend",
    "AsyncOpenAIBackend",    # deprecated alias
    "LLMResponse",
    "RetryConfig",
    "RateLimiter",
    "AsyncRateLimiter",
    "AgentLLMConfig",
    # State
    "SharedState",
    "StateManager",
    "RoundRecord",
    "Conflict",
    "Decision",
    "AlarmFlag",
    "ConsensusEstimate",
    "ComplexityTier",
    "ComplexityAssessment",
    # Convergence
    "ConvergenceJudge",
    "Verdict",
    "JudgeResult",
    # Triggers
    "TriggerSystem",
    # Replay
    "ReplayPlayer",
    # Cost
    "CostTracker",
    # Discovery
    "discover_agents_from_dir",
    "discover_tools_from_dir",
    "load_skills_from_dir",
    # Protocol
    "SigmaProtocol",
    "AgentSpec",
    "ToolSpec",
    # Orchestrator
    "SigmaOrchestrator",
    # Hooks
    "HookSystem",
    "HookPoint",
    # Human Gate
    "HumanGate",
    "GateDecision",
    "GateAction",
    # Code Sandbox
    "CodeSandbox",
    "validate_sandbox_code",
    # Cache
    "CacheConfig",
    "CachedLLMBackend",
    "AsyncCachedLLMBackend",
    # Schema
    "validate_against_schema",
    # Memory
    "MemoryStore",
    # Provenance
    "AuditTrail",
    "ProvenanceEntry",
    # Pydantic
    "PydanticOutputParser",
    "PydanticValidationError",
    "validate_output",
    # Tool Cache
    "ToolCache",
    # Guardrails
    "Guardrail",
    "GuardrailSet",
    "GuardResult",
    "GuardrailReport",
    "Severity",
    "RangeCheck",
    "CrossParamCheck",
    "SetCheck",
    "CustomCheck",
    # Learning
    "ExecutionRecord",
    "LearningStore",
    "record_from_tau_state",
    # Observability
    "init_tracing",
    "get_tracer",
    "is_initialized",
    "traced",
    "tracing_span",
    # Knowledge Base
    "KnowledgeBase",
    "Chunk",
    "RetrievalResult",
    # Multimodal
    "build_message",
    "text_content",
    "image_url",
    "encode_image_base64",
    "is_multimodal",
    "ContentPart",
    "ChatMessage",
    "TextContent",
    "ImageUrlContent",
    # Logging
    "get_logger",
    "setup_logging",
    # Standard Tools
    "HttpApiTool",
    "FileSystemTool",
    "SQLDatabaseTool",
    "JsonTool",
    "CsvTool",
    "TxtGrepTool",
    "WebScrapeTool",
    "DirectorySearchTool",
    "PdfTool",
    "ExcelTool",
    "DocxTool",
    # Benchmark
    "BenchmarkTask",
    "BENCHMARK_TASKS",
    "compute_all_metrics",
    "run_suite_replay",
    "BenchmarkSuiteResult",
    "generate_markdown_report",
    "generate_json_report",
    "save_report",
    # Tau
    "SubTask",
    "TaskGraph",
    "SubtaskResult",
    "InterfaceConflict",
    "ConflictReport",
    "ResolutionResult",
    "TauState",
    "TauDecomposer",
    "IndependentExecutor",
    "ExecutorConfig",
    "AsyncIndependentExecutor",
    "InterfaceConflictDetector",
    "TauResolver",
    "TauOrchestrator",
    "select_mode",
    "ModeSelection",
    "AgentCapability",
    "CapabilityRegistry",
    # Tau Benchmark
    "TauBenchmarkTask",
    "TauBenchmarkMetrics",
    "TAU_BENCHMARK_TASKS",
    "score_decomposition_quality",
    "score_resolution_effectiveness",
    "score_convergence_efficiency",
    "compute_tau_composite",
    "run_tau_benchmark_replay",
    "run_tau_suite_replay",
    "TauSuiteResult",
]
