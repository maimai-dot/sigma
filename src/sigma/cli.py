"""sigma CLI — zero-concept entry point for the Sigma framework.

Usage:
    sigma run "设计一枚KNSB固体火箭"           # auto-detect mode
    sigma run "设计火箭" --mode tau            # force Tau mode
    sigma run "分析推进方案" --mode sigma       # force Sigma AERC mode
    sigma run "设计火箭" -y                    # non-interactive
    sigma run "设计火箭" --output v42          # named output version
    sigma list                                 # list past outputs
    sigma view v11                             # read report
    sigma config                               # show current config
"""

import argparse
import os
import sys
from pathlib import Path


def _reconfigure_stdout():
    """Ensure stdout supports UTF-8 on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _find_dotenv() -> str | None:
    """Search for .env in common locations: CWD, parent dirs, RocketFactory path."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(os.environ.get("SIGMA_HOME", "")) / ".env",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _setup_env(require_key: bool = True) -> dict:
    """Load environment and return config dict for backend init."""
    from dotenv import load_dotenv

    env_file = _find_dotenv()
    if env_file:
        load_dotenv(env_file)
    load_dotenv()
    load_dotenv(".env")

    api_key = os.environ.get("SIGMA_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("SIGMA_BASE_URL") or os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com")
    model = os.environ.get("SIGMA_MODEL", "deepseek-v4-pro")

    if require_key and not api_key:
        _error("No API key found. Set SIGMA_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY) in .env or environment.")
        sys.exit(1)

    return {"api_key": api_key or "", "base_url": base_url, "model": model}


def _create_backend(env: dict):
    """Create LLM backend from env config."""
    from sigma.llm import UniversalBackend
    return UniversalBackend(api_key=env["api_key"], base_url=env["base_url"])


def cmd_run(args):
    """Execute a task using the Sigma framework."""
    from sigma.orchestrator import SigmaOrchestrator
    from sigma.config import SigmaConfig
    from sigma.protocol import AgentSpec
    from sigma.discovery import discover_agents_from_dir, discover_tools_from_dir, load_skills_from_dir

    env = _setup_env()
    backend = _create_backend(env)

    # ── Discover agents/tools/skills ──
    agents_dir = args.agents_dir or os.environ.get("SIGMA_AGENTS_DIR", "")
    tools_dir = args.tools_dir or os.environ.get("SIGMA_TOOLS_DIR", "")
    skills_dir = args.skills_dir or os.environ.get("SIGMA_SKILLS_DIR", "")

    agents = {}
    tools = {}
    skills = {}

    if agents_dir and Path(agents_dir).exists():
        agents = discover_agents_from_dir(agents_dir)
        _info(f"发现 {len(agents)} 个智能体: {', '.join(agents.keys())}")
    if tools_dir and Path(tools_dir).exists():
        tools = discover_tools_from_dir(tools_dir)
        _info(f"发现 {len(tools)} 个工具: {', '.join(tools.keys())}")
    if skills_dir and Path(skills_dir).exists():
        skills = load_skills_from_dir(skills_dir)
        _info(f"加载 {len(skills)} 个技能")

    if not agents:
        from sigma.protocol import AgentSpec
        agents = {
            "Generalist": AgentSpec(
                name="Generalist",
                role="通用分析员",
                goal="分析问题并给出具体、有数值的答案",
                backstory="你是一位通用工程分析员。请用中文回复，给出具体数值和推理过程。",
            ),
        }
        _info("未指定智能体，使用默认通用分析员")

    # ── Config ──
    config = SigmaConfig(project_name=args.project or "sigma-cli")
    if args.output:
        config.output_base_dir = args.output

    # ── Orchestrate ──
    orch = SigmaOrchestrator(
        config=config,
        agents=agents,
        tools=tools,
        skills=skills,
        llm_backend=backend,
        max_rounds=args.max_rounds,
        verbose=not args.quiet,
        interactive=not args.yes,
    )

    output_dir = None
    if args.output:
        output_dir = Path(args.output)

    result = orch.run(
        instruction=args.instruction,
        output_dir=output_dir,
        mode=args.mode,
    )
    return result


def cmd_list(args):
    """List past outputs."""
    from pathlib import Path
    base = Path(args.output or "output")
    if not base.exists():
        print("No output directory found.")
        return

    versions = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0,
    )
    if not versions:
        print("No outputs found.")
        return

    print(f"{'Version':<10} {'Date':<12} {'Instruction':<50}")
    print("-" * 74)
    for v in versions:
        report = v / "REPORT.md"
        result = v / "result.json"
        if report.exists():
            text = report.read_text(encoding="utf-8")[:200]
            # First line after --- header
            lines = text.split("\n")
            desc = ""
            for line in lines:
                if line.startswith("# "):
                    desc = line[2:][:48]
                    break
            mtime = v.stat().st_mtime
            from datetime import datetime
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            print(f"  {v.name:<8} {date_str:<12} {desc:<50}")


def cmd_view(args):
    """View a specific report."""
    from pathlib import Path
    vid = args.version
    if not vid.startswith("v"):
        vid = f"v{vid}"

    base = Path(args.output or "output")
    report_path = base / vid / "REPORT.md"
    result_path = base / vid / "result.json"

    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"))
    elif result_path.exists():
        import json
        data = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        _error(f"Output not found: {vid}")
        sys.exit(1)


def cmd_config(args):
    """Show current configuration."""
    env = _setup_env(require_key=False)
    key = env["api_key"]
    if key:
        masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
    else:
        masked_key = "(not set)"
    print(f"API Key:     {masked_key}")
    print(f"Base URL:    {env['base_url']}")
    print(f"Model:       {env['model']}")
    print(f"Agents Dir:  {os.environ.get('SIGMA_AGENTS_DIR', '(not set)')}")
    print(f"Tools Dir:   {os.environ.get('SIGMA_TOOLS_DIR', '(not set)')}")
    print(f"Skills Dir:  {os.environ.get('SIGMA_SKILLS_DIR', '(not set)')}")


# ── Helpers ──────────────────────────────────────────────────────────

def _info(msg: str):
    print(f"\033[2m[sigma]\033[0m {msg}")


def _error(msg: str):
    print(f"\033[91m[sigma] ERROR: {msg}\033[0m", file=sys.stderr)


def main():
    _reconfigure_stdout()
    parser = argparse.ArgumentParser(
        prog="sigma",
        description="Sigma — Generic Multi-Agent Collaboration Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sigma run "计算150mm铝管质量"               # auto-detect mode
  sigma run "设计火箭——先推进后结构再飞控" -y  # non-interactive
  sigma run "分析KNSB比冲" --mode sigma       # force Sigma AERC mode
  sigma list                                  # list past outputs
  sigma view v11                              # read report v11
  sigma config                                # show current config
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──
    run_parser = sub.add_parser("run", help="Execute a task")
    run_parser.add_argument("instruction", help="Task instruction (natural language)")
    run_parser.add_argument("--mode", "-m", choices=["auto", "sigma", "tau"], default="auto",
                            help="Execution mode (default: auto-detect)")
    run_parser.add_argument("--yes", "-y", action="store_true",
                            help="Non-interactive: auto-continue each round")
    run_parser.add_argument("--quiet", "-q", action="store_true",
                            help="Suppress progress output")
    run_parser.add_argument("--output", "-o", help="Output directory")
    run_parser.add_argument("--project", "-p", help="Project name")
    run_parser.add_argument("--max-rounds", "-r", type=int, default=4,
                            help="Maximum AERC rounds (default: 4)")
    run_parser.add_argument("--agents-dir", help="Directory with agent definitions")
    run_parser.add_argument("--tools-dir", help="Directory with tool definitions")
    run_parser.add_argument("--skills-dir", help="Directory with skill files")

    # ── list ──
    list_parser = sub.add_parser("list", help="List past outputs")
    list_parser.add_argument("--output", "-o", help="Output base directory")

    # ── view ──
    view_parser = sub.add_parser("view", help="View a report")
    view_parser.add_argument("version", help="Version ID (e.g., v11)")
    view_parser.add_argument("--output", "-o", help="Output base directory")

    # ── config ──
    sub.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "view":
        cmd_view(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
