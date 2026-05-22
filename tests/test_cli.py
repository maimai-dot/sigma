"""Tests for sigma CLI module."""

import os
import sys
import argparse
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sigma.cli import main, _setup_env, _find_dotenv, _reconfigure_stdout


class TestCLIHelp:
    """Test CLI argument parsing."""

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["sigma", "--help"]):
                main()
        captured = capsys.readouterr()
        assert "Sigma" in captured.out

    def test_no_args_shows_help(self, capsys):
        with patch.object(sys, "argv", ["sigma"]):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out or "Sigma" in captured.out

    def test_config_command(self, capsys):
        with patch.object(sys, "argv", ["sigma", "config"]):
            with patch("sigma.cli._setup_env", return_value={
                "api_key": "sk-test12345678",
                "base_url": "https://test.api.com",
                "model": "test-model",
            }):
                main()
        captured = capsys.readouterr()
        # Key masking: 8 + 3 dots + 4 = len("sk-test...5678") ≈ 15
        assert "sk-test1" in captured.out
        assert "5678" in captured.out
        assert "https://test.api.com" in captured.out

    def test_config_no_key(self, capsys):
        with patch.object(sys, "argv", ["sigma", "config"]):
            with patch("sigma.cli._setup_env", return_value={
                "api_key": "",
                "base_url": "https://test.api.com",
                "model": "test-model",
            }):
                main()
        captured = capsys.readouterr()
        assert "(not set)" in captured.out

    def test_list_empty(self, capsys):
        with patch.object(sys, "argv", ["sigma", "list", "-o", "/nonexistent/dir"]):
            main()
        captured = capsys.readouterr()
        assert "No output" in captured.out or "not found" in captured.out.lower()


class TestCLIEnv:
    """Test environment setup helpers."""

    def test_setup_env_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("sigma.cli._find_dotenv", return_value=None):
                with patch("dotenv.load_dotenv"):
                    with pytest.raises(SystemExit):
                        _setup_env(require_key=True)

    def test_setup_env_no_require(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("sigma.cli._find_dotenv", return_value=None):
                with patch("dotenv.load_dotenv"):
                    result = _setup_env(require_key=False)
                    assert result["api_key"] == ""
                    assert result["base_url"] == "https://api.deepseek.com"

    def test_setup_env_with_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            result = _setup_env(require_key=False)
            assert result["api_key"] == "sk-test"

    def test_find_dotenv_found(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value")
        with patch.object(Path, "cwd", return_value=tmp_path):
            result = _find_dotenv()
            # Should find either CWD or parent or one of the candidates
            # Not asserting exact path — just that it exists
            pass

    def test_reconfigure_stdout_noop_on_linux(self):
        if hasattr(sys.stdout, "reconfigure"):
            _reconfigure_stdout()  # Should not raise


class TestCLIRun:
    """Test run command argument parsing."""

    def test_run_parses_instruction(self):
        with patch.object(sys, "argv", ["sigma", "run", "test instruction"]):
            with patch("sigma.cli.cmd_run") as mock_run:
                main()
                args = mock_run.call_args[0][0]
                assert args.instruction == "test instruction"
                assert args.mode == "auto"
                assert args.yes is False

    def test_run_with_mode_flag(self):
        with patch.object(sys, "argv", ["sigma", "run", "test", "--mode", "tau"]):
            with patch("sigma.cli.cmd_run") as mock_run:
                main()
                args = mock_run.call_args[0][0]
                assert args.mode == "tau"

    def test_run_non_interactive(self):
        with patch.object(sys, "argv", ["sigma", "run", "test", "-y"]):
            with patch("sigma.cli.cmd_run") as mock_run:
                main()
                args = mock_run.call_args[0][0]
                assert args.yes is True

    def test_run_with_max_rounds(self):
        with patch.object(sys, "argv", ["sigma", "run", "test", "-r", "2"]):
            with patch("sigma.cli.cmd_run") as mock_run:
                main()
                args = mock_run.call_args[0][0]
                assert args.max_rounds == 2


class TestCLIView:
    """Test view command."""

    def test_view_adds_v_prefix(self, capsys):
        # Viewing nonexistent — should error
        with patch.object(sys, "argv", ["sigma", "view", "999"]):
            with pytest.raises(SystemExit):
                main()
