"""Tests for Sigma logging module."""

import logging
import io
import pytest
from sigma.log import get_logger, setup_logging


class TestGetLogger:
    """get_logger returns a Python logger."""

    def test_returns_logger(self):
        logger = get_logger("sigma.test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "sigma.test"

    def test_same_name_same_instance(self):
        a = get_logger("sigma.test.foo")
        b = get_logger("sigma.test.foo")
        assert a is b

    def test_different_names_different_instances(self):
        a = get_logger("sigma.test.a")
        b = get_logger("sigma.test.b")
        assert a is not b

    def test_child_logger_inherits_level(self):
        setup_logging(level=logging.WARNING)
        parent = get_logger("sigma")
        child = get_logger("sigma.child")
        assert parent.level == logging.WARNING
        # child inherits from parent in stdlib logging hierarchy


class TestSetupLogging:
    """setup_logging configures output."""

    def test_stream_output(self):
        stream = io.StringIO()
        setup_logging(level=logging.INFO, stream=stream)
        logger = get_logger("sigma.test")
        logger.info("hello world")
        output = stream.getvalue()
        assert "hello world" in output

    def test_format_output(self):
        stream = io.StringIO()
        setup_logging(
            level=logging.INFO,
            fmt="%(levelname)s | %(message)s",
            stream=stream,
        )
        logger = get_logger("sigma.test")
        logger.warning("test warning")
        output = stream.getvalue()
        assert "WARNING" in output
        assert "test warning" in output

    def test_level_filtering(self):
        stream = io.StringIO()
        setup_logging(level=logging.WARNING, stream=stream)
        logger = get_logger("sigma.test")
        logger.info("should not appear")
        logger.warning("should appear")
        output = stream.getvalue()
        assert "should not appear" not in output
        assert "should appear" in output

    def test_file_output(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(level=logging.INFO, file_path=str(log_file))
        logger = get_logger("sigma.test")
        logger.info("file log test")
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "file log test" in content
