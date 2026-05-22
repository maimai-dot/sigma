"""Tests for multimodal content types and helpers in sigma.llm."""

import pytest
import os
import tempfile
import base64

from sigma.llm import (
    build_message, text_content, image_url, encode_image_base64,
    is_multimodal, ContentPart, ChatMessage, TextContent, ImageUrlContent,
)


# ── TextContent ──────────────────────────────────────────────────────

class TestTextContent:
    def test_basic(self):
        tc = text_content("What is this?")
        assert tc == {"type": "text", "text": "What is this?"}
        assert tc["type"] == "text"

    def test_empty_text(self):
        tc = text_content("")
        assert tc["text"] == ""


# ── ImageUrlContent ──────────────────────────────────────────────────

class TestImageUrlContent:
    def test_basic(self):
        iu = image_url("https://example.com/photo.jpg")
        assert iu["type"] == "image_url"
        assert iu["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_default_detail(self):
        iu = image_url("https://example.com/photo.jpg")
        assert iu["image_url"]["detail"] == "auto"

    def test_high_detail(self):
        iu = image_url("https://example.com/photo.jpg", detail="high")
        assert iu["image_url"]["detail"] == "high"

    def test_low_detail(self):
        iu = image_url("https://example.com/photo.jpg", detail="low")
        assert iu["image_url"]["detail"] == "low"

    def test_base64_data_uri(self):
        uri = "data:image/png;base64,iVBORw0KGgo="
        iu = image_url(uri)
        assert iu["image_url"]["url"] == uri


# ── build_message ────────────────────────────────────────────────────

class TestBuildMessage:
    def test_text_only(self):
        msg = build_message("user", "Hello")
        assert msg == {"role": "user", "content": "Hello"}

    def test_system_role(self):
        msg = build_message("system", "You are helpful")
        assert msg["role"] == "system"
        assert msg["content"] == "You are helpful"

    def test_assistant_role(self):
        msg = build_message("assistant", "I can help")
        assert msg == {"role": "assistant", "content": "I can help"}

    def test_multimodal_content(self):
        parts = [
            text_content("Describe this image:"),
            image_url("https://example.com/img.png"),
        ]
        msg = build_message("user", parts)
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image_url"


# ── encode_image_base64 ─────────────────────────────────────────────

class TestEncodeImageBase64:
    def test_encode_png(self):
        # Create a minimal valid PNG
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            # Minimal PNG: 1x1 red pixel
            png_bytes = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PUwNgQAAAABJRU5ErkJggg=="
            )
            tmp.write(png_bytes)
            tmp.close()
            result = encode_image_base64(tmp.name)
            assert result.startswith("data:image/png;base64,")
        finally:
            os.unlink(tmp.name)

    def test_encode_jpg(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            tmp.write(b"fake jpeg content")
            tmp.close()
            result = encode_image_base64(tmp.name)
            assert result.startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(tmp.name)

    def test_encode_jpeg_extension(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
        try:
            tmp.write(b"fake jpeg content")
            tmp.close()
            result = encode_image_base64(tmp.name)
            assert result.startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(tmp.name)

    def test_encode_gif(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        try:
            tmp.write(b"GIF89a fake")
            tmp.close()
            result = encode_image_base64(tmp.name)
            assert result.startswith("data:image/gif;base64,")
        finally:
            os.unlink(tmp.name)

    def test_encode_webp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
        try:
            tmp.write(b"RIFF fake webp")
            tmp.close()
            result = encode_image_base64(tmp.name)
            assert result.startswith("data:image/webp;base64,")
        finally:
            os.unlink(tmp.name)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            encode_image_base64("/nonexistent/image.png")

    def test_unsupported_extension(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
        try:
            tmp.write(b"fake bitmap")
            tmp.close()
            with pytest.raises(ValueError, match="Unsupported image format"):
                encode_image_base64(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_unsupported_txt(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        try:
            tmp.write(b"not an image")
            tmp.close()
            with pytest.raises(ValueError, match="Unsupported image format"):
                encode_image_base64(tmp.name)
        finally:
            os.unlink(tmp.name)


# ── is_multimodal ────────────────────────────────────────────────────

class TestIsMultimodal:
    def test_string_is_not_multimodal(self):
        assert is_multimodal("hello") is False

    def test_empty_list_is_multimodal(self):
        # Empty list is technically valid multimodal (no parts)
        assert is_multimodal([]) is True

    def test_text_parts_list_is_multimodal(self):
        assert is_multimodal([text_content("hi")]) is True

    def test_mixed_parts_list_is_multimodal(self):
        parts = [text_content("hi"), image_url("https://ex.com/i.png")]
        assert is_multimodal(parts) is True

    def test_non_dict_items_not_multimodal(self):
        assert is_multimodal(["not a dict"]) is False

    def test_dict_without_type_not_multimodal(self):
        assert is_multimodal([{"text": "no type field"}]) is False

    def test_int_not_multimodal(self):
        assert is_multimodal(42) is False

    def test_none_not_multimodal(self):
        assert is_multimodal(None) is False


# ── TypedDict conformance ────────────────────────────────────────────

class TestTypedDictConformance:
    def test_text_content_is_typed_dict(self):
        tc = text_content("hi")
        assert isinstance(tc, dict)
        assert tc["type"] == "text"
        assert tc["text"] == "hi"

    def test_image_url_content_is_typed_dict(self):
        iu = image_url("https://ex.com/img.png", detail="high")
        assert isinstance(iu, dict)
        assert iu["type"] == "image_url"
        assert iu["image_url"]["url"] == "https://ex.com/img.png"
        assert iu["image_url"]["detail"] == "high"


# ── Integration: real-world multimodal message ───────────────────────

class TestMultimodalIntegration:
    def test_build_vision_message(self):
        """Build a message that a Vision LLM (GPT-4V, GLM-4V, Qwen-VL) would accept."""
        msg = build_message("user", [
            text_content("Analyze this engineering diagram:"),
            image_url("https://cdn.example.com/rocket_diagram.png", detail="high"),
        ])
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Analyze this engineering diagram:"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example.com/rocket_diagram.png", "detail": "high"},
        }

    def test_multiple_images(self):
        """Multiple images in one message (some VL models support this)."""
        msg = build_message("user", [
            text_content("Compare these two CAD screenshots:"),
            image_url("https://cdn.example.com/cad_view1.png"),
            image_url("https://cdn.example.com/cad_view2.png"),
        ])
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"

    def test_multiple_images_different_details(self):
        """Images with different resolution tiers."""
        msg = build_message("user", [
            image_url("https://cdn.example.com/overview.png", detail="low"),
            image_url("https://cdn.example.com/detail.png", detail="high"),
        ])
        content = msg["content"]
        assert content[0]["image_url"]["detail"] == "low"
        assert content[1]["image_url"]["detail"] == "high"
