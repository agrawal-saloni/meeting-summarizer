"""Smoke tests for input processing."""

from pathlib import Path

import pytest

from src.input_processing import detect_input_type


def test_detect_input_type():
    assert detect_input_type(Path("x.mp4")) == "video"
    assert detect_input_type(Path("x.WAV")) == "audio"
    assert detect_input_type(Path("x.srt")) == "transcript"
    with pytest.raises(ValueError):
        detect_input_type(Path("x.xyz"))
