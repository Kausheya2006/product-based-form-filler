import pytest

from src.domain.domain import Conversation, ConversationVersion
from src.domain.speakers import (
    normalize_history_value,
    render_history_for_model,
    strip_speaker_timestamp_suffix,
)


def test_strip_speaker_timestamp_suffix_removes_numeric_suffix():
    assert strip_speaker_timestamp_suffix("Doctor 000123") == "Doctor"


def test_strip_speaker_timestamp_suffix_keeps_plain_speaker():
    assert strip_speaker_timestamp_suffix("Nurse") == "Nurse"


def test_normalize_history_value_list_joins_non_blank_items():
    value = ["hello", "   ", "world", 42]
    assert normalize_history_value(value) == "hello\nworld\n42"


def test_conversation_version_normalizes_history_values_to_strings():
    version = ConversationVersion(
        version_index=0,
        history={"Speaker 1": ["line 1", "line 2"], "Speaker 2": None},
    )

    assert version.history["Speaker 1"] == "line 1\nline 2"
    assert version.history["Speaker 2"] == ""


def test_conversation_latest_history_uses_highest_version_index():
    convo = Conversation(
        conversation_id="c1",
        form_id="f1",
        conversation_name="",
        versions=[
            ConversationVersion(version_index=2, history={"A 000002": "new"}),
            ConversationVersion(version_index=1, history={"A 000001": "old"}),
        ],
    )

    assert convo.latest_history == {"A 000002": "new"}


def test_conversation_full_text_renders_with_clean_speaker_names():
    convo = Conversation(
        conversation_id="c2",
        form_id="f2",
        conversation_name="",
        versions=[
            ConversationVersion(
                version_index=0,
                history={"Doctor 000001": "patient stable", "Nurse 000002": "noted"},
            )
        ],
    )

    assert convo.full_text == "Doctor: patient stable\nNurse: noted"


def test_render_history_for_model_handles_mixed_value_types():
    history = {"Agent 000111": ["Hi", "there"], "User 000112": "OK"}
    rendered = render_history_for_model(history)
    assert rendered == "Agent: Hi\nthere\nUser: OK"
