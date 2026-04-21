from src.interface.helpers import (
    _build_schema_from_pairs,
    _default_field_question,
    _flatten_nested_field_map,
    _has_nested_field,
    _merge_display_fields,
    _normalize_flat_field_map,
    _parse_conversation_text,
    _set_nested_field,
)


def test_parse_conversation_text_adds_monotonic_turn_suffixes():
    parsed = _parse_conversation_text("Doctor: Hello\nPatient: Hi")

    assert list(parsed.keys()) == ["Doctor 000001", "Patient 000002"]
    assert parsed["Doctor 000001"] == "Hello"
    assert parsed["Patient 000002"] == "Hi"


def test_parse_conversation_text_appends_continuation_lines_to_current_turn():
    parsed = _parse_conversation_text("Doctor: First line\nsecond line")

    assert parsed == {"Doctor 000001": "First line\nsecond line"}


def test_default_field_question_generates_human_question():
    assert _default_field_question("customer_name") == "What is the customer name?"


def test_default_field_question_preserves_question_like_prefix():
    assert _default_field_question("when was admitted") == "when was admitted?"


def test_build_schema_from_pairs_autogenerates_when_value_missing():
    schema = _build_schema_from_pairs(
        ["customer_name", "", "visit_date"],
        ["", "ignored", "When did patient arrive?"],
        autogenerate_question=True,
    )

    assert schema == {
        "customer_name": "What is the customer name?",
        "visit_date": "When did patient arrive?",
    }


def test_set_and_has_nested_field_work_for_dotted_keys():
    target = {}
    _set_nested_field(target, "patient.address.city", "Boston")

    assert target == {"patient": {"address": {"city": "Boston"}}}
    assert _has_nested_field(target, "patient.address.city") is True
    assert _has_nested_field(target, "patient.address.zip") is False


def test_normalize_flat_field_map_trims_and_skips_invalid_keys():
    payload = {"  a.b  ": "  x  ", "": "bad", 99: "bad", "c": None}
    normalized = _normalize_flat_field_map(payload)
    assert normalized == {"a.b": "x", "c": ""}


def test_flatten_nested_field_map_handles_none_and_strips_values():
    source = {"a": {"b": "  y "}, "c": None}
    flattened = _flatten_nested_field_map(source)
    assert flattened == {"a.b": "y", "c": ""}


def test_merge_display_fields_applies_accepted_new_fields_as_nested_values():
    merged = _merge_display_fields(
        {"existing": "value"},
        {"new_field.sub": "  fresh  ", "blank": None},
    )

    assert merged["existing"] == "value"
    assert merged["new_field"]["sub"] == "fresh"
    assert merged["blank"] == ""
