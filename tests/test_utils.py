"""Utility contract tests."""

from types import SimpleNamespace

from gecco.utils import mapping_get


def test_mapping_get_handles_none_dict_and_attribute_containers():
    """mapping_get should read from None, dicts, and attribute containers."""
    assert mapping_get(None, "missing", "fallback") == "fallback"
    assert mapping_get({"present": 3}, "present") == 3

    container = SimpleNamespace(present="value")
    assert mapping_get(container, "present") == "value"
    assert mapping_get(container, "missing", "fallback") == "fallback"
