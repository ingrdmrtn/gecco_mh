import json

from gecco.construct_feedback.tool_judge import AngleAnalysis, _parse_verdict_from_text


def test_parse_verdict_valid_structured_json_preserves_fields():
    text = (
        "```json\n"
        '{"per_angle": [{"angle": "fit", "findings": "Looks stable", "supporting_tool_calls": ["tool.a"], "confidence": "high"}], '
        '"key_recommendations": ["Try X"], "synthesized_feedback": "Keep going."}'
        "\n```"
    )

    verdict = _parse_verdict_from_text(text, iteration=3, tool_call_count=7, wall_time=1.5)

    assert verdict.iteration == 3
    assert verdict.tool_call_count == 7
    assert verdict.wall_time_seconds == 1.5
    assert verdict.per_angle == [
        AngleAnalysis(
            angle="fit",
            findings="Looks stable",
            supporting_tool_calls=["tool.a"],
            confidence="high",
        )
    ]
    assert verdict.key_recommendations == ["Try X"]
    assert verdict.synthesized_feedback == "Keep going."


def test_parse_verdict_salvages_string_per_angle_item():
    text = json.dumps(
        {
            "per_angle": ["text finding"],
            "key_recommendations": [],
            "synthesized_feedback": "Use the saved feedback.",
        }
    )

    verdict = _parse_verdict_from_text(text, iteration=1, tool_call_count=2, wall_time=0.5)

    assert len(verdict.per_angle) == 1
    assert verdict.per_angle[0] == AngleAnalysis(
        angle="",
        findings="text finding",
        supporting_tool_calls=[],
        confidence="medium",
    )
    assert verdict.synthesized_feedback == "Use the saved feedback."


def test_parse_verdict_normalizes_minor_schema_defects():
    text = json.dumps(
        {
            "per_angle": [
                {
                    "angle": 42,
                    "findings": 123,
                    "supporting_tool_calls": "call-a",
                    "confidence": "unavailable",
                },
                ["ignored nested container"],
            ],
            "key_recommendations": "Try something else.",
            "synthesized_feedback": 123,
        }
    )

    verdict = _parse_verdict_from_text(text, iteration=4, tool_call_count=5, wall_time=2.25)

    assert verdict.per_angle == [
        AngleAnalysis(
            angle="42",
            findings="123",
            supporting_tool_calls=["call-a"],
            confidence="medium",
        )
    ]
    assert verdict.key_recommendations == ["Try something else."]
    assert verdict.synthesized_feedback == "123"


def test_parse_verdict_falls_back_for_malformed_json():
    text = "```json\n[{\"broken\": true}\n```"

    verdict = _parse_verdict_from_text(text, iteration=9, tool_call_count=11, wall_time=3.0)

    assert verdict.per_angle == []
    assert verdict.key_recommendations == []
    assert verdict.synthesized_feedback == text
