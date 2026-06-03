"""Tests for the JudgeLesion framework."""

from types import SimpleNamespace

import pytest

from config.schema import JudgeLesionConfig, JudgeConfig
from gecco.construct_feedback.judge_lesion import JudgeLesion


def _make_verdict(feedback, recommendations=None):
    """Create a minimal verdict-like object for testing."""
    return SimpleNamespace(
        synthesized_feedback=feedback,
        key_recommendations=recommendations or [],
    )


def _make_cfg(enabled=True, lesion_type="complete", noise_text="Noise."):
    """Create a JudgeConfig with lesion settings."""
    return JudgeConfig(
        lesion=JudgeLesionConfig(
            enabled=enabled,
            lesion_type=lesion_type,
            noise_text=noise_text,
        )
    )


class TestJudgeLesionDisabled:
    def test_disabled_returns_original(self):
        cfg = _make_cfg(enabled=False)
        lesion = JudgeLesion(cfg)
        original = "Some feedback text"
        verdict = _make_verdict(original)
        result = lesion.apply(verdict, iteration=1, original_feedback=original)
        assert result == original


class TestLesionComplete:
    def test_returns_empty_string(self):
        cfg = _make_cfg(lesion_type="complete")
        lesion = JudgeLesion(cfg)
        verdict = _make_verdict("Detailed feedback here")
        result = lesion.apply(
            verdict, iteration=1, original_feedback="Detailed feedback here"
        )
        assert result == ""


class TestLesionNoise:
    def test_returns_noise_text(self):
        cfg = _make_cfg(lesion_type="noise", noise_text="Try something else.")
        lesion = JudgeLesion(cfg)
        verdict = _make_verdict("Real feedback")
        result = lesion.apply(verdict, iteration=1, original_feedback="Real feedback")
        assert result == "Try something else."

    def test_uses_default_noise_text(self):
        cfg = _make_cfg(lesion_type="noise")
        lesion = JudgeLesion(cfg)
        verdict = _make_verdict("Real feedback")
        result = lesion.apply(verdict, iteration=1, original_feedback="Real feedback")
        assert result == cfg.lesion.noise_text


class TestLesionSummaryOnly:
    def test_strips_recommendations(self):
        cfg = _make_cfg(lesion_type="summary_only")
        lesion = JudgeLesion(cfg)
        feedback = (
            "BIC trajectory: 100 -> 95 -> 90\n"
            "Best model has BIC=90.\n\n"
            "Key Recommendations:\n"
            "- Try simpler models\n"
            "- Check parameter recovery"
        )
        verdict = _make_verdict(
            feedback,
            recommendations=["Try simpler models", "Check parameter recovery"],
        )
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "Key Recommendations" not in result
        assert "Try simpler models" not in result
        assert "BIC trajectory" in result

    def test_keeps_quantitative_lines(self):
        cfg = _make_cfg(lesion_type="summary_only")
        lesion = JudgeLesion(cfg)
        feedback = (
            "BIC improved from 120 to 105.\n"
            "The model shows good fit quality.\n\n"
            "However, I think the approach is interesting and you should explore it further."
        )
        verdict = _make_verdict(feedback)
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "BIC" in result
        assert "fit quality" in result


class TestLesionNoTools:
    def test_removes_tool_dependent_analysis(self):
        cfg = _make_cfg(lesion_type="no_tools")
        lesion = JudgeLesion(cfg)
        feedback = (
            "The BIC trajectory shows improvement.\n"
            "PPC analysis reveals the model underpredicts stay probabilities.\n"
            "Parameter recovery diagnostics indicate good identifiability."
        )
        verdict = _make_verdict(feedback)
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "BIC trajectory" in result
        assert "PPC" not in result
        assert "Parameter recovery" not in result


class TestLesionNoRecommendations:
    def test_removes_recommendation_section(self):
        cfg = _make_cfg(lesion_type="no_recommendations")
        lesion = JudgeLesion(cfg)
        feedback = (
            "The models are improving steadily.\n\n"
            "Key Recommendations:\n"
            "- Focus on parameter identifiability\n"
            "- Simplify the learning rate structure"
        )
        verdict = _make_verdict(
            feedback,
            recommendations=[
                "Focus on parameter identifiability",
                "Simplify the learning rate structure",
            ],
        )
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "Key Recommendations" not in result
        assert "improving steadily" in result

    def test_removes_recommendations_from_verdict(self):
        cfg = _make_cfg(lesion_type="no_recommendations")
        lesion = JudgeLesion(cfg)
        feedback = "Good progress.\n\n- Focus on identifiability\n- Try simpler priors"
        verdict = _make_verdict(
            feedback,
            recommendations=["Focus on identifiability", "Try simpler priors"],
        )
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "Focus on identifiability" not in result


class TestLesionNoCitations:
    def test_replaces_model_references(self):
        cfg = _make_cfg(lesion_type="no_citations")
        lesion = JudgeLesion(cfg)
        feedback = (
            "Model iter_3_run0 showed better BIC than model iter_2_run0. "
            "Following model participant_42, you should try a different approach."
        )
        verdict = _make_verdict(feedback)
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "iter_3_run0" not in result
        assert "participant_42" not in result
        assert "a previous model" in result


class TestLesionNoDiagnostics:
    def test_removes_ppc_sections(self):
        cfg = _make_cfg(lesion_type="no_diagnostics")
        lesion = JudgeLesion(cfg)
        feedback = (
            "The BIC trajectory is improving.\n\n"
            "Predictive model check analysis shows systematic deviations.\n\n"
            "Overall, keep iterating."
        )
        verdict = _make_verdict(feedback)
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "Predictive model check" not in result
        assert "BIC trajectory" in result

    def test_removes_residual_analysis(self):
        cfg = _make_cfg(lesion_type="no_diagnostics")
        lesion = JudgeLesion(cfg)
        feedback = (
            "Best BIC is 85.\n\n"
            "Residual analysis reveals patterns in block 2.\n\n"
            "Continue exploring."
        )
        verdict = _make_verdict(feedback)
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "Residual analysis" not in result
        assert "Best BIC" in result

    def test_removes_parameter_recovery(self):
        cfg = _make_cfg(lesion_type="no_diagnostics")
        lesion = JudgeLesion(cfg)
        feedback = (
            "BIC=90.\n\nParameter recovery shows good identifiability.\n\nNext steps."
        )
        verdict = _make_verdict(feedback)
        result = lesion.apply(verdict, iteration=1, original_feedback=feedback)
        assert "Parameter recovery" not in result


class TestUnknownLesionType:
    def test_raises_value_error(self):
        cfg = _make_cfg(lesion_type="invalid_type")
        lesion = JudgeLesion(cfg)
        verdict = _make_verdict("feedback")
        with pytest.raises(ValueError, match="Unknown lesion type"):
            lesion.apply(verdict, iteration=1, original_feedback="feedback")


class TestLesionIterationTracking:
    def test_updates_iteration(self):
        cfg = _make_cfg(lesion_type="complete")
        lesion = JudgeLesion(cfg)
        verdict = _make_verdict("feedback")
        assert lesion._iteration == 0
        lesion.apply(verdict, iteration=5, original_feedback="feedback")
        assert lesion._iteration == 5
