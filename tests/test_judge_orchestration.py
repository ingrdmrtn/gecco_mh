"""
Integration tests for centralized judge orchestration.

Tests the barrier synchronization, shared feedback, and fallback behavior.
"""

import json
import tempfile
import time
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from gecco.coordination import SharedRegistry


class TestBarrierPrimitives:
    """Test SharedRegistry barrier and judge feedback methods."""

    def test_count_clients_at_iteration(self):
        """Test counting distinct clients at an iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SharedRegistry(str(Path(tmpdir) / "registry.json"))

            # Add results from 2 different clients
            registry.update(client_id=0, iteration=0, results=[{"function_name": "model1"}])
            registry.update(client_id=1, iteration=0, results=[{"function_name": "model2"}])

            # Should count 2 unique clients
            count = registry.count_clients_at_iteration(iteration=0)
            assert count == 2

            # Add another iteration with 1 client
            registry.update(client_id=0, iteration=1, results=[{"function_name": "model3"}])
            count = registry.count_clients_at_iteration(iteration=1)
            assert count == 1

    def test_wait_for_iteration_immediate(self):
        """Test wait_for_iteration returns immediately when clients are ready."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SharedRegistry(str(Path(tmpdir) / "registry.json"))

            # Add results from 2 clients before waiting
            registry.update(client_id=0, iteration=0, results=[{"function_name": "model1"}])
            registry.update(client_id=1, iteration=0, results=[{"function_name": "model2"}])

            # Should return immediately with count=2
            start = time.time()
            count = registry.wait_for_iteration(
                iteration=0, n_expected=2, timeout_seconds=5.0, poll_seconds=0.1
            )
            elapsed = time.time() - start

            assert count == 2
            assert elapsed < 1.0  # Should be fast

    def test_wait_for_iteration_timeout(self):
        """Test wait_for_iteration times out if clients don't arrive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SharedRegistry(str(Path(tmpdir) / "registry.json"))

            # Add only 1 client but expect 2
            registry.update(client_id=0, iteration=0, results=[{"function_name": "model1"}])

            # Should timeout after ~1 second and return count=1
            start = time.time()
            count = registry.wait_for_iteration(
                iteration=0, n_expected=2, timeout_seconds=1.0, poll_seconds=0.1
            )
            elapsed = time.time() - start

            assert count == 1
            assert 0.9 < elapsed < 1.5  # Should wait roughly 1 second

    def test_set_and_get_judge_feedback(self):
        """Test storing and retrieving shared judge feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SharedRegistry(str(Path(tmpdir) / "registry.json"))

            # Store judge feedback
            feedback_text = "Consider adding a learning rate parameter."
            verdict = {"model_count": 5, "best_bic": 123.45}

            registry.set_judge_feedback(
                iteration=0,
                synthesized_feedback=feedback_text,
                verdict_payload=verdict,
            )

            # Retrieve it
            result = registry.get_judge_feedback(iteration=0)
            assert result is not None
            assert result["synthesized_feedback"] == feedback_text
            assert result["verdict"]["model_count"] == 5

    def test_wait_for_judge_feedback_immediate(self):
        """Test wait_for_judge_feedback returns immediately when ready."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SharedRegistry(str(Path(tmpdir) / "registry.json"))

            # Store feedback before waiting
            feedback_text = "Feedback for iteration 0"
            registry.set_judge_feedback(
                iteration=0,
                synthesized_feedback=feedback_text,
                verdict_payload={},
            )

            # Should return immediately
            start = time.time()
            result = registry.wait_for_judge_feedback(
                iteration=0, timeout_seconds=5.0, poll_seconds=0.1
            )
            elapsed = time.time() - start

            assert result is not None
            assert result["synthesized_feedback"] == feedback_text
            assert elapsed < 1.0

    def test_wait_for_judge_feedback_timeout(self):
        """Test wait_for_judge_feedback times out if feedback not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SharedRegistry(str(Path(tmpdir) / "registry.json"))

            # Don't store feedback; just wait for it
            start = time.time()
            result = registry.wait_for_judge_feedback(
                iteration=99, timeout_seconds=1.0, poll_seconds=0.1
            )
            elapsed = time.time() - start

            assert result is None
            assert 0.9 < elapsed < 1.5

    def test_judge_iterations_in_registry(self):
        """Test that judge_iterations section is properly stored in registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SharedRegistry(str(Path(tmpdir) / "registry.json"))

            # Add some client results
            registry.update(client_id=0, iteration=0, results=[])

            # Add judge feedback
            registry.set_judge_feedback(
                iteration=0,
                synthesized_feedback="Test feedback",
                verdict_payload={"test": "data"},
            )

            # Read raw registry to verify structure
            data = registry.read()
            assert "judge_iterations" in data
            assert "0" in data["judge_iterations"]  # Keyed by iteration string
            assert data["judge_iterations"]["0"]["synthesized_feedback"] == "Test feedback"


class TestOrchestratorIntegration:
    """Test the orchestrator logic without full distributed setup."""

    def test_config_with_orchestration(self):
        """Test loading a config with orchestration enabled."""
        from config.schema import load_config

        # Create test config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            config_path.write_text("""
task:
  name: "test_task"
  description: "Test"
  goal: "Test goal"
  instructions: "Test"

loop:
  max_iterations: 2
  n_clients: 2

judge:
  mode: "tool_using"
  orchestrated: true
  barrier:
    orchestrator_wait_seconds: 120
    client_wait_seconds: 120
""")

            cfg = load_config(str(config_path))
            assert cfg.loop.n_clients == 2
            assert cfg.judge.orchestrated is True
            assert cfg.judge.barrier.orchestrator_wait_seconds == 120

    def test_orchestrator_detects_orchestration(self):
        """Test that orchestrator properly detects when it should be enabled."""
        # This is a simple logic test
        cfg_raw = {
            "judge": {"orchestrated": True},
            "loop": {"n_clients": 2},
        }

        # Simulate launcher logic
        judge_cfg = cfg_raw.get("judge", {})
        orchestrated = judge_cfg.get("orchestrated", False)
        loop_cfg = cfg_raw.get("loop", {})
        n_clients = loop_cfg.get("n_clients")

        assert orchestrated is True
        assert n_clients == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
