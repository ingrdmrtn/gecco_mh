"""CLI dotenv loading contract tests."""
import os
from unittest.mock import patch

SENTRY_DSN = "SENTRY_DSN"


def test_cli_loads_project_dotenv_before_handler(tmp_path, monkeypatch):
    """CLI loads .env before dispatching to the command handler."""
    monkeypatch.delenv(SENTRY_DSN, raising=False)
    (tmp_path / ".env").write_text("SENTRY_DSN=https://example.invalid/1\n")

    from gecco.cli import main

    monkeypatch.setattr("gecco.cli.PROJECT_ROOT", tmp_path)

    def spy(**kwargs):
        assert os.environ["SENTRY_DSN"] == "https://example.invalid/1"

    with patch("gecco.cli.run_local_client.run_local_client", side_effect=spy):
        main(["run", "local-client", "--config", "dummy.yaml"])


def test_cli_dotenv_does_not_override_existing_environment(tmp_path, monkeypatch):
    """Existing env var wins over .env when override=False."""
    monkeypatch.delenv(SENTRY_DSN, raising=False)
    monkeypatch.setenv(SENTRY_DSN, "https://original.invalid/1")
    (tmp_path / ".env").write_text("SENTRY_DSN=https://example.invalid/1\n")

    from gecco.cli import main

    monkeypatch.setattr("gecco.cli.PROJECT_ROOT", tmp_path)

    def spy(**kwargs):
        assert os.environ["SENTRY_DSN"] == "https://original.invalid/1"

    with patch("gecco.cli.run_local_client.run_local_client", side_effect=spy):
        main(["run", "local-client", "--config", "dummy.yaml"])


def test_cli_missing_dotenv_is_allowed(tmp_path, monkeypatch):
    """CLI startup succeeds with no .env file present."""
    monkeypatch.delenv(SENTRY_DSN, raising=False)

    from gecco.cli import main

    monkeypatch.setattr("gecco.cli.PROJECT_ROOT", tmp_path)

    def spy(**kwargs):
        assert os.environ.get(SENTRY_DSN) is None

    with patch("gecco.cli.run_local_client.run_local_client", side_effect=spy):
        main(["run", "local-client", "--config", "dummy.yaml"])
