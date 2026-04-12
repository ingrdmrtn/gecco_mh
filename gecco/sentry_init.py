import os
import warnings
from typing import Optional

import sentry_sdk


ALLOWLIST_FIELDS = {"iteration", "run", "model_name", "task_name", "client_id", "config_name"}


def _scrub_event(event, hint):
    """Strip sensitive data from Sentry events while preserving debugging context.

    Sensitive data includes:
    - LLM prompts and responses (may contain task descriptions, participant text)
    - Generated model code (could embed participant data in variable names)
    - Raw participant/behavioral data in exception messages or local variables

    Strategy:
    - Strip all local variables from exception frames
    - Redact string values >500 chars in extra/breadcrumb data
    - Keep exception type, message (truncated), stack trace file/line info
    - Preserve specific structured fields: iteration, run, model_name, task_name, client_id
    """
    if event.get("tags"):
        for field in list(event["tags"].keys()):
            if field not in ALLOWLIST_FIELDS:
                del event["tags"][field]

    extra = event.get("extra", {})
    scrubbed_extra = {}
    for key, value in extra.items():
        if isinstance(value, str) and len(value) > 500:
            scrubbed_extra[key] = f"<redacted {len(value)} chars>"
        else:
            scrubbed_extra[key] = value
    event["extra"] = scrubbed_extra

    breadcrumbs_container = event.get("breadcrumbs", {})
    for breadcrumb in breadcrumbs_container.get("values", []):
        if "data" in breadcrumb:
            scrubbed_data = {}
            for key, value in breadcrumb["data"].items():
                if isinstance(value, str) and len(value) > 500:
                    scrubbed_data[key] = f"<redacted {len(value)} chars>"
                else:
                    scrubbed_data[key] = value
            breadcrumb["data"] = scrubbed_data

    if "threads" in event:
        for thread in event["threads"].get("values", []):
            if "stacktrace" in thread:
                _scrub_stacktrace(thread["stacktrace"])
            if "vars" in thread:
                thread["vars"] = {}

    for exception in event.get("exception", {}).get("values", []):
        if "stacktrace" in exception:
            _scrub_stacktrace(exception["stacktrace"])
        if "mechanism" in exception and "source" in exception["mechanism"]:
            mechanism_source = exception["mechanism"]["source"]
            if isinstance(mechanism_source, str) and len(mechanism_source) > 500:
                exception["mechanism"]["source"] = mechanism_source[:500] + "..."

    event_length = len(str(event))
    original_length = event.get("_original_length", event_length)
    if original_length > 0:
        scrub_ratio = event_length / original_length
        if scrub_ratio < 0.2:
            warnings.warn(
                f"Sentry scrubbing removed >80% of event data. "
                f"Original estimate: {original_length}, after: {event_length}. "
                f"This may indicate the event is too noisy to be useful."
            )

    return event


def _scrub_stacktrace(stacktrace):
    """Remove local variables from a stacktrace's frames."""
    if "frames" in stacktrace:
        for frame in stacktrace["frames"]:
            if "vars" in frame:
                frame["vars"] = {}
            if "co_locals" in frame:
                frame["co_locals"] = {}


def _before_send(event, hint):
    """Public before_send callback that wraps _scrub_event."""
    import copy

    event_copy = copy.deepcopy(event)
    event_copy["_original_length"] = len(str(event))
    return _scrub_event(event_copy, hint)


def init_sentry(
    cfg: Optional[object] = None,
    task_name: Optional[str] = None,
    client_id: Optional[str] = None,
    config_name: Optional[str] = None,
    **tags,
) -> None:
    """Initialize Sentry SDK for error monitoring and distributed tracing.

    Parameters
    ----------
    cfg : optional
        Configuration object with optional .sentry attribute containing:
        - environment: str = "development"
        - traces_sample_rate: float = 0.1
        - profiles_sample_rate: float = 0.0
        - release: Optional[str] = None
    task_name : optional
        Task name for Sentry tag.
    client_id : optional
        Client ID for distributed runs.
    config_name : optional
        Config name for Sentry tag.
    **tags
        Additional tags to set on Sentry events.

    Returns
    -------
    None
        If SENTRY_DSN is not set, this function is a no-op.
    """
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return

    sentry_cfg = getattr(cfg, "sentry", None) if cfg else None

    environment = (
        getattr(sentry_cfg, "environment", "development")
        if sentry_cfg
        else "development"
    )
    traces_sample_rate = (
        getattr(sentry_cfg, "traces_sample_rate", 0.1) if sentry_cfg else 0.1
    )
    profiles_sample_rate = (
        getattr(sentry_cfg, "profiles_sample_rate", 0.0) if sentry_cfg else 0.0
    )
    release = getattr(sentry_cfg, "release", None) if sentry_cfg else None

    if not release:
        from gecco import __version__

        release = __version__

    sentry_tags = {}
    if task_name:
        sentry_tags["task_name"] = task_name
    if client_id:
        sentry_tags["client_id"] = str(client_id)
    if config_name:
        sentry_tags["config_name"] = config_name
    sentry_tags.update(tags)

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        release=release,
        before_send=_before_send,
    )

    for key, value in sentry_tags.items():
        sentry_sdk.set_tag(key, value)


def capture_fit_error(
    iteration: int,
    model_name: str,
    error: Exception,
    **extra,
) -> None:
    """Report a FIT_ERROR exception with custom fingerprinting.

    Fingerprint groups related errors together regardless of variable
    content in exception messages.

    Parameters
    ----------
    iteration : int
        Current iteration number.
    model_name : str
        Name of the model that failed.
    error : Exception
        The exception that was caught.
    **extra
        Additional context to attach to the event.
    """
    sentry_sdk.capture_exception(
        error,
        fingerprint=["fit-error", str(iteration), model_name],
        extras={"iteration": iteration, "model_name": model_name, **extra},
    )


def capture_recovery_failed(
    iteration: int,
    model_name: str,
    error: Exception,
    **extra,
) -> None:
    """Report a RECOVERY_FAILED exception with custom fingerprinting.

    Parameters
    ----------
    iteration : int
        Current iteration number.
    model_name : str
        Name of the model that failed.
    error : Exception
        The exception that was caught.
    **extra
        Additional context to attach to the event.
    """
    sentry_sdk.capture_exception(
        error,
        fingerprint=["recovery-failed", str(iteration), model_name],
        extras={"iteration": iteration, "model_name": model_name, **extra},
    )


def capture_coordination_error(
    error: Exception,
    operation: str,
    **extra,
) -> None:
    """Report a SharedRegistry coordination error.

    Parameters
    ----------
    error : Exception
        The exception that was caught.
    operation : str
        Name of the coordination operation that failed.
    **extra
        Additional context to attach to the event.
    """
    sentry_sdk.capture_exception(
        error,
        fingerprint=["coordination-error", operation],
        extras={"operation": operation, **extra},
    )
