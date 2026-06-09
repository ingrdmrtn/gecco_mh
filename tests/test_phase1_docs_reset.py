"""Documentation checks for Phase 1 cleanup work."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DOCS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "centralised_judge_implementation.md",
    *sorted((PROJECT_ROOT / "config").glob("*.yaml")),
]
STALE_ENTRYPOINTS = [
    "scripts/launch_distributed.py",
    "scripts/run_gecco_distributed.py",
    "scripts/run_judge_orchestrator.py",
    "scripts/reset_distributed.py",
    "scripts/monitor_distributed.py",
    "scripts/launch_cmg_distributed.py",
    "scripts/two_step_psychiatry_group.py",
    "bash/run_gecco_distributed.sh",
]
SCAN_SUFFIXES = {".md", ".yaml"}
EXCLUDED_ROOTS = {
    PROJECT_ROOT / ".git",
    PROJECT_ROOT / ".serena",
    PROJECT_ROOT / "docs" / "codebase_cleanup_plan",
    PROJECT_ROOT / "plans",
    PROJECT_ROOT / "tests" / "fixtures",
}
PYTEST_SCRIPT_PATTERNS = ("test_*.py", "*_test.py")


def _iter_broad_scan_paths():
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(excluded == path or excluded in path.parents for excluded in EXCLUDED_ROOTS):
            continue
        yield path


def _iter_pytest_collectable_scripts():
    scripts_dir = PROJECT_ROOT / "scripts"
    seen_paths = set()
    for pattern in PYTEST_SCRIPT_PATTERNS:
        for path in scripts_dir.glob(pattern):
            if path not in seen_paths:
                seen_paths.add(path)
                yield path


def _is_network_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return (func.value.id, func.attr) in {
            ("requests", "get"),
            ("requests", "post"),
            ("requests", "put"),
            ("requests", "delete"),
            ("requests", "request"),
        }
    return False


def test_active_docs_do_not_reference_removed_script_entrypoints():
    """Ensure active docs point at the unified CLI surface instead."""
    for doc_path in ACTIVE_DOCS:
        content = doc_path.read_text(encoding="utf-8")
        for stale_entrypoint in STALE_ENTRYPOINTS:
            assert stale_entrypoint not in content, (
                f"{doc_path.relative_to(PROJECT_ROOT)} still references "
                f"{stale_entrypoint}"
            )


def test_project_docs_and_configs_do_not_reference_stale_entrypoints():
    """Broad-scan active docs and config files for removed entrypoints."""
    for file_path in _iter_broad_scan_paths():
        content = file_path.read_text(encoding="utf-8")
        for stale_entrypoint in STALE_ENTRYPOINTS:
            assert stale_entrypoint not in content, (
                f"{file_path.relative_to(PROJECT_ROOT)} still references "
                f"{stale_entrypoint}"
            )


def test_pytest_collectable_scripts_do_not_make_top_level_network_calls():
    """Collectable scripts must keep network calls out of import-time module scope."""
    for script_path in _iter_pytest_collectable_scripts():
        module = ast.parse(script_path.read_text(encoding="utf-8"))
        top_level_network_calls = []

        for statement in module.body:
            nodes = ast.walk(statement)
            if any(_is_network_call(node) for node in nodes):
                top_level_network_calls.append(type(statement).__name__)

        assert top_level_network_calls == []
