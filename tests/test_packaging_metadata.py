"""Verify dependency-name parity between pyproject.toml and compatibility requirements files."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_pyproject_dep_names() -> set[str]:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return {_bare_name(dep) for dep in data["project"]["dependencies"]}


def _parse_requirements_names(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.exists():
        return names
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = _bare_name(line)
        names.add(name)
    return names


def _bare_name(spec: str) -> str:
    name = re.split(r"[>=<~!;]", spec, maxsplit=1)[0].strip()
    return _normalize(name)


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def test_pyproject_deps_match_requirements_txt():
    """Every direct runtime dependency in pyproject.toml must appear in requirements.txt and vice versa."""
    pyproject_deps = _parse_pyproject_dep_names()
    requirements_deps = _parse_requirements_names(PROJECT_ROOT / "requirements.txt")

    missing_in_requirements = pyproject_deps - requirements_deps
    extra_in_requirements = requirements_deps - pyproject_deps

    assert not missing_in_requirements, (
        f"Packages in pyproject.toml but missing from requirements.txt: {sorted(missing_in_requirements)}"
    )
    assert not extra_in_requirements, (
        f"Packages in requirements.txt but missing from pyproject.toml: sorted(extra_in_requirements)"
    )


def test_dashboard_extra_matches_dashboard_requirements():
    """The dashboard extra dependencies must match gecco-mh-dashboard/requirements.txt."""  # noqa: E501
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    dashboard_extra = data["project"].get("optional-dependencies", {}).get("dashboard", [])
    extra_names = {_bare_name(dep) for dep in dashboard_extra}

    dash_req_path = PROJECT_ROOT / "gecco-mh-dashboard" / "requirements.txt"
    req_names = _parse_requirements_names(dash_req_path)

    missing_in_req = extra_names - req_names
    extra_in_req = req_names - extra_names

    assert not missing_in_req, (
        f"Dashboard extra packages missing from gecco-mh-dashboard/requirements.txt: {sorted(missing_in_req)}"
    )
    assert not extra_in_req, (
        f"Packages in gecco-mh-dashboard/requirements.txt missing from dashboard extra: {sorted(extra_in_req)}"
    )


def test_pytest_is_in_uv_dev_dependency_group():
    """Pytest should be installed from uv's dev dependency group."""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    dev_group = data.get("dependency-groups", {}).get("dev", [])
    dev_names = {_bare_name(dep) for dep in dev_group}

    assert "pytest" in dev_names
