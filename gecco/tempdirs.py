"""Temporary-directory configuration for parallel workers.

Joblib/loky can otherwise default to node-local `/tmp` or `/dev/shm`, which
is often small on shared clusters. Configure these before any parallel worker
pools are created.
"""

from __future__ import annotations

import os
from pathlib import Path


def configure_temp_dirs(project_root: str | Path, *, prefix: str = "GeCCo") -> Path:
    """Set joblib/loky temp directories under the project/run directory.

    Existing user-provided environment variables are respected. Returns the
    joblib temp directory path.
    """

    project_root = Path(project_root)
    tmp_root = Path(os.environ.get("GECCO_TMPDIR", project_root / "tmp"))
    joblib_tmp = Path(os.environ.get("JOBLIB_TEMP_FOLDER", tmp_root / "joblib"))

    tmp_root.mkdir(parents=True, exist_ok=True)
    joblib_tmp.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GECCO_TMPDIR", str(tmp_root))
    os.environ.setdefault("TMPDIR", str(tmp_root))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_tmp))
    os.environ.setdefault("LOKY_TEMP_FOLDER", str(joblib_tmp))

    print(f"[{prefix}] Temp dir: {tmp_root}")
    print(f"[{prefix}] Joblib/loky temp dir: {joblib_tmp}")
    return joblib_tmp
