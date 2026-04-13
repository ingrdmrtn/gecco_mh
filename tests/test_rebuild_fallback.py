import pytest
import json
import orjson
from pathlib import Path
from gecco.diagnostic_store.rebuild import _load_json_file


def test_load_json_file_fallback(tmp_path):
    # JSON with non-standard Infinity
    json_file = tmp_path / "iter1_run0.json"
    content = (
        '[{"function_name": "model1", "ppc": [1.0, 2.0], "metric_value": Infinity}]'
    )
    json_file.write_text(content, encoding="utf-8")

    # Arguments: (json_file_path, iteration, client_id, tag)
    args = (json_file, 1, None, "")

    result = _load_json_file(args)

    assert result is not None
    assert result["iteration"] == 1
    assert result["iteration_results"][0]["metric_value"] == float("inf")
    assert result["ppc_results_map"]["model1"] == [1.0, 2.0]
