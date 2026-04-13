import json
import math
import numpy as np
import pytest
from gecco.run_gecco import _NumpyJSONEncoder


def test_numpy_json_encoder():
    encoder = _NumpyJSONEncoder()

    # Test non-finite floats
    assert encoder._sanitize(float("inf")) is None
    assert encoder._sanitize(float("-inf")) is None
    assert encoder._sanitize(float("nan")) is None

    # Test NumPy non-finite values
    assert encoder._sanitize(np.inf) is None
    assert encoder._sanitize(np.nan) is None

    # Test normal values
    assert encoder._sanitize(1.23) == 1.23
    assert encoder._sanitize("test") == "test"

    # Test nested structures
    data = {"a": float("inf"), "b": [1.0, float("nan")], "c": {"d": np.inf}}
    expected = {"a": None, "b": [1.0, None], "c": {"d": None}}
    assert encoder._sanitize(data) == expected
