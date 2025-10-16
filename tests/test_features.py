from __future__ import annotations
import pandas as pd
from mlc.features import build_preprocessor


def test_preprocessor_y_agnostic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    pre = build_preprocessor(df)
    pre.fit(df)  # should not require y


def test_unknown_categories():
    df = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "a"]})
    pre = build_preprocessor(df)
    pre.fit(df)
    df2 = pd.DataFrame({"num": [4], "cat": ["zzz"]})
    Xt = pre.transform(df2)
    assert Xt.shape[1] >= 2
