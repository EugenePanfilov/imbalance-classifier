from __future__ import annotations
from typing import List, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def _infer_columns(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(
    X_sample: pd.DataFrame, cfg=None, num_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    if num_cols is None or cat_cols is None:
        num_cols_i, cat_cols_i = _infer_columns(X_sample)
        num_cols = num_cols or num_cols_i
        cat_cols = cat_cols or cat_cols_i

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return preproc
