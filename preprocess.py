import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric_features)),
            ("cat", cat_pipe, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def split_xy(df: pd.DataFrame, target_col: str, id_col: str | None = None):
    y = df[target_col].astype(int)
    drop_cols = [target_col]
    if id_col and id_col in df.columns:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols)
    return X, y
