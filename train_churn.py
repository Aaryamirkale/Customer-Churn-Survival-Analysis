import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from src.config import CFG
from src.preprocess import build_preprocessor, split_xy

def train_churn(df: pd.DataFrame, out_dir: str = "models", report_dir: str = "reports", seed: int = 42):
    X, y = split_xy(df, CFG.target_col, CFG.id_col)

    pre = build_preprocessor(CFG.numeric_features, CFG.categorical_features)
    model = LogisticRegression(max_iter=2000, n_jobs=None)

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_test": float(np.mean(y_test)),
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    dump(pipe, Path(out_dir) / "churn_model.joblib")
    with open(Path(report_dir) / "churn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_telco.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    m = train_churn(df)
    print(m)
