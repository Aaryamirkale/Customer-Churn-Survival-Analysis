import argparse
from pathlib import Path
import pandas as pd
from joblib import load

from src.config import CFG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/churn_model.joblib")
    parser.add_argument("--input", required=True, help="CSV with the same feature columns used during training")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()

    pipe = load(args.model)
    df = pd.read_csv(args.input)

    ids = df[CFG.id_col] if CFG.id_col in df.columns else pd.Series(range(len(df)))
    X = df.drop(columns=[c for c in [CFG.target_col, CFG.id_col] if c in df.columns])

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = pd.DataFrame({
        CFG.id_col: ids,
        "churn_probability": proba,
        "churn_pred_0_1": pred,
    })

    Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
