import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.config import CFG
from src.preprocess import build_preprocessor
from src.survival import kaplan_meier, concordance_index

def _design_matrix(df: pd.DataFrame):
    """Create numeric design matrix for CoxPH using the same preprocessing approach."""
    X = df[list(CFG.numeric_features) + list(CFG.categorical_features)].copy()
    pre = build_preprocessor(CFG.numeric_features, CFG.categorical_features)
    Xp = pre.fit_transform(X)
    feature_names = pre.get_feature_names_out()
    Xp = pd.DataFrame(Xp, columns=feature_names)
    return Xp, pre, feature_names

def train_survival(df: pd.DataFrame, out_dir: str = "models", report_dir: str = "reports"):
    # Prepare duration/event
    duration = df[CFG.duration_col].astype(float).values
    event = df[CFG.target_col].astype(int).values

    Xp, pre, feature_names = _design_matrix(df)

    # statsmodels CoxPH expects: endog = duration, exog = X, status = event
    cox = sm.duration.hazard_regression.PHReg(duration, Xp, status=event)
    res = cox.fit(disp=0)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir + "/figures").mkdir(parents=True, exist_ok=True)

    # Save hazard ratios
    params = res.params
    hr = np.exp(params)
    hr_df = pd.DataFrame({
        "feature": feature_names,
        "coef": params,
        "hazard_ratio": hr,
    }).sort_values("hazard_ratio", ascending=False)
    hr_path = Path(report_dir) / "survival_hazard_ratios.csv"
    hr_df.to_csv(hr_path, index=False)

    # Risk score for concordance (linear predictor)
    risk = Xp.values @ params
    c_index = float(concordance_index(duration, event, risk))

    out = {
        "c_index": c_index,
        "n": int(len(df)),
        "note": "C-index computed with a simple Harrell's C implementation.",
    }
    with open(Path(out_dir) / "survival_cox_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # KM plot overall + by churn flag segments (example by contract)
    t, s = kaplan_meier(duration, event)
    plt.figure()
    plt.step(t, s, where="post")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Survival probability (not churned yet)")
    plt.title("Kaplan–Meier Survival Curve (Overall)")
    plt.tight_layout()
    plt.savefig(Path(report_dir) / "figures" / "km_overall.png", dpi=160)
    plt.close()

    # KM by contract type
    for ct in df["contract"].unique():
        sub = df[df["contract"] == ct]
        tt, ss = kaplan_meier(sub[CFG.duration_col].values, sub[CFG.target_col].values)
        plt.step(tt, ss, where="post", label=str(ct))
    plt.xlabel("Tenure (months)")
    plt.ylabel("Survival probability")
    plt.title("Kaplan–Meier by Contract")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(report_dir) / "figures" / "km_by_contract.png", dpi=160)
    plt.close()

    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_telco.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    out = train_survival(df)
    print(out)
