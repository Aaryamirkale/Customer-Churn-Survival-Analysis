from pathlib import Path
import pandas as pd

from src.make_synthetic_data import make_synthetic_telco
from src.train_churn import train_churn
from src.train_survival import train_survival

def main():
    Path("data").mkdir(exist_ok=True)
    df = make_synthetic_telco(n=3000, seed=42)
    df.to_csv("data/synthetic_telco.csv", index=False)

    churn_metrics = train_churn(df)
    survival_out = train_survival(df)

    print("Churn metrics:", churn_metrics)
    print("Survival results:", survival_out)
    print("Done. See /reports and /models.")

if __name__ == "__main__":
    main()
