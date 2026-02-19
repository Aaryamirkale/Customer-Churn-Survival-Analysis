import numpy as np
import pandas as pd

def make_synthetic_telco(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n, p=[0.35, 0.20, 0.25, 0.20])
    internet_service = rng.choice(["Fiber", "DSL", "None"], size=n, p=[0.50, 0.35, 0.15])
    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    senior = rng.choice(["Yes", "No"], size=n, p=[0.18, 0.82])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # Tenure distribution: more short-tenure than long-tenure
    tenure = rng.integers(1, 73, size=n)  # months 1..72

    # Numeric behaviors
    base_charge = rng.normal(70, 25, size=n).clip(18, 130)
    # Internet service affects charges & usage
    is_fiber = (internet_service == "Fiber").astype(int)
    is_none = (internet_service == "None").astype(int)
    monthly_charges = (base_charge + 15*is_fiber - 20*is_none).clip(18, 160)

    avg_usage = (rng.gamma(shape=2.0, scale=8.0, size=n) + 12*is_fiber - 8*is_none).clip(0, 200)
    tickets = rng.poisson(lam=1.1 + 0.6*(tech_support=="No").astype(int), size=n).clip(0, 12)
    late_pay = rng.poisson(lam=0.9 + 0.7*(payment_method=="Electronic check").astype(int), size=n).clip(0, 12)

    # Churn risk score (log-odds)
    # Higher churn: month-to-month, electronic check, fiber (price), no tech support, short tenure, more tickets/late payments
    score = (
        1.1*(contract=="Month-to-month").astype(int)
        -0.6*(contract=="One year").astype(int)
        -1.0*(contract=="Two year").astype(int)
        +0.6*(payment_method=="Electronic check").astype(int)
        +0.3*is_fiber
        -0.8*(tech_support=="Yes").astype(int)
        +0.08*tickets
        +0.10*late_pay
        -0.035*tenure
        +0.15*(senior=="Yes").astype(int)
    )

    # Convert score to churn probability
    p = 1/(1+np.exp(-score))
    churn = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "tenure_months": tenure.astype(int),
        "monthly_charges": monthly_charges.round(2),
        "avg_monthly_usage_gb": avg_usage.round(2),
        "support_tickets_90d": tickets.astype(int),
        "late_payments_12m": late_pay.astype(int),
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "senior_citizen": senior,
        "partner": partner,
        "dependents": dependents,
        "churn": churn.astype(int),
    })
    return df

if __name__ == "__main__":
    df = make_synthetic_telco()
    print(df.head())
