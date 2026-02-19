from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    target_col: str = "churn"
    duration_col: str = "tenure_months"
    id_col: str = "customer_id"

    # Columns used for modeling (must exist in the input CSV)
    numeric_features = (
        "tenure_months",
        "monthly_charges",
        "avg_monthly_usage_gb",
        "support_tickets_90d",
        "late_payments_12m",
    )
    categorical_features = (
        "contract",
        "payment_method",
        "internet_service",
        "tech_support",
        "senior_citizen",
        "partner",
        "dependents",
    )

CFG = Config()
