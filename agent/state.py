from typing import TypedDict, List


class AgentState(TypedDict):
    customer_data: dict
    churn_probability: float
    reasons: List[str]
    retrieved_strategies: str
    risk_summary: str
    retention_recommendations: str
    structured_report: dict
    ethical_disclaimer: str
