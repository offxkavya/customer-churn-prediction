RECOMMENDATIONS_SYSTEM_PROMPT = (
    "You are a customer retention expert. Be concise and specific."
)

RECOMMENDATIONS_USER_PROMPT = """
You are an expert AI Customer Retention Strategist.
CUSTOMER RISK PROFILE:
{risk_summary}
RETRIEVED BEST PRACTICES:
{retrieved_strategies}
Generate:
1. Top 3 specific, concrete retention actions
2. Priority order
3. Expected outcome for each action
Use clear bullet points. Be specific and actionable.
"""

STRUCTURED_REPORT_SYSTEM_PROMPT = (
    "Return only valid JSON. No markdown fences, no explanation."
)

STRUCTURED_REPORT_USER_PROMPT = """
Based on this analysis, return ONLY a valid JSON object (no markdown, no explanation):
Risk Summary: {risk_summary}
Recommendations: {retention_recommendations}
{{
  "risk_level": "HIGH/MODERATE/LOW",
  "churn_probability_pct": <number>,
  "top_risk_factors": ["factor1", "factor2"],
  "immediate_actions": ["action1", "action2", "action3"],
  "expected_outcomes": ["outcome1", "outcome2", "outcome3"],
  "priority_score": <1-10>,
  "escalate_to_human": true/false
}}
"""
