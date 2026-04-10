import json
import os
from groq import Groq
from agent.state import AgentState
from agent.prompts import (
    RECOMMENDATIONS_SYSTEM_PROMPT,
    RECOMMENDATIONS_USER_PROMPT,
    STRUCTURED_REPORT_SYSTEM_PROMPT,
    STRUCTURED_REPORT_USER_PROMPT,
)
from rag.retriever import retrieve_strategies

# ---------------------------------------------------
# Groq Client
# ---------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------
# Node Functions
# ---------------------------------------------------
def analyze_risk_node(state: AgentState) -> AgentState:
    prob = state["churn_probability"]
    data = state["customer_data"]
    reasons = state["reasons"]
    risk_level = "HIGH" if prob >= 0.7 else ("MODERATE" if prob >= 0.4 else "LOW")
    state["risk_summary"] = (
        f"Risk Level: {risk_level} ({prob*100:.1f}%)\n"
        f"Tenure: {data.get('Tenure', ['N/A'])[0]} months | "
        f"Support Calls: {data.get('Support Calls', ['N/A'])[0]} | "
        f"Payment Delay: {data.get('Payment Delay', ['N/A'])[0]} days\n"
        f"Issues: {', '.join(reasons) if reasons else 'None detected'}"
    )
    return state


def retrieve_rag_node(state: AgentState, vectorstore) -> AgentState:
    query = (
        f"retention for: {', '.join(state['reasons'])} "
        f"churn risk {state['churn_probability']*100:.0f}%"
    )
    state["retrieved_strategies"] = retrieve_strategies(query, vectorstore)
    return state


def generate_recommendations_node(state: AgentState) -> AgentState:
    prompt = RECOMMENDATIONS_USER_PROMPT.format(
        risk_summary=state["risk_summary"],
        retrieved_strategies=state["retrieved_strategies"],
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RECOMMENDATIONS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    state["retention_recommendations"] = response.choices[0].message.content
    return state


def generate_structured_report_node(state: AgentState) -> AgentState:
    prompt = STRUCTURED_REPORT_USER_PROMPT.format(
        risk_summary=state["risk_summary"],
        retention_recommendations=state["retention_recommendations"],
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": STRUCTURED_REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    raw = (
        response.choices[0]
        .message.content.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    try:
        state["structured_report"] = json.loads(raw)
    except Exception:
        state["structured_report"] = {
            "risk_level": "UNKNOWN",
            "churn_probability_pct": state["churn_probability"] * 100,
            "top_risk_factors": state["reasons"],
            "immediate_actions": ["Review customer profile manually"],
            "expected_outcomes": ["Improved retention"],
            "priority_score": 5,
            "escalate_to_human": True,
        }
    return state


def add_disclaimer_node(state: AgentState) -> AgentState:
    state["ethical_disclaimer"] = (
        "This analysis is AI-generated and intended as decision support only. "
        "All retention strategies must respect customer autonomy and data privacy regulations. "
        "Human review is strongly recommended before executing high-impact interventions. "
        "Predictions carry inherent uncertainty — use alongside human judgment."
    )
    return state
