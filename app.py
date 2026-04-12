from groq import Groq
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# ---------------------------------------------------
# Groq Setup
# ---------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------
# RAG Knowledge Base
# ---------------------------------------------------
RETENTION_KNOWLEDGE = """
Payment Delay Strategy: Customers with high payment delays benefit from flexible payment plans,
auto-pay incentives, payment reminders, and grace period extensions. Offer 10-15% discount for
enabling auto-pay. Send friendly reminders 3 days before due date.

Support Calls Strategy: Customers making frequent support calls indicate product confusion or
dissatisfaction. Assign dedicated account managers, provide proactive onboarding tutorials,
create personalized FAQ documents, and schedule follow-up satisfaction calls.

Low Tenure Strategy: New customers (under 6 months) are highest churn risk. Implement 30-60-90
day onboarding journeys, provide welcome bonuses, assign onboarding specialists, and create
milestone rewards for reaching 3, 6, and 12 month anniversaries.

High Churn Risk General Strategy: For customers above 70% churn probability, immediate human
intervention is required. Escalate to retention specialists, offer personalized discount
(15-25%), conduct exit-intent surveys to understand core issues, and provide service upgrades
at no extra cost for 3 months.

Moderate Churn Risk Strategy: For customers between 40-70% churn probability, proactive
engagement campaigns work best. Send personalized email campaigns, offer loyalty points,
highlight new features relevant to their usage pattern, and invite them to beta programs.

Contract Renewal Strategy: Monthly contract customers are 3x more likely to churn. Offer
incentives to switch to annual contracts - typically 20% discount. Highlight long-term value
and savings. Send renewal reminders 30 days in advance.

Usage Frequency Strategy: Low usage frequency signals disengagement. Send re-engagement
emails with use-case tutorials, offer personalized feature demos, and create gamification
elements to encourage daily/weekly usage habits.

Ethical Retention Practices: All retention strategies must respect customer autonomy. Never
use manipulative dark patterns. Be transparent about pricing changes. Honor cancellation
requests promptly while presenting alternatives respectfully.
"""

@st.cache_resource
def build_rag_index():
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([RETENTION_KNOWLEDGE])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def retrieve_strategies(query: str, vectorstore, k: int = 3) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

# ---------------------------------------------------
# LangGraph State
# ---------------------------------------------------
class AgentState(TypedDict):
    customer_data: dict
    churn_probability: float
    reasons: List[str]
    retrieved_strategies: str
    risk_summary: str
    retention_recommendations: str
    structured_report: dict
    ethical_disclaimer: str

# ---------------------------------------------------
# LangGraph Nodes
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
    query = f"retention for: {', '.join(state['reasons'])} churn risk {state['churn_probability']*100:.0f}%"
    state["retrieved_strategies"] = retrieve_strategies(query, vectorstore)
    return state

def generate_recommendations_node(state: AgentState) -> AgentState:
    prompt = f"""
You are an expert AI Customer Retention Strategist.

CUSTOMER RISK PROFILE:
{state['risk_summary']}

RETRIEVED BEST PRACTICES:
{state['retrieved_strategies']}

Generate:
1. Top 3 specific, concrete retention actions
2. Priority order
3. Expected outcome for each action

Use clear bullet points. Be specific and actionable.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a customer retention expert. Be concise and specific."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    state["retention_recommendations"] = response.choices[0].message.content
    return state

def generate_structured_report_node(state: AgentState) -> AgentState:
    prompt = f"""
Based on this analysis, return ONLY a valid JSON object (no markdown, no explanation):

Risk Summary: {state['risk_summary']}
Recommendations: {state['retention_recommendations']}

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
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown fences, no explanation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
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
            "escalate_to_human": True
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

def build_agent_graph(vectorstore):
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_risk", analyze_risk_node)
    workflow.add_node("retrieve_rag", lambda s: retrieve_rag_node(s, vectorstore))
    workflow.add_node("generate_recommendations", generate_recommendations_node)
    workflow.add_node("structured_report", generate_structured_report_node)
    workflow.add_node("add_disclaimer", add_disclaimer_node)
    workflow.set_entry_point("analyze_risk")
    workflow.add_edge("analyze_risk", "retrieve_rag")
    workflow.add_edge("retrieve_rag", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "structured_report")
    workflow.add_edge("structured_report", "add_disclaimer")
    workflow.add_edge("add_disclaimer", END)
    return workflow.compile()

def generate_reason(input_data):
    reasons = []
    if input_data["Payment Delay"][0] > 20:
        reasons.append("High payment delay")
    if input_data["Support Calls"][0] > 5:
        reasons.append("Frequent support calls")
    if input_data["Tenure"][0] < 6:
        reasons.append("Low customer tenure")
    if input_data["Usage Frequency"][0] < 10:
        reasons.append("Low usage frequency")
    return reasons

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Premium CSS
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #040d1a !important;
    color: #c8d8f0 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem 3rem !important;
    max-width: 1400px !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f20 0%, #040d1a 100%) !important;
    border-right: 1px solid #0d2444 !important;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem !important; }
[data-testid="stSidebar"] label {
    color: #4a9eff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0a1f3d !important;
    border: 1px solid #0d2444 !important;
    color: #c8d8f0 !important;
}

.hero-wrap {
    background: linear-gradient(135deg, #040d1a 0%, #071628 50%, #040d1a 100%);
    border: 1px solid #0d2444;
    border-radius: 16px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,120,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #0078ff;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(90deg, #e8f4ff 0%, #4a9eff 50%, #00c8b4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #4a6a8a;
    max-width: 520px;
    line-height: 1.8;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,120,255,0.12);
    border: 1px solid rgba(0,120,255,0.3);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.68rem;
    color: #4a9eff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 1.2rem;
    margin-right: 0.5rem;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    background: #060f20;
    border: 1px solid #0d2444;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0078ff, #00c8b4);
}
.stat-label {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2a5a8a;
    margin-bottom: 0.5rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8f4ff;
    line-height: 1;
}
.stat-sub {
    font-size: 0.68rem;
    color: #2a5a8a;
    margin-top: 0.35rem;
}

.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e8f4ff;
    letter-spacing: 0.04em;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #0d2444, transparent);
}

.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.risk-high { background: rgba(255,59,59,0.12); border: 1px solid rgba(255,59,59,0.35); color: #ff6b6b; }
.risk-moderate { background: rgba(255,190,50,0.12); border: 1px solid rgba(255,190,50,0.35); color: #ffd166; }
.risk-low { background: rgba(0,200,130,0.12); border: 1px solid rgba(0,200,130,0.35); color: #00e59b; }

.info-card {
    background: #060f20;
    border: 1px solid #0d2444;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 0.75rem 0;
}
.info-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #0078ff;
    margin-bottom: 0.8rem;
}

.workflow-wrap {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0;
    overflow-x: auto;
    padding: 0.5rem 0;
}
.workflow-step {
    background: #060f20;
    border: 1px solid #0d2444;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    text-align: center;
    min-width: 130px;
    flex-shrink: 0;
}
.workflow-step.active { border-color: #0078ff; background: rgba(0,120,255,0.07); }
.workflow-step .step-num { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: #0078ff; letter-spacing: 0.15em; text-transform: uppercase; }
.workflow-step .step-name { font-family: 'Syne', sans-serif; font-size: 0.76rem; font-weight: 600; color: #c8d8f0; margin-top: 0.3rem; }
.workflow-arrow { color: #0d2444; font-size: 1.4rem; padding: 0 0.3rem; flex-shrink: 0; }

.action-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.9rem 1.1rem;
    background: rgba(0,120,255,0.04);
    border: 1px solid #0d2444;
    border-left: 3px solid #0078ff;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-size: 0.83rem;
    color: #a0b8d8;
    line-height: 1.6;
}
.action-num { font-family: 'Syne', sans-serif; font-weight: 700; color: #0078ff; font-size: 0.9rem; flex-shrink: 0; margin-top: 0.05rem; }

.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,100,80,0.1);
    border: 1px solid rgba(255,100,80,0.25);
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.73rem;
    color: #ff8070;
    margin: 0.25rem;
}

[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a6a8a !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4a9eff !important;
    border-bottom: 2px solid #0078ff !important;
    background: transparent !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0050c8 0%, #0078ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(0,120,255,0.3) !important;
}

.sidebar-logo { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 800; color: #e8f4ff; margin-bottom: 0.3rem; }
.sidebar-tagline { font-size: 0.63rem; color: #2a5a8a; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 1.5rem; }

hr { border-color: #0d2444 !important; }

[data-testid="stMetric"] { background: #060f20; border: 1px solid #0d2444; border-radius: 10px; padding: 1rem 1.2rem; }
[data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: #2a5a8a !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #e8f4ff !important; }

[data-testid="stExpander"] { background: #060f20 !important; border: 1px solid #0d2444 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { font-family: 'DM Mono', monospace !important; font-size: 0.73rem !important; color: #4a6a8a !important; letter-spacing: 0.08em !important; }

.escalate-banner {
    background: rgba(255,59,59,0.08);
    border: 1px solid rgba(255,59,59,0.25);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: #ff8080;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    margin: 1rem 0;
}

.disclaimer-box {
    background: rgba(255,190,50,0.05);
    border: 1px solid rgba(255,190,50,0.2);
    border-left: 3px solid #ffd166;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.5rem;
    font-size: 0.77rem;
    color: #8a7a50;
    line-height: 1.8;
    margin-top: 1rem;
}

[data-testid="stNumberInput"] input {
    background: #0a1f3d !important;
    border: 1px solid #0d2444 !important;
    color: #c8d8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model & RAG
# ---------------------------------------------------
ml_model = joblib.load("churn_model.pkl")

with st.spinner("Initializing AI knowledge base..."):
    vectorstore = build_rag_index()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>🛡️ ChurnGuard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-tagline'>Agentic Retention Intelligence</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;color:#2a5a8a;margin-bottom:1rem;'>Customer Profile</div>", unsafe_allow_html=True)

    age = st.slider("Age", 18, 80, 30)
    tenure = st.slider("Tenure (Months)", 0, 120, 12)
    usage = st.slider("Usage Frequency", 0, 100, 10)
    support = st.slider("Support Calls", 0, 20, 2)
    payment_delay = st.slider("Payment Delay (Days)", 0, 60, 5)
    total_spend = st.number_input("Total Spend ($)", 0.0, 100000.0, 5000.0)
    last_interaction = st.slider("Days Since Last Interaction", 0, 365, 30)
    st.markdown("---")
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    st.markdown("---")
    predict_button = st.button("⚡ Analyze Customer")

# ---------------------------------------------------
# Hero
# ---------------------------------------------------
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-eyebrow'>Powered by LangGraph · FAISS RAG · Groq LLM</div>
    <div class='hero-title'>Customer Churn<br>Intelligence Platform</div>
    <div class='hero-sub'>Agentic AI that predicts churn risk, retrieves retention best practices, and generates structured intervention strategies in real-time.</div>
    <span class='hero-badge'>Milestone 2</span>
    <span class='hero-badge'>LangGraph Workflow</span>
    <span class='hero-badge'>RAG · FAISS</span>
    <span class='hero-badge'>Groq LLM</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Main Logic
# ---------------------------------------------------
if predict_button:
    input_data = pd.DataFrame({
        "Age": [age], "Gender": [gender], "Tenure": [tenure],
        "Usage Frequency": [usage], "Support Calls": [support],
        "Payment Delay": [payment_delay], "Subscription Type": [subscription],
        "Contract Length": [contract], "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    prediction = ml_model.predict(input_data)[0]
    probability = ml_model.predict_proba(input_data)[0][1]
    reasons = generate_reason(input_data)

    risk_level = "HIGH" if probability >= 0.7 else ("MODERATE" if probability >= 0.4 else "LOW")
    risk_class = {"HIGH": "risk-high", "MODERATE": "risk-moderate", "LOW": "risk-low"}[risk_level]
    risk_icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[risk_level]
    prob_color = "#ff6b6b" if probability > 0.7 else ("#ffd166" if probability > 0.4 else "#00e59b")

    # Stat Cards
    st.markdown(f"""
    <div class='stat-grid'>
        <div class='stat-card'>
            <div class='stat-label'>Churn Probability</div>
            <div class='stat-value' style='color:{prob_color}'>{probability*100:.1f}%</div>
            <div class='stat-sub'>Model confidence score</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Risk Level</div>
            <div class='stat-value' style='font-size:1.4rem;'>{risk_icon} {risk_level}</div>
            <div class='stat-sub'>Classification result</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Risk Factors</div>
            <div class='stat-value'>{len(reasons)}</div>
            <div class='stat-sub'>Issues detected</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Tenure</div>
            <div class='stat-value'>{tenure}<span style='font-size:1rem;color:#2a5a8a'> mo</span></div>
            <div class='stat-sub'>{contract} · {subscription}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  📡  RISK ANALYSIS  ", "  📊  CUSTOMER PROFILE  ", "  🤖  AI AGENT REPORT  "])

    # ── TAB 1: Risk Analysis ──
    with tab1:
        col_gauge, col_info = st.columns([1.2, 1])

        with col_gauge:
            st.markdown("<div class='section-head'>Churn Risk Gauge</div>", unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': '%', 'font': {'size': 52, 'color': prob_color, 'family': 'Syne'}},
                title={'text': "CHURN PROBABILITY", 'font': {'size': 11, 'color': '#2a5a8a', 'family': 'DM Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#0d2444', 'tickfont': {'color': '#2a5a8a', 'size': 9}},
                    'bar': {'color': prob_color, 'thickness': 0.22},
                    'bgcolor': '#060f20',
                    'bordercolor': '#0d2444',
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(0,200,130,0.06)'},
                        {'range': [40, 70], 'color': 'rgba(255,210,50,0.06)'},
                        {'range': [70, 100], 'color': 'rgba(255,60,60,0.06)'},
                    ],
                    'threshold': {'line': {'color': prob_color, 'width': 2}, 'thickness': 0.75, 'value': probability * 100}
                }
            ))
            gauge.update_layout(
                paper_bgcolor='#060f20', plot_bgcolor='#060f20',
                font={'family': 'DM Mono', 'color': '#c8d8f0'},
                height=290, margin=dict(t=40, b=20, l=30, r=30)
            )
            st.plotly_chart(gauge, use_container_width=True)

        with col_info:
            st.markdown("<div class='section-head'>Risk Assessment</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='risk-badge {risk_class}'>{risk_icon} {risk_level} RISK</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='info-card-title' style='margin-top:1rem;letter-spacing:0.15em;font-size:0.62rem;text-transform:uppercase;color:#0078ff;'>Detected Risk Factors</div>", unsafe_allow_html=True)
            if reasons:
                pills_html = "".join([f"<span class='risk-pill'>⚠ {r}</span>" for r in reasons])
                st.markdown(f"<div style='margin-top:0.5rem'>{pills_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#00e59b;font-size:0.85rem;'>✓ No critical risk factors detected</span>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Radar
            categories = ['Payment', 'Support', 'Engagement', 'Loyalty', 'Spend']
            values = [
                min(payment_delay / 60, 1),
                min(support / 20, 1),
                1 - min(usage / 100, 1),
                1 - min(tenure / 120, 1),
                1 - min(total_spend / 100000, 1)
            ]
            radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]], theta=categories + [categories[0]],
                fill='toself', fillcolor='rgba(255,80,80,0.08)',
                line=dict(color='#ff6b6b', width=2), marker=dict(color='#ff6b6b', size=5)
            ))
            radar.update_layout(
                polar=dict(
                    bgcolor='#060f20',
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='#2a5a8a', size=8), gridcolor='#0d2444'),
                    angularaxis=dict(tickfont=dict(color='#4a6a8a', size=10), gridcolor='#0d2444', linecolor='#0d2444'),
                ),
                paper_bgcolor='#060f20', showlegend=False,
                height=230, margin=dict(t=20, b=20, l=40, r=40)
            )
            st.markdown("<div class='info-card-title' style='letter-spacing:0.15em;font-size:0.62rem;text-transform:uppercase;color:#0078ff;margin-top:1.5rem;'>Risk Radar</div>", unsafe_allow_html=True)
            st.plotly_chart(radar, use_container_width=True)

    # ── TAB 2: Customer Profile ──
    with tab2:
        col_a, col_b = st.columns([1, 1.2])
        with col_a:
            st.markdown("<div class='section-head'>Feature Summary</div>", unsafe_allow_html=True)
            st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True, height=380)
        with col_b:
            st.markdown("<div class='section-head'>Behavioral Metrics</div>", unsafe_allow_html=True)
            metrics_data = {
                "Usage Frequency": (usage, 100, "#0078ff"),
                "Support Calls": (support, 20, "#ff6b6b"),
                "Payment Delay": (payment_delay, 60, "#ffd166"),
                "Last Interaction": (last_interaction, 365, "#00c8b4")
            }
            bar_fig = go.Figure()
            for metric, (val, max_val, color) in metrics_data.items():
                bar_fig.add_trace(go.Bar(
                    x=[val / max_val * 100], y=[metric], orientation='h',
                    marker_color=color, marker_line_width=0, name=metric,
                    text=f"{val}", textposition='inside',
                    textfont=dict(color='white', size=11, family='DM Mono')
                ))
            bar_fig.update_layout(
                paper_bgcolor='#060f20', plot_bgcolor='#060f20',
                showlegend=False, barmode='group',
                xaxis=dict(range=[0, 100], ticksuffix='%', tickfont=dict(color='#2a5a8a', size=9), gridcolor='#0d2444', zeroline=False),
                yaxis=dict(tickfont=dict(color='#c8d8f0', size=11, family='DM Mono'), gridcolor='#0d2444'),
                height=240, margin=dict(t=10, b=20, l=10, r=20)
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Spend", f"${total_spend:,.0f}")
            mc2.metric("Tenure", f"{tenure} mo")
            mc3.metric("Age", f"{age} yrs")

    # ── TAB 3: AI Agent Report ──
    with tab3:
        st.markdown("<div class='section-head'>LangGraph Agent Pipeline</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='workflow-wrap'>
            <div class='workflow-step active'><div class='step-num'>Node 01</div><div class='step-name'>Risk Analysis</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step active'><div class='step-num'>Node 02</div><div class='step-name'>RAG Retrieval</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step active'><div class='step-num'>Node 03</div><div class='step-name'>LLM Strategy</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step active'><div class='step-num'>Node 04</div><div class='step-name'>Struct. Report</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step active'><div class='step-num'>Node 05</div><div class='step-name'>Disclaimer</div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🧠 Running agentic pipeline..."):
            agent = build_agent_graph(vectorstore)
            initial_state: AgentState = {
                "customer_data": input_data.to_dict(),
                "churn_probability": float(probability),
                "reasons": reasons,
                "retrieved_strategies": "",
                "risk_summary": "",
                "retention_recommendations": "",
                "structured_report": {},
                "ethical_disclaimer": ""
            }
            final_state = agent.invoke(initial_state)

        report = final_state["structured_report"]

        if report.get("escalate_to_human"):
            st.markdown("<div class='escalate-banner'>🚨 ESCALATION REQUIRED — Assign to Human Retention Specialist Immediately</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>Structured Retention Report</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Level", report.get("risk_level", "N/A"))
        m2.metric("Churn Probability", f"{report.get('churn_probability_pct', 0):.1f}%")
        m3.metric("Priority Score", f"{report.get('priority_score', 0)} / 10")

        st.markdown("<br>", unsafe_allow_html=True)
        col_factors, col_outcomes = st.columns(2)
        with col_factors:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("<div class='info-card-title'>⚠ Top Risk Factors</div>", unsafe_allow_html=True)
            for f in report.get("top_risk_factors", []):
                st.markdown(f"<span class='risk-pill'>⚠ {f}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_outcomes:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("<div class='info-card-title'>✅ Expected Outcomes</div>", unsafe_allow_html=True)
            for o in report.get("expected_outcomes", []):
                st.markdown(f"<div style='font-size:0.8rem;color:#4a9eff;padding:0.25rem 0;line-height:1.5;'>→ {o}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>Immediate Action Plan</div>", unsafe_allow_html=True)
        for i, action in enumerate(report.get("immediate_actions", []), 1):
            st.markdown(f"<div class='action-item'><span class='action-num'>0{i}</span>{action}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>AI-Generated Strategy Detail</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-card'>
            <div class='info-card-title'>LLM Retention Analysis · llama-3.3-70b-versatile via Groq</div>
            <div style='font-size:0.82rem;line-height:1.9;color:#a0b8d8;white-space:pre-wrap;'>{final_state['retention_recommendations']}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📚 Retrieved Knowledge Base Context (FAISS RAG)"):
            st.markdown(f"""<div style='font-family:DM Mono,monospace;font-size:0.74rem;color:#4a6a8a;line-height:1.9;white-space:pre-wrap;'>{final_state['retrieved_strategies']}</div>""", unsafe_allow_html=True)

        with st.expander("🎯 Agent Risk Summary"):
            st.markdown(f"""<div style='font-family:DM Mono,monospace;font-size:0.78rem;color:#4a9eff;line-height:1.8;'>{final_state['risk_summary']}</div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>Ethical AI & Compliance</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='disclaimer-box'>
            ⚖️ <strong>Ethical AI Disclaimer:</strong> {final_state['ethical_disclaimer']}<br><br>
            📌 <strong>Business Disclaimer:</strong> Retention strategies are generated for decision-support only. All customer interactions must comply with your organization's data privacy policy and applicable consumer protection laws. AI predictions carry inherent uncertainty — human judgment must complement automated recommendations.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;'>
        <div style='font-size:3.5rem;margin-bottom:1.5rem;'>🛡️</div>
        <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;color:#e8f4ff;margin-bottom:0.75rem;'>Ready to Analyze</div>
        <div style='font-family:DM Mono,monospace;font-size:0.78rem;color:#2a5a8a;max-width:420px;margin:0 auto;line-height:1.9;'>
            Enter customer details in the sidebar and click <strong style='color:#4a9eff;'>⚡ Analyze Customer</strong> to run the full 5-node agentic AI pipeline.
        </div>
    </div>
    <div style='margin-top:3rem;'>
        <div style='font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:0.2em;text-transform:uppercase;color:#2a5a8a;text-align:center;margin-bottom:1.5rem;'>5-Node LangGraph Workflow</div>
        <div class='workflow-wrap' style='justify-content:center;'>
            <div class='workflow-step'><div class='step-num'>Node 01</div><div class='step-name'>Risk Analysis</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step'><div class='step-num'>Node 02</div><div class='step-name'>RAG Retrieval</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step'><div class='step-num'>Node 03</div><div class='step-name'>LLM Strategy</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step'><div class='step-num'>Node 04</div><div class='step-name'>Struct. Report</div></div>
            <div class='workflow-arrow'>→</div>
            <div class='workflow-step'><div class='step-num'>Node 05</div><div class='step-name'>Disclaimer</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)