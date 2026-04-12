import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------
# Retention Knowledge Base
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
    """Split knowledge base, embed with HuggingFace, and build FAISS index."""
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([RETENTION_KNOWLEDGE])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
