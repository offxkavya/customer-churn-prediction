from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    analyze_risk_node,
    retrieve_rag_node,
    generate_recommendations_node,
    generate_structured_report_node,
    add_disclaimer_node,
)


def build_agent_graph(vectorstore):
    """Build and compile the 5-node LangGraph retention agent."""
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
