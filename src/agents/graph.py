from functools import partial
from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.nodes import (
    start_node,
    route_node,
    retrieve_node,
    rerank_node,
    generate_node,
    end_node,
)


def build_rag_graph(
    embeddings,
    collection_name: str = "rag_documents",
    qdrant_path: str = None,
):
    """Constructs and compiles the 5-node LangGraph agentic RAG graph."""
    
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("start", start_node)
    
    # Bind embeddings configuration to retrieve_node
    retrieve_fn = partial(
        retrieve_node,
        embeddings=embeddings,
        collection_name=collection_name,
        qdrant_path=qdrant_path,
    )
    workflow.add_node("retrieve", retrieve_fn)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("end", end_node)

    # 2. Define Edges & Flow
    workflow.add_edge(START, "start")

    # Conditional Routing from START/START_NODE
    workflow.add_conditional_edges(
        "start",
        route_node,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # Sequential Edges
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "end")
    workflow.add_edge("end", END)

    # Compile Graph
    return workflow.compile()