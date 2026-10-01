import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The central state of the LangGraph Agentic RAG pipeline.

    Attributes:
        messages: Standard message sequence (HumanMessage, AIMessage, ToolMessage).
                 The `add_messages` reducer appends new messages automatically.
        question: The re-phrased/standalone user question being processed.
        documents: Retrieved chunk documents from Qdrant.
        generation: Final generated answer text from the LLM.
        metadata: Arbitrary runtime execution parameters or telemetry data.
    """

    # Message reducer appends new messages without overwriting history
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Custom state variables for Advanced RAG
    question: str
    documents: List[Document]
    generation: str
    metadata: Dict[str, Any]


def create_initial_state(
    question: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """Helper utility to initialize clean graph state at runtime."""
    return {
        "messages": [],
        "question": question,
        "documents": [],
        "generation": "",
        "metadata": metadata or {},
    }