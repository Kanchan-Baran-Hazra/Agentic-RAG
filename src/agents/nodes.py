import os
import logfire
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from src.models.main import get_llm,get_groq_llm
from langchain_qdrant import QdrantVectorStore

from src.agents.state import AgentState
from src.ingestion.vectorstore.qdrant import load_existing_qdrant_store
from src.config import Config

from flashrank import Ranker, RerankRequest
from src.models.main import ranker  # Import the ranker instance from models/main.py    



class RouteQueryResponse(BaseModel):
    """Pydantic schema for routing decision."""

    destination: Literal["retrieve", "generate"] = Field(
        description="Choose 'retrieve' if document knowledge base is needed, or 'generate' for direct answers/greetings."
    )


# -----------------------------------------------------------------------------
# 1. Start Node
# -----------------------------------------------------------------------------
def start_node(state: AgentState) -> Dict[str, Any]:
    """Initializes the pipeline state and logs incoming query."""
    question = state.get("question", "")
    logfire.info("Graph Started", question=question)

    return {
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "generation": "",
    }


# -----------------------------------------------------------------------------
# 2. Route Node (Conditional Logic)
# -----------------------------------------------------------------------------
def route_node(state: AgentState) -> str:
    """Evaluates if retrieval from Qdrant is required.

    Returns:
        'retrieve' or 'generate'
    """
    question = state["question"]
    llm = get_llm().with_structured_output(RouteQueryResponse)

    prompt = f"""You are an expert query router.
    Analyze the user question and determine if it requires retrieving domain specific documents/context.

    - Choose 'retrieve': If the query asks about technical details, documents, specs, or specific project knowledge.
    - Choose 'generate': If the query is a simple greeting (e.g., 'hi', 'hello'), general chit-chat, or basic reasoning.

    Question: {question}"""

    try:
        decision = llm.invoke(prompt)
        print("...")
        logfire.info("Query Route Decision", decision=decision.destination)
        print("....")
        return decision.destination
    except Exception as e:
        logfire.error("Routing failed, defaulting to retrieve", error=str(e))
        return "retrieve"


# -----------------------------------------------------------------------------
# 3. Retrieve Node
# -----------------------------------------------------------------------------
def retrieve_node(
    state: AgentState, 
    embeddings, 
    collection_name: str = "rag_documents",
    qdrant_path: str = Config.QDRANT_CLUSTER_ENDPOINT
) -> Dict[str, Any]:
    """Fetches relevant document chunks from Qdrant."""
    question = state["question"]

    with logfire.span("qdrant_retrieval", question=question):
        # Load existing Qdrant vector store
        vector_store: QdrantVectorStore = load_existing_qdrant_store(
            embeddings=embeddings,
            collection_name=collection_name,
            qdrant_path=qdrant_path,
        )

        # Retrieve top-k documents
        docs = vector_store.similarity_search(query=question, k=8)
        logfire.info("Retrieved Documents", count=len(docs))

        return {"documents": docs}


# -----------------------------------------------------------------------------
# 4. Generate Node
# -----------------------------------------------------------------------------
def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generates an answer using Gemini with retrieved context (if present)."""
    question = state["question"]
    docs = state.get("documents", [])
    llm = get_groq_llm()

    with logfire.span("llm_generation", doc_count=len(docs)):
        if docs:
            # Context-Augmented Generation
            context_text = "\n\n".join(
                [f"--- Chunk {i+1} ---\n{d.page_content}" for i, d in enumerate(docs)]
            )
            prompt = f"""You are a helpful assistant. Use the following retrieved context to answer the question.
If you don't know the answer or if the context doesn't contain it, state that clearly.

Context:
{context_text}

Question: {question}
Answer:"""
        else:
            # Direct Conversation Generation
            prompt = f"Answer the user query concisely and politely:\n\nQuestion: {question}"

        response = llm.invoke(prompt)
        generation_text = str(response.content)

        return {"generation": generation_text}


# -----------------------------------------------------------------------------
# 5. End Node
# -----------------------------------------------------------------------------
def end_node(state: AgentState) -> Dict[str, Any]:
    """Finalizes graph execution and formats the output message."""
    generation = state["generation"]
    logfire.info("Graph Execution Completed")

    return {
        "messages": [AIMessage(content=generation)]
    }




def rerank_node(state: AgentState):
    question = state["question"]
    raw_docs = state["documents"]  # e.g., 20 retrieved docs from Qdrant
    
    if not raw_docs:
        return {"documents": []}
    
    # Format documents for FlashRank
    passages = [
        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
        for idx, doc in enumerate(raw_docs)
    ]
    
    rerank_request = RerankRequest(query=question, passages=passages)
    results = ranker.rerank(rerank_request)
    
    # Select top 4 highest scoring documents
    top_results = results[:4]
    
    # Map back to standard Document format
    reranked_docs = [
        raw_docs[item["id"]] for item in top_results
    ]
    
    return {"documents": reranked_docs}