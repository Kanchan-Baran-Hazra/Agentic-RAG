import os
from dotenv import load_dotenv
from src.agents.graph import build_rag_graph
from src.agents.state import create_initial_state
from src.models.main import get_gemini_embedder


def main():
    # 2. Initialize Embeddings & Build Compiled Graph
    print("Initializing embeddings and building agent graph...")
    embeddings = get_gemini_embedder()
    agent = build_rag_graph(embeddings=embeddings)
    
    print("Agent graph initialized successfully!\n" + "=" * 50)
    
    
    
    test_1_input = create_initial_state(question="what is kubernaties and how does it work?")
    
    print("\n--- Test 1: Casual Greeting ---")
    output_1 = agent.invoke(test_1_input)
    
    print("\nRetrieved Docs Count:", len(output_1.get("documents", [])))
    print("Response:\n", output_1["generation"])
    
    



