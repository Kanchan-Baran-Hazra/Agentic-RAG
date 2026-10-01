import streamlit as st
import requests

# -----------------------------------------------------------------------------
# Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic RAG Assistant")
st.caption("Powered by LangGraph, Qdrant & Groq")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "FastAPI Server URL",
        value="http://localhost:8000/api/v1/query",
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# -----------------------------------------------------------------------------
# Display Chat Messages
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If retrieved documents exist for this message, show them in an expander
        if "docs" in msg and msg["docs"]:
            with st.expander("📚 Retrieved Source Chunks"):
                for idx, doc in enumerate(msg["docs"]):
                    st.markdown(f"**Chunk {idx + 1}**")
                    st.caption(doc.get("page_content", ""))
                    st.divider()

# -----------------------------------------------------------------------------
# Handle User Input
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user prompt immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send query to FastAPI endpoint
    with st.chat_message("assistant"):
        with st.spinner("Thinking & searching context..."):
            try:
                response = requests.post(
                    api_url,
                    json={"question": prompt},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response generated.")
                    retrieved_docs = data.get("retrieved_documents", [])

                    # Render generated answer
                    st.markdown(answer)

                    # Render retrieved context chunks in an expander
                    if retrieved_docs:
                        with st.expander(f"📚 Retrieved ({len(retrieved_docs)} Chunks)"):
                            for idx, doc in enumerate(retrieved_docs):
                                st.markdown(f"**Chunk {idx + 1}**")
                                st.caption(doc.get("page_content", ""))
                                st.divider()

                    # Save response and context into chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "docs": retrieved_docs,
                    })
                else:
                    error_msg = f"API Error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except requests.exceptions.RequestException as e:
                error_msg = f"Connection error: Could not reach backend server at `{api_url}`. Make sure FastAPI is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})