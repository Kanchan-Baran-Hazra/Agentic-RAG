from src.config import Config
from src.middleware import init_observability
from langchain_core.documents import Document


# Initialize telemetry BEFORE creating chains or graphs
init_observability(Config.LOGFIRE_SERVICE_NAME)

import logfire

logfire.info('Hello, {place}!', place='World')



# from src.ingestion.loaders.pdf import safe_load_pdf
# from src.ingestion.loaders.text import safe_load_text
# from src.ingestion.loaders.office import safe_load_word_doc
# from src.ingestion.loaders.pptx import safe_load_pptx
# from src.ingestion.loaders.html import safe_load_html
# print(safe_load_html(r"DATA\true_data\pods_autoscale.html"))
# print(safe_load_pptx(r"DATA\true_data\architecture.pptx"))
# print(safe_load_word_doc(r"DATA\true_data\cronjobs.docx"))
# print(safe_load_text(r"DATA\true_data\parallel_work_queue.txt"))
# print(safe_load_pdf(r"DATA\noisy_data\5-Level Paging and 5-Level EPT - Intel - Revision 1.0 (December, 2016).pdf"))



# from src.ingestion.processor import load_documents_from_directory
# from src.ingestion.chuncking.chunker import safe_chunk_documents
# from src.ingestion.retrival.embedder import embed_chunks_resumable
# from src.models.main import get_gemini_embedder
# from src.ingestion.vectorstore.qdrant import store_documents_in_qdrant, load_existing_qdrant_store

# docs=load_documents_from_directory(r"DATA\true_data", recursive=True)

# chunks=safe_chunk_documents([Document(page_content="Sample document content", metadata={"source": "sample_source.txt"})])
# embedder=get_gemini_embedder()

# # print(embed_chunks_resumable(chunks, get_gemini_embedder())[0])

# # 5. Step 4: Embed & Store directly into Qdrant
# vector_store = store_documents_in_qdrant(
#     documents=chunks,
#     embeddings=embedder,
#     collection_name="rag_documents",
#     qdrant_path="./qdrant_db",
#     q_url=Config.QDRANT_URL,
#     q_key=Config.QDRANT_API_KEY,
# )

# print(f"\nPipeline Complete! Ingested {len(chunks)} chunks into Qdrant.")
# # 6. Test Query Retrieval
# query = "What is the kubernetes Horizontal Pod Autoscaler?"
# vector_store = load_existing_qdrant_store(
#     embeddings=get_gemini_embedder(),
#     collection_name=Config.QDRANT_COLLECTION_NAME,
#     qdrant_path="./qdrant_db",
# )
# search_results = vector_store.similarity_search(query, k=3)

# print("\n--- Similarity Search Results ---")
# for i, doc in enumerate(search_results, start=1):
#     print(f"\nResult {i}:")
#     print(f"Content: {doc.page_content[:150]}...")
#     print(f"Metadata: {doc.metadata}")

# from src.ingestion.processor import run_ingestion_pipeline

# if __name__ == "__main__":
#     # Execute integrated pipeline
#     result = run_ingestion_pipeline(
#         data_directory=r"DATA\noisy_data",
#         collection_name="rag_documents",
#         chunk_size=1000,
#         chunk_overlap=200,
#         batch_size=10,
#         recursive=True,
#     )

#     print("\n--- Pipeline Execution Summary ---")
#     for key, value in result.items():
#         print(f"{key}: {value}")

# from src.agents.main import main

# if __name__ == "__main__":
#     main()