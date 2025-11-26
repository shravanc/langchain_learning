import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PDF_DIR_PATH = './boxilla_docs'
DB_PATH = './boxilla_api_db_data'
MODEL_NAME = 'nomic-embed-text'

embeddings = OllamaEmbeddings(
    model=MODEL_NAME,
    base_url="http://localhost:11434"
)

if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
  print("Loading existing vector store from disk....")
  vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
  )
else:
  print("Database not found. Creating new one (This takes time)...")
  loader = DirectoryLoader(
          PDF_DIR_PATH, 
          glob="*.pdf",
          loader_cls=PyPDFLoader,
          show_progress=True
  )
  docs = loader.load()
  
  text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
  )
  chunks = text_splitter.split_documents(docs)


  vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
  )
  print("Finished embedding and saved to disk.")


query = "what is boxilla about?"
print(f"Query: {query}")
results = vector_store.similarity_search(query, k=3)

for i, res in enumerate(results):
  print(f"\nResult {i+1}:")
  print(res.page_content[:200] + "...")


