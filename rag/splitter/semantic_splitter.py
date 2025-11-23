import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

# 1. Setup Embeddings with explicit Base URL and Model
# 'nomic-embed-text' is highly recommended for local stability over llama3
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434" 
)

# 2. Initialize Semantic Chunker
# "percentile" is the standard threshold logic
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# 3. Load PDF
# If the PDF is huge, test with just one page first to verify stability
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pdf_path = os.path.join(project_root, "beagle.pdf") # Assumes beagle.pdf is in the project root
loader = PyPDFLoader(pdf_path)
docs = loader.load() 

print(f"Loaded {len(docs)} pages. Starting semantic split (this may take time)...")

try:
    # 4. Split
    chunks = semantic_splitter.split_documents(docs)
    print(f"Successfully created {len(chunks)} chunks.")
    print(f"First chunk preview: {chunks[0].page_content[:200]}")
except Exception as e:
    print(f"Error during splitting: {e}")
    print("Tip: Ensure 'ollama serve' is running in a separate terminal.")
