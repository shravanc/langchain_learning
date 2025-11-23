from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings



embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2",
  model_kwargs={'device': 'cpu'}
)

semantic_splitter = SemanticChunker(
  embeddings,
  breakpoint_threshold_type="percentile"
)

loader = PyPDFLoader('./beagle.pdf')
docs = loader.load()
chunks = semantic_splitter.split_documents(docs)


print(f"Created {len(chunks)} chunks using SentenceTransformer.")
