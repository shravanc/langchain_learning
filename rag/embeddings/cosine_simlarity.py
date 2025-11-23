import numpy as np
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

from sklearn.metrics.pairwise import cosine_similarity


FILE_PATH = './beagle.pdf'
USER_QUERY = 'what is the color of the beagle?'
MODEL_NAME = 'nomic-embed-text'

print("Initializing model...")
embeddings = OllamaEmbeddings(model=MODEL_NAME)

print("Loading and splitting PDF...")
loader = PyPDFLoader(FILE_PATH)
docs = loader.load()

text_splitter = SemanticChunker(embeddings)
chunks = text_splitter.split_documents(docs)

chunk_texts = [chunk.page_content for chunk in chunks]


print(f"Embedding {len(chunks)} chunks (this may take a moment)...")
query_vector = embeddings.embed_query(USER_QUERY)
chunk_vector = embeddings.embed_documents(chunk_texts)


query_vec_np = np.array([query_vector])
chunk_vec_np = np.array(chunk_vector)

scores = cosine_similarity(query_vec_np, chunk_vec_np).flatten()


plt.figure(figsize=(12,6))

num_to_plot = min(len(scores), 20)
plt.bar(range(num_to_plot), scores[:num_to_plot], color='skyblue')

plt.axhline(y=0.5, color='r', linestyle='--', label='Relevance Threshold')


plt.xlabel('Chunk Index')
plt.ylabel('Cosine Similarity Score (0-1)')
plt.title(f'Relevance to Query: "{USER_QUERY}"')
plt.legend()

plt.tight_layout()
plt.show()


best_idx = np.argmax(scores)
print(f"Most relevant chunk (Index {best_idx}, Score {scores[best_idx]:.4f}):")
print(f"---\n{chunk_texts[best_idx]}\n---")


















