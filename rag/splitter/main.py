# Length Based Splitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = './beagle.pdf'
loader    = PyPDFLoader(file_path)


pages = loader.load()


print(f"Total pages loaded: {len(pages)}") 
print(f"Content of Pae 1: \n{pages[0].page_content[:10]}...")
print(f"Metadata: {pages[0].metadata}")

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200
)

chunks = loader.load_and_split(text_splitter=text_splitter)

print(f"Total chunks created: {len(chunks)}")
print(f"First chunk content: {chunks[0].page_content}")



