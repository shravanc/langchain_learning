from langchain_community.document_loaders import PyPDFLoader

file_path = './beagle.pdf'
loader    = PyPDFLoader(file_path)


pages = loader.load()


print(f"Total pages loaded: {len(pages)}") 
print(f"Content of Pae 1: \n{pages[0].page_content[:10]}...")
print(f"Metadata: {pages[0].metadata}")
