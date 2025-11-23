from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


embeddings = OllamaEmbeddings(model='nomic-embed-text')

llm = ChatOllama(model='llama3.2')

vector_store = Chroma(
  persist_directory='./chroma_db_data',
  embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

system_prompt_text = """You are a cynical, precision-focused kynologist (dog expert). You have access to a specific reference document about Beagles, Beagle-Harriers, and Bearded Collies.

Your instructions:
1.  Answer the user's question using ONLY the provided context. Do not use outside knowledge.
2.  If the answer is not in the text, state clearly: "I cannot answer this based on the provided document."
3.  Cite your sources using the format provided in the text.
    * Example: "Beagles were developed primarily for tracking hare ."
4.  Keep answers concise and technical.
5.  Do not make up facts to fill gaps.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
  ("system", system_prompt_text),
  ("human", "{question}")
])

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


query = 'How does a beagle looks like!'

response = rag_chain.invoke(query)

print(response)



