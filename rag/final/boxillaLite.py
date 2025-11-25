import os
from operator import itemgetter

import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Global variables
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2")
vector_store = Chroma(persist_directory="./boxilla_db_data", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

system_prompt_text = """You are a Senior Technical Support Specialist for Black Box Boxilla KVM Systems. You have access to the official user manual for the Boxilla KVM Manager (Model BXAMGR-R2, Firmware v5.1.5+).

Your core mission is to assist system administrators with installation, configuration, and troubleshooting using ONLY the provided context text.

### Instructions:
1. **Source-Based Truth:** Answer strictly based on the "Context" provided below. Do not use outside knowledge or make assumptions about network configurations not specified in the text.
2. **Negative Constraint:** If the answer is not explicitly present in the text, you must state: "I cannot answer this based on the provided document."
3. **Citation Style:** Cite your sources immediately following the claim.
   - Format: [Source: EN_KVM_Manual_BXAMGR-R2_Ver5.1.5_Rev1_2509_30819.pdf - Page X]
4. **Tone & Style:** Maintain a professional, technical, and instructional tone. Use step-by-step formatting for configuration procedures.

### Context Data:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    """
    Formats the retrieved documents to include the page content and citation.
    """
    formatted_docs = []
    for doc in docs:
        # Extract page content
        content = doc.page_content

        # Extract source and page number from metadata, with fallbacks
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "N/A")

        # Create a citation string
        citation = f"[Source: {os.path.basename(source)} - Page {page}]"

        # Append citation to the content
        formatted_docs.append(f"{content}\n{citation}")

    return "\n\n".join(formatted_docs)


@cl.on_chat_start
def on_chat_start():
    """
    Initializes the RAG chain when a new chat session starts.
    """
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("chat_history", [])
    cl.Message(content="Kynologist assistant ready. How can I help you?").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages from the user.
    """
    rag_chain = cl.user_session.get("rag_chain")
    chat_history = cl.user_session.get("chat_history", [])

    response_message = cl.Message(content="")
    await response_message.send()

    chain_input = {"question": message.content, "chat_history": chat_history}

    response_content = ""
    async for chunk in rag_chain.astream(chain_input):
        await response_message.stream_token(chunk)
        response_content += chunk

    await response_message.update()

    chat_history.extend(
        [
            HumanMessage(content=message.content),
            AIMessage(content=response_content),
        ]
    )
    cl.user_session.set("chat_history", chat_history)



