"""
Task 2.3 — RAG-Based Campaign Knowledge Bot
A Streamlit chatbot that answers questions ONLY from provided agency documents.
Uses LangChain + FAISS for retrieval, Groq LLM for generation.

Documents: case studies and brand guidelines in the 'documents/' folder.
Run: streamlit run rag_chatbot.py
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI

load_dotenv()

# --- Configuration ---
DOCUMENTS_FOLDER = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight embedding model
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Groq LLM Client ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def load_documents(folder_path: str):
    """Loads all .txt and .pdf files from the documents folder."""
    documents = []
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in txt_files:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        # Add source filename to metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        documents.extend(docs)

    print(f"Loaded {len(documents)} documents from '{folder_path}/'")
    return documents


def create_vector_store(documents):
    """Splits documents into chunks and creates FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Use HuggingFace embeddings (free, no API key needed)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"FAISS vector store created with {len(chunks)} vectors")

    return vector_store


def query_llm(system_prompt: str, user_prompt: str) -> str:
    """Calls Groq LLM for answer generation."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,  # Low temperature for factual accuracy
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """You are a knowledge assistant for an advertising agency.
You answer questions ONLY based on the provided document context.

STRICT RULES:
1. ONLY use information from the provided context to answer questions
2. If the answer is NOT in the context, say: "I don't have information about that in the provided documents."
3. Always cite the source document name where you found the answer
4. Include a relevant direct quote from the document when possible
5. Be concise and accurate — do not make up or infer information beyond what is stated
6. Do not use any general knowledge — only the provided documents

RESPONSE FORMAT:
Answer: [Your answer based on the documents]
Source: [Document filename]
Quote: "[Relevant quote from the document]"
"""


def answer_question(question: str, vector_store) -> str:
    """Retrieves relevant chunks and generates an answer."""
    # Retrieve top 4 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "I don't have information about that in the provided documents."

    # Build context from retrieved chunks
    context_parts = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Based on the following document context, answer the question.

CONTEXT:
{context}

QUESTION: {question}"""

    return query_llm(SYSTEM_PROMPT, user_prompt)


# --- Initialize RAG System ---
print("=" * 60)
print("RAG CAMPAIGN KNOWLEDGE BOT")
print("=" * 60)
print("\nInitializing...")

documents = load_documents(DOCUMENTS_FOLDER)
vector_store = create_vector_store(documents)

print("\nReady! Loading interface...\n")


# --- Streamlit UI ---
try:
    import streamlit as st

    st.set_page_config(page_title="Campaign Knowledge Bot", page_icon="📚", layout="wide")
    st.title("📚 Campaign Knowledge Bot")
    st.caption("Ask questions about agency case studies and brand guidelines. Answers come only from provided documents.")

    # Show loaded documents in sidebar
    st.sidebar.header("Loaded Documents")
    for doc in documents:
        st.sidebar.markdown(f"- {doc.metadata['source']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**How it works:**")
    st.sidebar.markdown("1. Your question is matched against document chunks")
    st.sidebar.markdown("2. Most relevant sections are retrieved")
    st.sidebar.markdown("3. LLM generates answer ONLY from those sections")
    st.sidebar.markdown("4. Source document and quote are cited")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about campaigns, case studies, or brand guidelines..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = answer_question(prompt, vector_store)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

except ImportError:
    # Fallback to CLI if Streamlit not available
    print("Streamlit not found. Running in CLI mode.\n")
    print("Type your questions below. Type 'quit' to exit.\n")
    print("-" * 60)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        print("\nSearching documents...")
        answer = answer_question(question, vector_store)
        print(f"\nBot: {answer}")
        print("-" * 60)
