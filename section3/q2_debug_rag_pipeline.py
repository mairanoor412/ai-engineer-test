"""
Section 3 — Q2: Debug a Broken LangChain RAG Pipeline
The original code below has 3 bugs. Each bug is marked with a comment
explaining what was wrong and how it was fixed.
"""

# ============================================================
# ORIGINAL BROKEN CODE (with bugs)
# ============================================================
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# BUG 1: Wrong file path — file does not exist at this location
loader = TextLoader("data/documents.txt")
docs = loader.load()

# BUG 2: chunk_overlap is LARGER than chunk_size — this causes an error
# chunk_overlap must always be smaller than chunk_size
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=300)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever()

# BUG 3: retriever.invoke() returns a list of Document objects, not raw strings.
# Passing Document objects directly into the prompt without extracting .page_content
# results in the LLM receiving repr(Document) instead of actual text.
query = "What was the campaign budget?"
results = retriever.invoke(query)

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer based on context."},
        {"role": "user", "content": f"Context: {results}\\nQuestion: {query}"},
    ],
)
print(response.choices[0].message.content)
"""

# ============================================================
# FIXED CODE (all 3 bugs resolved)
# ============================================================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# FIX 1: Corrected the file path to point to an actual existing document.
# ORIGINAL: loader = TextLoader("data/documents.txt")  — file didn't exist
# FIX: Use the correct path where documents are actually stored.
loader = TextLoader("documents/case_study_nike.txt", encoding="utf-8")
docs = loader.load()

# FIX 2: chunk_overlap must be SMALLER than chunk_size.
# ORIGINAL: chunk_size=200, chunk_overlap=300 — overlap > size causes ValueError
# FIX: Set chunk_overlap=50, which is less than chunk_size=200.
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever()

query = "What was the campaign budget?"
results = retriever.invoke(query)

# FIX 3: Extract .page_content from each Document object before passing to LLM.
# ORIGINAL: f"Context: {results}" — passes list of Document objects (shows repr)
# FIX: Join the .page_content of each Document into a single context string.
context = "\n".join([doc.page_content for doc in results])

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Answer based on context only."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {query}"},
    ],
)
print(f"Question: {query}")
print(f"Answer: {response.choices[0].message.content}")


"""
SUMMARY OF 3 BUGS FOUND AND FIXED:

BUG 1 — Wrong File Path
  Problem:  TextLoader("data/documents.txt") — file doesn't exist
  Fix:      TextLoader("documents/case_study_nike.txt") — correct path
  Why:      FileNotFoundError crashes the pipeline before any processing starts

BUG 2 — chunk_overlap > chunk_size
  Problem:  chunk_size=200, chunk_overlap=300 — overlap cannot exceed chunk size
  Fix:      chunk_size=200, chunk_overlap=50
  Why:      Raises ValueError in RecursiveCharacterTextSplitter, prevents splitting

BUG 3 — Document Objects Passed as Context Instead of Text
  Problem:  f"Context: {results}" — results is a list of Document objects
  Fix:      context = "\\n".join([doc.page_content for doc in results])
  Why:      LLM receives Python repr() of objects instead of actual document text,
            making the context unreadable and answers inaccurate
"""
