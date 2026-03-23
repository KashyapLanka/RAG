import os
import numpy as np
import pymupdf
import spacy
from typing import List

from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------------
# STEP 1: Extract PDF → text
# -----------------------------
pdf_path = "Docs/review paper.pdf"
output_txt = "docs/data.txt"

doc = pymupdf.open(pdf_path)
out = ""

for page in doc:
    out += page.get_text()

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(out)

print("PDF extracted.")


# -----------------------------
# STEP 2: Load Documents
# -----------------------------
loader = TextLoader(output_txt)
documents = loader.load()


# -----------------------------
# STEP 3: Semantic Chunking
# -----------------------------
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


def adaptive_semantic_chunking(text: str, threshold=0.8, base_chunk_size=5) -> List[Document]:

    sentences = [
        sent.text.strip()
        for sent in nlp(text).sents
        if sent.text.strip()
    ]

    embeddings = model.encode(sentences)

    chunks = []
    start = 0

    while start < len(sentences):
        end = min(start + base_chunk_size, len(sentences))

        while end < len(sentences):
            sim = np.dot(
                embeddings[end - 1], embeddings[end - 2]
            ) / (
                np.linalg.norm(embeddings[end - 1]) *
                np.linalg.norm(embeddings[end - 2])
            )

            if sim < threshold:
                break

            end += 1

        chunk_text = " ".join(sentences[start:end])
        chunks.append(Document(page_content=chunk_text))

        start = end

    return chunks


# Apply chunking to all documents
all_chunks = []

for doc in documents:
    chunks = adaptive_semantic_chunking(doc.page_content)
    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")


# -----------------------------
# STEP 4: Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# STEP 5: Vector DB (Chroma)
# -----------------------------
vector_db = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_db.persist()

print("Vector DB created and saved.")


# -----------------------------
# STEP 6: Retriever
# -----------------------------
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


# -----------------------------
# STEP 7: Connect LM Studio
# -----------------------------
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0
)


# -----------------------------
# STEP 8: Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context.

Context:
{context}

Question:
{question}
""")


# -----------------------------
# STEP 9: Format Docs
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# STEP 10: RAG Pipeline (LCEL)
# -----------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# STEP 11: Query
# -----------------------------
query = "What is RAG?"

response = rag_chain.invoke(query)

print("\nAnswer:")
print(response)