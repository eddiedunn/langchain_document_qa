from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

from typing import List, Tuple
from dotenv import load_dotenv
import os
import torch

# InstructorEmbedding
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings


# pip install python-dotenv openai langchain InstructorEmbedding pypdf pgvector psycopg2-binary torch torchvision torchaudio sentence_transformers
# load multiple pdfs recursively from a directory into postgres using InstructorEmbeddings.

torch.cuda.empty_cache()
load_dotenv()

loader = DirectoryLoader(
    os.getenv("PDF_ROOT"),
    glob="./*.pdf",
    loader_cls=PyMuPDFLoader,
    recursive=True,
)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=333, chunk_overlap=33)
texts = text_splitter.split_documents(documents)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=os.getenv("EMBEDDINGS_MODEL"), model_kwargs={"device": "cuda"}
)


CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.getenv("PGVECTOR_DRIVER"),
    host=os.getenv("PGVECTOR_HOST"),
    port=os.getenv("PGVECTOR_PORT"),
    database=os.getenv("PGVECTOR_DATABASE"),
    user=os.getenv("PGVECTOR_USER"),
    password=os.getenv("PGVECTOR_PASSWORD"),
)

print("starting to create embeddings")

db = PGVector.from_documents(
    documents=texts,
    embedding=instructor_embeddings,
    collection_name=os.getenv("COLLECTION_NAME"),
    connection_string=CONNECTION_STRING,
)

# Test query
query = "What is the nature of Pure Soul?"
docs_with_score: List[Tuple[Document, float]] = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
