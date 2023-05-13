import os
import openai
import textwrap
import readline

from typing import List, Tuple
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# InstructorEmbedding
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

load_dotenv()


instructor_embeddings = HuggingFaceInstructEmbeddings(
                            model_name="hkunlp/instructor-xl", 
                            model_kwargs={"device": "cuda" }
)

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.getenv('PGVECTOR_DRIVER'),
    host=os.getenv('PGVECTOR_HOST'),
    port=os.getenv('PGVECTOR_PORT'),
    database=os.getenv('PGVECTOR_DATABASE'),
    user=os.getenv('PGVECTOR_USER'),
    password=os.getenv('PGVECTOR_PASSWORD')
)

    # EUCLIDEAN = EmbeddingStore.embedding.l2_distance
    # COSINE = EmbeddingStore.embedding.cosine_distance
    # MAX_INNER_PRODUCT = EmbeddingStore.embedding.max_inner_product
store = PGVector(
    connection_string=CONNECTION_STRING, 
    embedding_function=instructor_embeddings, 
    collection_name=os.getenv('COLLECTION_NAME'),
    distance_strategy=DistanceStrategy.COSINE
)

retriever = store.as_retriever()

#create the chain to answer questions 
qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=OpenAI(temperature=1, ), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# qa_chain_instrucEmbed = ConversationalRetrievalChain.from_llm(
#                                   llm=OpenAI(temperature=0 ), 
#                                   chain_type="stuff", 
#                                   retriever=retriever, 
#                                   memory=memory,
#                                   return_source_documents=True
# )




## Cite sources



def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(f"{source.metadata['source']}: Page {source.metadata['page']}")




try:
    while True:
        query = input("Enter your query (or 'ctrl-c' to exit): ")
        llm_response = qa_chain_instrucEmbed(query)
        #llm_response = qa_chain_instrucEmbed({"question": query})
        print(f"Query: {query}")
        print("Answer")
        process_llm_response(llm_response)
except KeyboardInterrupt:
    print("\nExiting the program.")

