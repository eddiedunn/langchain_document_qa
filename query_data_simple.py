from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
import os


PERSIST_DIRECTORY = "./db-simple"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})

    # load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
)
retriever = db.as_retriever()

# Load up your LLM
#model="gpt-3.5-turbo-0613"
model="gpt-4-0613"
llm = ChatOpenAI(model_name=model, openai_api_key=os.getenv("OPENAI_API_KEY"))

qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever ,
                                return_source_documents=True
                                )

# Start the loop
while True:
    # Ask a question
    query = input("Ask a question (or type 'quit' to exit): ")
    
    # Check if user wants to quit
    if query.lower() == "quit" or query.lower() == "exit":
        break
    
    # Ask the question to the QA model
    result = qa({"query": query})
    
    # Print the result
    result_text = result['result']
    source_documents = result['source_documents']
    print("Answer:")
    print(result_text)
#    print('----------------------------------------------------------------------------------------------------------------')
#    print("Source Documents:")
#    print(source_documents)
#    print()
