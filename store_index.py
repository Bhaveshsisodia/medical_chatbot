from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_docs = load_pdf_files("data/")
minimal_docs = filter_to_minimal_docs(extracted_docs)
split_docs = text_splitter(minimal_docs)
embeddings = download_embeddings()

pinecone_api_key=PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)




index_name = "medical-chatbot"

pc.create_index(
    name=index_name,
    dimension=384,  # e.g., OpenAI embedding size
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    }
)

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=index_name
)