from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings




def load_pdf_files(directory):
    # Load all PDF files from the specified directory
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents



def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    minimal_docs : List[Document] = []
    for doc in documents:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content,
                                      metadata={"source": src})
                                    )
    return minimal_docs


def text_splitter(minimal_docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(minimal_docs)
    return split_docs



def download_embeddings() -> HuggingFaceEmbeddings:
    """
    Download the HuggingFace embeddings model.

    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings