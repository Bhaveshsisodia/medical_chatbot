from flask import Flask, request, jsonify , render_template
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings=download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm_groq = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system' , system_prompt),
        ('human',"{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm_groq, prompt)


rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=["GET",'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response:", response['answer'])
    return str(response['answer'])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)


