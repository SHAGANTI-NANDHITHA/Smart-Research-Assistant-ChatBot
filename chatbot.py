from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import streamlit as st

# model and hugging face embeddings API keys
GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')
HF_API_KEY= os.getenv('HF_API_KEY')

# loading model and embeddings
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# loading pdf
loader = UnstructuredPDFLoader("BTP_TinyML_IEEE_CEM_New.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# Create Vector db
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local('faiss_db')

# load Vector db
# vectorstore = FAISS.load_local("faiss_db", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

template = """Answer the question based only on the following context:
{context}

Question: {question}


"""
prompt = ChatPromptTemplate.from_template(template)


retrieval_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

def reply_llm(msg):
    result = retrieval_chain.invoke({"question": msg})
    return result 




