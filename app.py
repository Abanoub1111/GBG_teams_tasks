import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import os
from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_version="2024-12-01-preview",
#     azure_endpoint="https://gbgacademy-genai.openai.azure.com/",
#     api_key=os.getenv("openai_api"),
# )

load_dotenv()

# API Keys
groq_api = os.getenv("groq_api")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("langchain_api")
os.environ["LANGCHAIN_PROJECT"] = "CV_RAG_Testing"

st.set_page_config(page_title="CV RAG Explorer", layout="wide")
st.title("Chat with 5-CV with multiple RAG config")

# ------------------- 1. LLM & EMBEDDINGS -------------------
@st.cache_resource
# def get_llm():
#     return ChatGroq(
#         model="llama-3.3-70b-versatile", 
#         api_key=groq_api,
#         temperature=0
#     )
@st.cache_resource
def get_llm():
    return AzureChatOpenAI(
        azure_endpoint="https://gbgacademy-genai.openai.azure.com/",
        api_key=os.getenv("openai_api"),
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1-nano",   # your Azure deployment name
        temperature=0
    )


# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
@st.cache_resource
def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint="https://gbgacademy-genai.openai.azure.com/",
        api_key=os.getenv("openai_api"),
        api_version="2024-12-01-preview",
        model="text-embedding-3-small",
        azure_deployment="text-embedding-3-small"  # your Azure deployment name
    )


