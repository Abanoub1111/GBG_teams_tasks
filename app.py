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
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

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


# ------------------- 2. VECTOR DB SETUP -------------------
def process_cvs(uploaded_files, strategy):
    all_chunks = []
    
    # 1. CLEANUP: If there is an existing vectorstore, we must delete its collection
    if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
        try:
            # This deletes the data from the underlying Chroma client
            st.session_state.vectorstore.delete_collection()
        except:
            pass # Handle cases where it's already gone

    for uploaded_file in uploaded_files:
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if strategy == "Document-Aware (Structural)":
            loader = UnstructuredPDFLoader(
                temp_path,
                mode="elements",
                strategy="fast",
                chunking_strategy="by_title",
                max_characters=1000,
                combine_text_under_n_chars=500
            )
            all_chunks.extend(loader.load())
        else:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            all_chunks.extend(text_splitter.split_documents(docs))
            
        os.remove(temp_path)

    #  2. CREATE FRESH: Always create a new collection name or clear the default
    vectorstore = Chroma.from_documents(
        documents=all_chunks, 
        embedding=get_embeddings(),
        # Force a unique collection name per "session set" to avoid merging
        collection_name="cv_collection_" + str(len(uploaded_files))
    )
    return vectorstore, all_chunks


#----------------process full cv---------------
# ---  Function to Process CVs (Whole File approach) ---
def process_cvs_full(directory_path):
    # Load all PDFs (this returns a list where each page is a separate object)
    loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_pages = loader.load()
    
    # Merge pages by filename so one CV = one Document
    combined_content = defaultdict(str)
    for page in raw_pages:
        source_file = page.metadata['source']
        combined_content[source_file] += page.page_content + "\n\n"
    
    # Create the final document list
    final_documents = [
        Document(page_content=text, metadata={"source": source}) 
        for source, text in combined_content.items()
    ]
    
    # Create Vector Store
    vector_store = FAISS.from_documents(final_documents, embeddings)
    return vector_store
