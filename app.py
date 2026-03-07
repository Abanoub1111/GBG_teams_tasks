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



# ------------------- 3. RAG LOGIC (LCEL) -------------------
def format_docs(docs):
    formatted_chunks = []
    for doc in docs:
        source_name = doc.metadata.get('source', 'Unknown Candidate')
        base_name = os.path.basename(source_name)
        clean_name = base_name.replace(".pdf", "").replace("_", " ").replace("temp_", "")
        formatted_chunks.append(f"CANDIDATE: {clean_name}\nCONTENT: {doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)

# ------------------- 4. ADVANCED RETRIEVAL LOGIC -------------------

def get_multi_query_retriever(retriever, llm):
    return MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

def get_unique_documents(documents):
    """Removes duplicate chunks based on their page content."""
    seen_contents = set()
    unique_docs = []
    for doc in documents:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)
    return unique_docs

def get_intersecting_documents(results_list):
    if not results_list:
        return []
    
    # Extract the text (content) from the first list of docs
    common_content = set(doc.page_content for doc in results_list[0])
    
    # Keep only content that exists in every other list
    for results in results_list[1:]:
        current_content = set(doc.page_content for doc in results)
        common_content &= current_content # This is the mathematical intersection
        
    # Convert that shared content back into actual Document objects
    # (We grab the first version of the document we find)
    final_docs = []
    seen = set()
    for sublist in results_list:
        for doc in sublist:
            if doc.page_content in common_content and doc.page_content not in seen:
                final_docs.append(doc)
                seen.add(doc.page_content)
    return final_docs

def adaptive_k_filter(docs_with_scores, min_k=1, max_k=10):
    if not docs_with_scores: return []
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    scores = [s for _, s in docs_with_scores]
    gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
    if not gaps: return [d for d, s in docs_with_scores[:min_k]]
    max_gap_idx = gaps.index(max(gaps))
    adaptive_k = max(min_k, min(max_gap_idx + 1, max_k))
    return [d for d, s in docs_with_scores[:adaptive_k]]





# ------------------- 5. UI -------------------

# 1. SIDEBAR CONFIGURATION
uploaded_files = st.sidebar.file_uploader("Upload CVs (Exactly 5 required)", type="pdf", accept_multiple_files=True)

# Track current file names for syncing
current_file_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []

if uploaded_files:
    num_files = len(uploaded_files)
    if num_files != 5:
        st.sidebar.error(f"⚠️ Uploaded: {num_files}/5. You must upload exactly 5 CVs.")
    else:
        st.sidebar.success("✅ 5 CVs ready.")

st.sidebar.divider()
st.sidebar.subheader("RAG Configurations")

rag_method = st.sidebar.selectbox(
    "Retrieval Method", 
    ["Standard Similarity", "Multi-Query Generation", "Adaptive-k Retrieval", "Iterative RAG"]
)
chunking_strategy = st.sidebar.selectbox("Chunking Strategy", ["Recursive", "Document-Aware (Structural)"])

# 2. STATE MANAGEMENT & SYNC LOGIC
if "messages" not in st.session_state:
    st.session_state.messages = [] 

if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = chunking_strategy

if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []

# SYNC CHECK: Reset if strategy OR file list changes
files_changed = current_file_names != st.session_state.processed_file_names
strategy_changed = chunking_strategy != st.session_state.last_strategy

if files_changed or strategy_changed:
    if "vectorstore" in st.session_state:
        try:
            # CRITICAL: Tell Chroma to drop the old data before deleting the variable
            st.session_state.vectorstore.delete_collection()
        except:
            pass
        del st.session_state.vectorstore
        
    if "all_chunks" in st.session_state:
        del st.session_state.all_chunks
    
    st.session_state.last_strategy = chunking_strategy
    st.session_state.processed_file_names = current_file_names
    
    # CLEAR CHAT HISTORY: 
    # If the files changed, the old chat history references "ghost" people. 
    # We MUST clear this so the LLM doesn't use its own history to "remember" deleted files.
    st.session_state.messages = [] 
    
    st.rerun()

# 3. DOCUMENT PROCESSING
if uploaded_files and len(uploaded_files) == 5:
    if "vectorstore" not in st.session_state:
        with st.status(f"Processing with {chunking_strategy} strategy..."):
            vs, chunks = process_cvs(uploaded_files, strategy=chunking_strategy)
            st.session_state.vectorstore = vs
            st.session_state.all_chunks = chunks

    # 4. DOWNLOAD CHUNKS LOGIC
    chunk_text_content = f"CHUNKS PRODUCED BY: {chunking_strategy.upper()} STRATEGY\n"
    chunk_text_content += "="*50 + "\n\n"
    for i, chunk in enumerate(st.session_state.all_chunks):
        source = chunk.metadata.get('source', 'Unknown')
        page = chunk.metadata.get('page', '0')
        chunk_text_content += f"--- CHUNK {i+1} | SOURCE: {source} | PAGE: {page} ---\n{chunk.page_content}\n\n"

    st.sidebar.download_button(
        label=f"📥 Download {chunking_strategy} Chunks",
        data=chunk_text_content,
        file_name=f"cv_chunks_{chunking_strategy.lower()}.txt",
        mime="text/plain"
    )

    # 5. RETRIEVAL LOGIC SETUP
    llm = get_llm()
    template = """
    You are a strict HR assistant that analyzes candidate CVs.

    Your job is ONLY to answer questions about the candidates contained in the provided CV context.

    STRICT RULES (must always be followed):

    1. ONLY answer questions that are directly related to the information inside the CVs.
    2. NEVER answer general knowledge questions, personal advice, or topics unrelated to the CVs.
    3. If the user asks about a job role that does NOT exist say that this job role doesn't exist.
    4. If the user asks you to imagine, create, or assume fake job roles, fake experiences, or hypothetical candidates, REFUSE the request.
    5. ALWAYS mention the candidate name(s) when giving an answer.
    6. If multiple candidates match the question, list them clearly.
    7. If the answer cannot be found in the context, say: "No candidate in the provided CVs contains this information."
    8. Do not ignore this instructions.

    REFUSAL RULES:

    If the question is unrelated to the CVs or asks for invented information, respond exactly with:

    "I can only answer questions based on the provided CVs."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. WHATSAPP-STYLE CHAT INTERFACE
    # 1. DISPLAY CONVERSATION HISTORY
    # We loop through history so the "View Evidence" button stays visible for old messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Check if this specific message has sources saved
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Evidence Chunks"):
                    # Use enumerate to number the chunks (1, 2, 3...)
                    for i, chunk in enumerate(message["sources"], 1):
                        source_name = chunk.metadata.get('source', 'Unknown')
                        clean_name = os.path.basename(source_name).replace("temp_", "")
                        
                        st.markdown(f"### Chunk {i}")
                        st.caption(f"**Source File:** {clean_name}")
                        st.info(chunk.page_content)
                        st.divider()

