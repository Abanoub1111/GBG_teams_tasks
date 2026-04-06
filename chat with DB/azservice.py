import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential
from PyPDF2 import PdfReader
from azure.storage.blob import BlobServiceClient
from io import BytesIO 
 
load_dotenv()
 
 
def get_required_env(var_name):
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value
 
 
SEARCH_ENDPOINT = get_required_env("SEARCH_ENDPOINT")
SEARCH_KEY = get_required_env("SEARCH_KEY")
INDEX_NAME = get_required_env("INDEX_NAME")
 
AZURE_OPENAI_ENDPOINT = get_required_env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = get_required_env("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = get_required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = get_required_env("AZURE_OPENAI_API_VERSION")
BLOB_CONNECTION_STRING = get_required_env("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = get_required_env("BLOB_CONTAINER_NAME")
 
 
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)
 
def get_embedding(text):
    """Generate vector for a given text."""
    response = openai_client.embeddings.create(
        input=[text],
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding
 
def read_pdf(blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    try:
        blob_data = blob_client.download_blob().readall()
        bytes_stream = BytesIO(blob_data)
        pdf_reader = PdfReader(bytes_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
 
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
 
def create_index():
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_KEY)
    )
 
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="my-hnsw")
        ],
        profiles=[
            VectorSearchProfile(name="my-vector-profile", algorithm_configuration_name="my-hnsw")
        ]
    )
 
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="chunk", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
       
        SearchField(
            name="vector_chunk",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-profile"
        )
    ]
 
    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
 
    try:
        index_client.create_index(index)
        print(f"Index '{INDEX_NAME}' created")
    except Exception as e:
        print("Index exists or error:", e)
 
def prepare_documents(chunks, title="my_pdf"):
    docs = []
    print(f"Generating embeddings for {len(chunks)} chunks...")
   
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
       
        docs.append({
            "id": f"doc-{i}",
            "chunk": chunk,
            "title": title,
            "source": "pdf",
            "vector_chunk": vector
        })
    return docs
 
def upload_documents(docs):
    client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    client.upload_documents(documents=docs)
    print(f"Uploaded {len(docs)} documents.")
 
def search_semantic(query_text):
    client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    query_vector = get_embedding(query_text)
   
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=3,
        fields="vector_chunk"
    )
 
    results = client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["title", "chunk"]
    )
 
    print(f"\nResults for: {query_text}")
    for r in results:
        print(f"Score: {r['@search.score']:.4f} | Content: {r['chunk'][:150]}...")
 
if __name__ == "__main__":
    create_index()
 
    raw_text = read_pdf("Abanoub Ibrahim CV.pdf")
    text_chunks = chunk_text(raw_text)
 
    documents = prepare_documents(text_chunks, title="CV_abanoub")
    upload_documents(documents)
 
    search_semantic("What is the candidate's experience with Python and AI?")