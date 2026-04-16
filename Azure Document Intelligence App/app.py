import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd

from azure.ai.documentintelligence import (
    DocumentIntelligenceClient,
    DocumentIntelligenceAdministrationClient
)
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import uuid
from datetime import datetime, timedelta

from azure.storage.blob import generate_container_sas, ContainerSasPermissions

# ===================== LOAD ENV =====================
load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_KEY")
conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER")

if not endpoint or not key or not conn_str or not container_name:
    st.error("Missing Azure credentials! Check .env file ❌")
    st.stop()

# ===================== CLIENTS =====================
client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
admin_client = DocumentIntelligenceAdministrationClient(endpoint, AzureKeyCredential(key))
blob_service_client = BlobServiceClient.from_connection_string(conn_str)

# 👇 مهم جدًا (بدون SAS)
account_name = blob_service_client.account_name
container_url = f"https://{account_name}.blob.core.windows.net/{container_name}"


# Create the SAS token
sas_token = generate_container_sas(
    account_name=blob_service_client.account_name,
    container_name=container_name,
    account_key=blob_service_client.credential.account_key,
    permission=ContainerSasPermissions(read=True, list=True),
    expiry=datetime.utcnow() + timedelta(hours=1)
)

# 🔥 Now include the token in the URL
container_url_with_sas = f"https://{account_name}.blob.core.windows.net/{container_name}?{sas_token}"

full_auth_url = f"{container_url}?{sas_token}"

# ===================== UI =====================
st.title("📄 Document Intelligence App")

option = st.selectbox(
    "Choose processing type",
    ["OCR / Read", "Layout Analysis", "General Documents", "Invoices", "Receipts", "Custom Model"]
)

# ===================== NORMAL MODELS =====================
model_map = {
    "OCR / Read": "prebuilt-read",
    "Layout Analysis": "prebuilt-layout",
    "General Documents": "prebuilt-document",
    "Invoices": "prebuilt-invoice",
    "Receipts": "prebuilt-receipt"
}

if option != "Custom Model":

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "png", "jpg"])

    if uploaded_file and st.button("🚀 Process Document"):

        with st.spinner("Processing..."):

            poller = client.begin_analyze_document(
                model_map[option],
                uploaded_file.read()
            )

            result = poller.result()

            data = []
            for page in result.pages:
                for line in page.lines:
                    data.append({"Text": line.content})

            df = pd.DataFrame(data)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📦 Raw JSON")
                st.json(result.as_dict())

            with col2:
                st.subheader("📊 Structured Table")
                st.dataframe(df)

# ===================== CUSTOM MODEL =====================
if option == "Custom Model":

    uploaded_files = st.file_uploader(
        "Upload at least 5 documents",
        #type=["pdf", "png", "jpg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if len(uploaded_files) < 5:
            st.warning(f"Uploaded {len(uploaded_files)} files. Need at least 5.")
        else:
            st.success("Ready to train 🚀")

            if st.button("🔥 Train Model"):

                container_client = blob_service_client.get_container_client(container_name)

                # create container if not exists
                try:
                    container_client.create_container()
                except:
                    pass

                # ================= Upload =================
                for file in uploaded_files:
                    file.seek(0)
                    container_client.upload_blob(
                        name=f"training/{file.name}",
                        data=file.read(),
                        overwrite=True
                    )

                st.success("Files uploaded to Blob Storage ✅")

                # ================= Train =================
                model_id = f"model-{uuid.uuid4()}"

                with st.spinner("Training model..."):

                    poller = admin_client.begin_build_document_model(
                        body={
                            "modelId": model_id,
                            "buildMode": "neural",
                            "azureBlobSource": {
                                "containerUrl": full_auth_url, # ✅ Now has access
                                "prefix": "training/"          # ✅ Matches upload path
                            }
                        },
                        polling_interval=15
                    )

                    # Tell the poller to wait 10 seconds between checks instead of the default
                    model = poller.result()

                st.success("Model trained successfully 🎉")
                st.write("Model ID:", model.model_id)

                # ================= TEST =================
                test_file = st.file_uploader("Upload file to test model", type=["pdf", "png", "jpg"])

                if test_file:

                    with st.spinner("Analyzing..."):

                        poller = client.begin_analyze_document(
                            model.model_id,
                            test_file.read()
                        )

                        result = poller.result()

                        st.subheader("📦 Raw JSON")
                        st.json(result.as_dict())