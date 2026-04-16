import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import json

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

account_name = blob_service_client.account_name
container_url = f"https://{account_name}.blob.core.windows.net/{container_name}"

# ===================== SAS TOKEN =====================
sas_token = generate_container_sas(
    account_name=blob_service_client.account_name,
    container_name=container_name,
    account_key=blob_service_client.credential.account_key,
    permission=ContainerSasPermissions(read=True, list=True),
    expiry=datetime.utcnow() + timedelta(hours=1)
)

full_auth_url = f"{container_url}?{sas_token}"

# ===================== HELPER =====================
def convert_to_azure_format(labeled_data):
    fields = {}

    for item in labeled_data:
        label = item["label"]
        text = item["text"]

        if label not in fields:
            fields[label] = {"content": text}
        else:
            fields[label]["content"] += f" {text}"

    return {
        "documents": [
            {
                "fields": fields
            }
        ]
    }

# ===================== UI =====================
st.title("📄 Document Intelligence App")

if "model_id" in st.session_state:
    st.info(f"Using Model ID: {st.session_state['model_id']}")

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

            lines = []
            for page in result.pages:
                for line in page.lines:
                    lines.append(line.content)

            st.success("Document processed ✅")

            # ===== LABELING =====
            st.subheader("📝 Label the extracted text")

            label_options = [
                "None",
                "invoice_id",
                "total_amount",
                "invoice_date",
                "vendor_name",
                "other"
            ]

            labeled_data = []

            for i, line in enumerate(lines):
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.write(line)

                with col2:
                    label = st.selectbox(
                        "Label",
                        label_options,
                        key=f"label_{i}"
                    )

                    if label != "None":
                        labeled_data.append({
                            "text": line,
                            "label": label
                        })

            # ===== SAVE + UPLOAD =====
            if st.button("💾 Save & Upload Labels"):

                if labeled_data:

                    azure_format = convert_to_azure_format(labeled_data)
                    json_filename = f"{uploaded_file.name}.labels.json"

                    with open(json_filename, "w") as f:
                        json.dump(azure_format, f, indent=2)

                    container_client = blob_service_client.get_container_client(container_name)

                    try:
                        container_client.create_container()
                    except:
                        pass

                    uploaded_file.seek(0)
                    container_client.upload_blob(
                        name=f"training/{uploaded_file.name}",
                        data=uploaded_file.read(),
                        overwrite=True
                    )

                    container_client.upload_blob(
                        name=f"training/{json_filename}",
                        data=json.dumps(azure_format),
                        overwrite=True
                    )

                    st.success("File + Labels uploaded ✅")
                    st.json(azure_format)

                else:
                    st.warning("No labels selected")

            # ===== DISPLAY =====
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📦 Raw JSON")
                st.json(result.as_dict())

            with col2:
                df = pd.DataFrame({"Text": lines})
                st.subheader("📊 Extracted Text")
                st.dataframe(df)

# ===================== CUSTOM MODEL =====================
if option == "Custom Model":

    mode = st.radio("Mode", ["Train Model", "Test Model"])

    # ================= TRAIN =================
    if mode == "Train Model":

        uploaded_files = st.file_uploader(
            "Upload at least 5 documents",
            accept_multiple_files=True,
            key="train_upload"
        )

        if uploaded_files:

            if len(uploaded_files) < 5:
                st.warning(f"Uploaded {len(uploaded_files)} files. Need at least 5.")
            else:
                st.success("Ready to train 🚀")

                if st.button("🔥 Train Model"):

                    container_client = blob_service_client.get_container_client(container_name)

                    try:
                        container_client.create_container()
                    except:
                        pass

                    for file in uploaded_files:
                        file.seek(0)
                        container_client.upload_blob(
                            name=f"training/{file.name}",
                            data=file.read(),
                            overwrite=True
                        )

                    st.success("Files uploaded ✅")

                    model_id = f"model-{uuid.uuid4()}"

                    with st.spinner("Training model..."):

                        poller = admin_client.begin_build_document_model(
                            body={
                                "modelId": model_id,
                                "buildMode": "template",
                                "azureBlobSource": {
                                    "containerUrl": full_auth_url,
                                    "prefix": "training/"
                                }
                            },
                            polling_interval=15
                        )

                        model = poller.result()

                    st.session_state["model_id"] = model.model_id
                    st.success(f"Model trained 🎉 ID: {model.model_id}")

    # ================= TEST =================
    if mode == "Test Model":

        if "model_id" not in st.session_state:
            st.warning("⚠️ Train a model first")
        else:
            st.success(f"Using Model: {st.session_state['model_id']}")

            test_file = st.file_uploader(
                "Upload file to test model",
                type=["pdf", "png", "jpg"],
                key="test_upload"
            )

            if test_file and st.button("🔍 Run Inference"):

                with st.spinner("Analyzing..."):

                    poller = client.begin_analyze_document(
                        st.session_state["model_id"],
                        test_file.read()
                    )

                    result = poller.result()

                    st.subheader("📦 Raw JSON")
                    st.json(result.as_dict())
