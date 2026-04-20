import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import json
import uuid
import time
import io
import base64
from datetime import datetime, timedelta

# Azure & Image Processing
from azure.ai.documentintelligence import DocumentIntelligenceClient, DocumentIntelligenceAdministrationClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, generate_container_sas, ContainerSasPermissions
from azure.core.exceptions import HttpResponseError

from PIL import Image
from pdf2image import convert_from_bytes
from streamlit_drawable_canvas import st_canvas

# ===================== LOAD ENV & CLIENTS =====================
load_dotenv()
endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_KEY")
conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER")

@st.cache_resource
def get_clients():
    di_client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    adm_client = DocumentIntelligenceAdministrationClient(endpoint, AzureKeyCredential(key))
    b_client = BlobServiceClient.from_connection_string(conn_str)
    return di_client, adm_client, b_client

client, admin_client, blob_service_client = get_clients()

# --- ADD THIS BLOCK TO GENERATE THE URL ---
account_name = blob_service_client.account_name
container_url = f"https://{account_name}.blob.core.windows.net/{container_name}"

# Create a SAS token that expires in 2 hours
sas_token = generate_container_sas(
    account_name=account_name,
    container_name=container_name,
    account_key=blob_service_client.credential.account_key,
    permission=ContainerSasPermissions(read=True, list=True, write=True, delete=True),
    expiry=datetime.utcnow() + timedelta(hours=2)
)
full_auth_url = f"{container_url}?{sas_token}"

# ===================== HELPER FUNCTIONS =====================
def get_image_base64(img):
    """Converts a PIL image to a Base64 string for the canvas component."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def safe_analyze(model_id, document_data, file_name="unknown"):
    try:
        poller = client.begin_analyze_document(model_id, document_data, polling_interval=10)
        time.sleep(5) 
        result = poller.result()
        
        # --- TRACK USAGE ---
        log_usage(model_id, file_name, "Success")
        # -------------------
        
        return result
    except Exception as e:
        log_usage(model_id, file_name, f"Error: {str(e)[:50]}")
        st.error(f"Azure Error: {e}")
        return None


# ===================== USAGE TRACKING =====================
def log_usage(model_id, document_name, status="Success"):
    """Tracks model calls in st.session_state and a local CSV"""
    usage_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model ID": model_id,
        "Document": document_name,
        "Status": status
    }
    
    # 1. Track in Session State (for this view)
    if "usage_history" not in st.session_state:
        st.session_state.usage_history = []
    st.session_state.usage_history.append(usage_entry)
    
    # 2. Track to CSV (Permanent local log)
    log_file = "model_usage_log.csv"
    df = pd.DataFrame([usage_entry])
    df.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))


# ===================== UI SETUP =====================
st.set_page_config(layout="wide", page_title="Azure Doc AI")
st.title("📄 Document Intelligence App")

option = st.selectbox("Choose processing type", ["OCR / Read", "Layout Analysis", "Invoices", "Receipts", "Custom Model"])

model_map = {
    "OCR / Read": "prebuilt-read",
    "Layout Analysis": "prebuilt-layout",
    "Invoices": "prebuilt-invoice",
    "Receipts": "prebuilt-receipt"
}

# ===================== PREBUILT MODELS =====================
if option != "Custom Model":
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "png", "jpg"])

    if uploaded_file and st.button("🚀 Process Document"):
        with st.spinner("Analyzing..."):
            result = safe_analyze(model_map[option], uploaded_file.getvalue())

            if result:
                # ---------------- OCR / READ VIEW ----------------
                if option == "OCR / Read":
                    data = [{"Type": "Line", "Content": line.content} for page in result.pages for line in page.lines]
                    st.table(pd.DataFrame(data))

                # ---------------- LAYOUT VIEW (Modern) ----------------
                elif option == "Layout Analysis":
                    tab1, tab2, tab3, tab4 = st.tabs(["📝 Full Content", "📊 Tables", "✅ Selection Marks", "🖼️ Figures"])
                    
                    with tab1:
                        st.text_area("Raw Text Content", value=result.content, height=500)
                    
                    with tab2:
                        if not result.tables:
                            st.info("No tables detected.")
                        else:
                            for i, table in enumerate(result.tables):
                                st.markdown(f"#### Table {i+1} ({table.row_count}x{table.column_count})")
                                table_data = {}
                                for cell in table.cells:
                                    if cell.row_index not in table_data: table_data[cell.row_index] = {}
                                    table_data[cell.row_index][cell.column_index] = cell.content
                                
                                rows = sorted(table_data.keys())
                                table_rows = [[table_data[r].get(c, "") for c in range(table.column_count)] for r in rows]
                                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                    with tab3:
                        marks = []
                        for page in result.pages:
                            for mark in (page.selection_marks or []):
                                marks.append({"Page": page.page_number, "State": mark.state, "Confidence": f"{mark.confidence*100:.1f}%"})
                        if marks: st.table(pd.DataFrame(marks))
                        else: st.info("No selection marks.")

                    with tab4:
                        if hasattr(result, "figures") and result.figures:
                            st.write(f"Detected {len(result.figures)} figures.")
                        else: st.info("No figures detected.")

                # ---------------- INVOICE / RECEIPT VIEW ----------------
                elif option in ["Invoices", "Receipts"]:
                    data = []
                    if result.documents:
                        for doc in result.documents:
                            for f_name, field in doc.fields.items():
                                val = field.get('valueString') or field.get('valueDate') or \
                                      field.get('valueCurrency') or field.get('content')
                                data.append({"Field": f_name, "Value": str(val), "Confidence": f"{field.get('confidence', 0)*100:.1f}%"})
                    
                    st.subheader("Extracted Fields")
                    st.table(pd.DataFrame(data))
                    with st.expander("Show Raw Content"):
                        st.write(result.content)

                # Always show JSON at bottom
                with st.expander("🛠️ View Developer Raw JSON"):
                    st.json(result.as_dict())

# ===================== CUSTOM MODEL (FIXED CANVAS) =====================
if option == "Custom Model":
    st.header("🛠️ Custom Document Labeling")
    
    # 1. Configuration
    label_input = st.text_input("Define Labels", "Vendor, Total, Date, Description")
    target_labels = [l.strip() for l in label_input.split(",")]
    
    uploaded_files = st.file_uploader("Upload training docs", accept_multiple_files=True)

    if uploaded_files:
        if "annotations" not in st.session_state: 
            st.session_state.annotations = {}
        
        file_name = st.selectbox("Select file to label", [f.name for f in uploaded_files])
        current_file = next(f for f in uploaded_files if f.name == file_name)
        
        # --- PDF/Image Display ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Document View")
            if "pdf" in current_file.type:
                # Convert PDF to image for display
                images = convert_from_bytes(current_file.getvalue(), first_page=1, last_page=1)
                display_img = images[0]
            else:
                display_img = Image.open(io.BytesIO(current_file.getvalue()))
            
            st.image(display_img, use_container_width=True)

        with col2:
            st.subheader("🏷️ Labeling Tool")
            
            # Button to trigger OCR for this specific file
            if st.button(f"🔍 Extract Words from {file_name}"):
                with st.spinner("Azure is reading the document..."):
                    result = safe_analyze("prebuilt-read", current_file.getvalue())
                    if result:
                        # Get every word found in the document
                        words = [word.content for page in result.pages for word in page.words]
                        st.session_state[f"words_{file_name}"] = words
                        st.success(f"Found {len(words)} words!")

            # If words are extracted, show the selection tool
            if f"words_{file_name}" in st.session_state:
                all_words = st.session_state[f"words_{file_name}"]
                
                # Multi-select mimics "drawing a box" by letting you click specific words
                selected_words = st.multiselect(
                    "Click the words that belong to the label:",
                    options=all_words,
                    key=f"words_sel_{file_name}"
                )
                
                chosen_label = st.selectbox("Assign to Label:", target_labels, key=f"label_sel_{file_name}")
                
                if st.button("➕ Save Annotation"):
                    if selected_words:
                        if file_name not in st.session_state.annotations:
                            st.session_state.annotations[file_name] = []
                        
                        combined_text = " ".join(selected_words)
                        st.session_state.annotations[file_name].append({
                            "label": chosen_label,
                            "content": combined_text
                        })
                        st.toast(f"Saved: {chosen_label} -> {combined_text}")
                    else:
                        st.warning("Select words first!")

            # Show the "Labels Table" for this file
            if file_name in st.session_state.annotations:
                st.divider()
                st.write("**Current Labels for this file:**")
                st.table(pd.DataFrame(st.session_state.annotations[file_name]))
                
                if st.button("🗑️ Clear All"):
                    st.session_state.annotations[file_name] = []
                    st.rerun()

    # --- Training Section ---
        st.divider()
        if st.button("🚀 Train Model"):
            if len(st.session_state.annotations) < 5:
                st.error("Azure requires at least 5 annotated documents to train a custom model.")
            else:
                with st.spinner("Uploading data and starting Azure Training..."):
                    try:
                        container_client = blob_service_client.get_container_client(container_name)
                        
                        # 1. Upload Files and Labels to Blob
                        for f in uploaded_files:
                            # Upload the actual document
                            f.seek(0)
                            blob_path = f"training/{f.name}"
                            container_client.upload_blob(name=blob_path, data=f.read(), overwrite=True)
                            
                            # Upload the labels.json if it exists for this file
                            if f.name in st.session_state.annotations:
                                # Format required by Azure Document Intelligence
                                label_data = {
                                    "$schema": "https://schema.cognitiveservices.azure.com/formrecognizer/2021-03-01/labels.json",
                                    "document": f.name,
                                    "labels": [
                                        {"label": a["label"], "value": [{"text": a["content"]}]} 
                                        for a in st.session_state.annotations[f.name]
                                    ]
                                }
                                label_blob_name = f"training/{f.name}.labels.json"
                                container_client.upload_blob(name=label_blob_name, data=json.dumps(label_data), overwrite=True)

                        # 2. Trigger the Build
                        
                        model_id = f"model-{uuid.uuid4().hex[:6]}"

                        # We create the request body according to the latest SDK specs
                        build_request = {
                            "modelId": model_id,
                            "description": "Custom model trained via Streamlit",
                            "buildMode": "template",  # Use "template" for F0/Free tier, "neural" for S0
                            "azureBlobSource": {
                                "containerUrl": full_auth_url,
                                "prefix": "training/"
                            }
                        }

                        try:
                            # Pass the dictionary directly to the poller
                            poller = admin_client.begin_build_document_model(build_request)
                            
                            st.success(f"✅ Training Started! Model ID: **{model_id}**")
                            st.info("The training process is now running on Azure. Check the 'Custom Models' tab in Azure AI Studio for progress.")
                        except Exception as build_error:
                            st.error(f"Azure API rejected the build request: {build_error}")

                    except Exception as e:
                        st.error(f"Failed to start training: {e}")



# ===================== ADD THE TESTING SECTION HERE =====================
        st.divider()
        st.header("🧪 Test Your Custom Model")
        st.info("Wait ~5-10 mins after training before testing. Check Azure Studio for 'Succeeded' status.")

        test_col1, test_col2 = st.columns([1, 1])

        with test_col1:
            test_model_id = st.text_input("Enter Model ID to test", value="model-8d4f7f")
            test_file = st.file_uploader("Upload a new document to test", type=["pdf", "png", "jpg"], key="test_upload")
            run_test = st.button("🔍 Run Custom Analysis")

        if test_file and run_test:
            with st.spinner(f"Analyzing with model {test_model_id}..."):
                # Use the same safe_analyze function but pass the new Model ID
                result = safe_analyze(test_model_id, test_file.getvalue())
                
                if result and hasattr(result, 'documents') and result.documents:
                    with test_col2:
                        st.success("Analysis Complete!")
                        
                        # Extract fields from the custom model result
                        custom_data = []
                        for doc in result.documents:
                            for field_name, field in doc.fields.items():
                                custom_data.append({
                                    "Label": field_name,
                                    "Value": field.get('content') or field.get('valueString'),
                                    "Confidence": f"{field.get('confidence', 0)*100:.1f}%"
                                })
                        
                        st.table(pd.DataFrame(custom_data))
                else:
                    st.error("No fields were extracted. Check if the model is still 'Running' in Azure Studio.")
        # =========================================================================

# ===================== USAGE DASHBOARD =====================
st.sidebar.divider()
if st.sidebar.checkbox("📊 Show Usage Analytics"):
    st.header("📈 Model Usage Analytics")
    
    if os.path.exists("model_usage_log.csv"):
        usage_df = pd.read_csv("model_usage_log.csv")
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Calls", len(usage_df))
        m2.metric("Unique Models", usage_df["Model ID"].nunique())
        m3.metric("Success Rate", f"{(usage_df['Status'] == 'Success').mean()*100:.1f}%")
        
        # # Usage Chart
        # st.subheader("Calls Over Time")
        # usage_df['Timestamp'] = pd.to_datetime(usage_df['Timestamp'])
        # chart_data = usage_df.resample('H', on='Timestamp').count()["Model ID"]
        # st.line_chart(chart_data)
        
        # Raw Data
        with st.expander("View Detailed Logs"):
            st.dataframe(usage_df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No usage data recorded yet. Run an analysis to start tracking.")
