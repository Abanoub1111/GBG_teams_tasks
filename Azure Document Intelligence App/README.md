# 📄 Azure Document Intelligence App

### 🚀 End-to-End Intelligent Document Processing using Azure AI (SDK-Based)

---

## 🧠 Overview

This project is a **full-stack AI document processing application** built using **Azure Document Intelligence** and **Streamlit**.

It demonstrates how to:

* Extract structured data from documents (PDFs, images)
* Use Azure **prebuilt models**
* Train **custom AI models**
* Integrate with **Azure Blob Storage**
* Build a real-world AI pipeline using **SDK (not portal)**

---

## 🎯 Why This Project?

Azure provides powerful document processing through its portal, but real-world applications require:

* Automation
* Backend integration
* Scalable pipelines

👉 This project implements everything using **Python SDK**, making it production-ready.

---

## 🏗️ System Architecture

```
User Upload → Streamlit UI → Azure Document Intelligence
                                 ↘ Azure Blob Storage (Training Data)
```

---

## 🧰 Azure Services Used

### 1️⃣ Azure Document Intelligence

A cloud AI service that extracts:

* Text
* Key-value pairs
* Tables
* Structured data

---

### 2️⃣ Azure Blob Storage

Used to:

* Store training documents
* Provide data source for custom models
* Enable scalable document pipelines

---

## 🤖 Model Types Explained

---

### 🔹 1. OCR / Read Model (`prebuilt-read`)

**Purpose:**
Extract raw text from documents

**Use Cases:**

* Digitizing books
* Extracting text from scanned PDFs
* Image-to-text conversion

---

### 🔹 2. Layout Model (`prebuilt-layout`)

**Purpose:**
Understand document structure

**Extracts:**

* Paragraphs
* Tables
* Lines
* Bounding boxes

**Use Cases:**

* Document formatting analysis
* Table extraction
* UI reconstruction

---

### 🔹 3. General Document Model (`prebuilt-document`)

**Purpose:**
Extract structured key-value data

**Use Cases:**

* Forms
* Contracts
* Reports

---

### 🔹 4. Invoice Model (`prebuilt-invoice`)

**Purpose:**
Extract invoice-specific fields

**Extracts:**

* Vendor name
* Invoice ID
* Total amount
* Dates

**Use Cases:**

* Accounting automation
* ERP integration

---

### 🔹 5. Receipt Model (`prebuilt-receipt`)

**Purpose:**
Extract receipt data

**Extracts:**

* Merchant name
* Items
* Total
* Taxes

**Use Cases:**

* Expense tracking
* Financial apps

---

### 🔹 6. Custom Model (Neural)

**Purpose:**
Train your own AI model on custom documents

**How it works:**

1. Upload ≥ 5 documents
2. Store them in Blob Storage
3. Train using SDK
4. Generate model_id
5. Use model for inference

**Use Cases:**

* CV parsing
* Custom forms
* Industry-specific documents

---

## ⚙️ SDK vs Portal (Important Concept)

| Feature             | Azure Portal | SDK (This Project) |
| ------------------- | ------------ | ------------------ |
| Easy UI             | ✅            | ❌                  |
| Automation          | ❌            | ✅                  |
| Production Ready    | ❌            | ✅                  |
| Backend Integration | ❌            | ✅                  |

👉 This project focuses on **SDK-based implementation**, which is required in real-world systems.

---

## 🔄 Workflow

### 🔹 Prebuilt Models Flow

1. Select model
2. Upload file
3. Send request to Azure
4. Receive structured result
5. Display:

   * JSON
   * Table

---

### 🔹 Custom Model Flow

1. Upload ≥ 5 documents
2. Upload to Blob Storage (`training/`)
3. Train model using SDK
4. Get `model_id`
5. Analyze new documents

---

## 📊 Output

The application displays results in two formats:

* 📦 Raw JSON (Azure response)
* 📊 Structured Table (clean view using Pandas)

---

## ⚙️ Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Create `.env`

```env
AZURE_ENDPOINT=your_endpoint
AZURE_KEY=your_key
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_CONTAINER=docs
```

---

### 3️⃣ Run the app

```bash
streamlit run main.py
```

---

## 🔐 Security

* `.env` is ignored using `.gitignore`
* API keys are not exposed
* Uses secure Azure authentication

---

## 🚀 Future Enhancements

* Export results (CSV / Excel)
* Deploy to cloud (Streamlit Cloud / Azure Web App)
* Add authentication system
* Integrate with RAG pipelines
* Add real-time processing

---

## 👨‍💻 Author

**Ebraam Nabil**

* GitHub: https://github.com/EbraamNabil
* LinkedIn: https://www.linkedin.com/in/ebraam-nabil/

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
