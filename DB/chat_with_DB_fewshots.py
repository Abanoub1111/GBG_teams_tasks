##################### CHAIN ARCHETICTURE ###############################

import streamlit as st
import pandas as pd
import json
import os
import re
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


# ------------------- CONFIG -------------------

DB_URL = os.getenv("DB_URL")

st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with Postgres DB :bar_chart:")

# ------------------- RAG / FEW-SHOT SETUP -------------------

@st.cache_resource
def get_retriever():
    # 1. Load your fewshots.json
    with open("fewshots.json", "r") as f:
        few_shots = json.load(f)
    
    # 2. Prepare documents (We embed the natural question)
    docs = [
        Document(
            page_content=ex["naturalQuestion"],
            metadata={"sql": ex["sqlQuery"]}
        ) for ex in few_shots
    ]
    
    # 3. Create Vector Store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 3}) # Get top 3 examples

def get_few_shot_context(question):
    retriever = get_retriever()
    relevant_docs = retriever.invoke(question)
    
    context = "Here are some examples of how to translate questions to SQL for this database:\n\n"
    for doc in relevant_docs:
        context += f"Question: {doc.page_content}\nSQL: {doc.metadata['sql']}\n\n"
    return context

# ------------------- DATABASE -------------------
@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

@st.cache_data
def get_schema():
    engine = get_db_engine()
    inspector_query = text("""
        SELECT table_name, column_name 
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)

    schema_string = ""
    with engine.connect() as conn:
        result = conn.execute(inspector_query)
        current_table = None
        for row in result:
            table_name, column_name = row
            if table_name != current_table:
                if current_table is not None:
                    schema_string += "\n"
                schema_string += f"Table: {table_name}\n"
                current_table = table_name
            schema_string += f"  - {column_name}\n"

    return schema_string

# ------------------- LLM -------------------
# @st.cache_resource
# def get_llm():
#     return ChatGroq(
#         model="meta-llama/llama-4-maverick-17b-128e-instruct",
#         api_key=groq_api,
#         temperature = 0
#     )
@st.cache_resource
def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0
    )

@st.cache_resource
def get_embeddings():
    return AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    
@st.cache_resource
def get_sql_chain(llm):
    # Added {few_shot_examples} to the template
    sql_prompt = PromptTemplate.from_template("""
You are an expert PostgreSQL data analyst.

Database schema:
{schema}

{few_shot_examples}

User Question:
{question}

Write ONLY the SQL query. 
- Use the examples above to understand naming conventions (e.g., if the user says 'United States', check if the examples use 'USA').
- Use double quotes around table and column names.
- Return only valid SELECT SQL.
""")
    return sql_prompt | llm

@st.cache_resource
def get_answer_chain(llm):
    answer_prompt = PromptTemplate.from_template("""
User Question:
{question}

SQL Result:
{sql_result}

Answer the question strictly based on the SQL result.
If the result is empty, say:
"The data does not provide a clear answer to the question."
""")
    return answer_prompt | llm

# ------------------- CLEAN SQL -------------------
def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()

# ------------------- STREAMLIT APP -------------------
if __name__ == "__main__":
    schema = get_schema()
    llm = get_llm()
    sql_chain = get_sql_chain(llm)
    answer_chain = get_answer_chain(llm)

    # --- INITIALIZE SESSION STATE ---
    if "sql_cache" not in st.session_state:
        st.session_state.sql_cache = {}
    
    # This was the missing part causing your error!
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = {}

    user_question = st.text_input("Ask a question about the database:")

    if st.button("Get Answer") and user_question:
        
        # ================= CHAIN 1 (SQL Generation) =================
        if user_question in st.session_state.sql_cache:
            sql_query, result_df = st.session_state.sql_cache[user_question]
        else:
            # Get relevant few-shots based on the user question via RAG
            few_shot_examples = get_few_shot_context(user_question)
            
            sql_response = sql_chain.invoke({
                "schema": schema,
                "question": user_question,
                "few_shot_examples": few_shot_examples 
            })
            sql_query = clean_sql(sql_response.content)

            # Execute SQL
            if sql_query.lower().startswith("select"):
                try:
                    engine = get_db_engine()
                    with engine.connect() as conn:
                        result_df = pd.read_sql(sql_query, conn)
                except Exception as e:
                    st.error(f"SQL Execution Error: {e}")
                    result_df = pd.DataFrame()
            else:
                st.warning("Generated query is not a SELECT statement.")
                result_df = pd.DataFrame()

            # Cache the SQL and result
            st.session_state.sql_cache[user_question] = (sql_query, result_df)

        # Display SQL & results
        st.code(sql_query, language="sql")
        st.dataframe(result_df)

        # ================= CHAIN 2 (Natural Language Answer) =================
        if not result_df.empty:
            if user_question in st.session_state.answer_cache:
                final_answer = st.session_state.answer_cache[user_question]
            else:
                final_answer_response = answer_chain.invoke({
                    "question": user_question,
                    "sql_result": result_df.to_string()
                })
                final_answer = final_answer_response.content
                st.session_state.answer_cache[user_question] = final_answer

            st.markdown(f"**Answer:** {final_answer}")
        else:
            st.markdown("**Answer:** The data does not provide a clear answer to the question.")