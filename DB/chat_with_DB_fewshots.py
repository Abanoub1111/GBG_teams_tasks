##################### CHAIN ARCHITECTURE ###############################
import streamlit as st
import pandas as pd
import json
import os
import re
from collections import Counter
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

DB_URL            = "postgresql://postgres:msFFfEoZeTdyGPmEVddUOOhmdEIUHIAd@gondola.proxy.rlwy.net:55236/railway"
NUM_VARIANTS      = 3   # how many variants to generate per question
MAX_HEAL_ATTEMPTS = 3   # max self-healing retries on the final chosen result
MAX_QUERY_RESULT_ROWS = 15  # max rows to include in the LLM prompt for answering

st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with Postgres DB :bar_chart:")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sql_cache" not in st.session_state:
    st.session_state.sql_cache = {}
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message:
            with st.expander("View SQL"):
                st.code(message["sql"], language="sql")
        if "df" in message:
            st.dataframe(message["df"])

# ------------------- RAG / FEW-SHOT SETUP -------------------

@st.cache_resource
def get_retriever():
    with open("fewshots.json", "r") as f:
        few_shots = json.load(f)

    docs = [
        Document(
            page_content=ex["naturalQuestion"],
            metadata={"sql": ex["sqlQuery"]}
        ) for ex in few_shots
    ]

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def get_few_shot_context(question: str) -> str:
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
def get_schema() -> str:
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


# ------------------- PROMPT CHAINS -------------------

# @st.cache_resource
# def get_variant_chain(llm):
#     """Generates N semantically-equivalent SQL variants from one question."""
#     variant_prompt = PromptTemplate.from_template("""
# You are an expert PostgreSQL data analyst.

# Database schema:
# {schema}

# {few_shot_examples}

# User Question:
# {question}

# Generate exactly {num_variants} different but semantically equivalent SQL queries that answer the question.
# Each query may use a different approach (e.g., subquery vs JOIN, CTE vs inline, different column aliases).
# Return ONLY the SQL queries, separated by the delimiter: ---VARIANT---
# No explanation, no markdown, no numbering.
# """)
#     return variant_prompt | llm


# ------------------- UPDATED PROMPT CHAINS -------------------

@st.cache_resource
def get_question_variant_chain(llm):
    prompt = PromptTemplate.from_template("""
You are a linguistic expert. Given the following conversation history and a new user question, 
rephrase the new question to be a standalone question that captures the full context.
Then, generate {num_variants} semantically equivalent versions of that standalone question.

Chat History:
{chat_history}

New User Question: {question}

Return ONLY the variants, separated by ---VARIANT---.
""")
    return prompt | llm

@st.cache_resource
def get_single_sql_chain(llm):
    """Generates exactly ONE SQL query for a given question."""
    sql_prompt = PromptTemplate.from_template("""
You are an expert PostgreSQL data analyst.

Database schema:
{schema}

{few_shot_examples}

User Question:
{question}

Write a single PostgreSQL query that answers the question.
Use double quotes around table and column names.
Return ONLY the SQL. No markdown, no explanation.
""")
    return sql_prompt | llm

@st.cache_resource
def get_judge_chain(llm):
    """
    When all variants produce different results, the LLM picks the best one.
    Returns a single integer (1-based index) of the best result.
    """
    judge_prompt = PromptTemplate.from_template("""
You are an expert PostgreSQL data analyst acting as a judge.

The user asked:
{question}

{num_variants} SQL queries were executed and each returned a different result.
Your job is to pick the result that BEST and most completely answers the user's question.

{candidates}

Reply with ONLY a single integer — the number of the best result (e.g. 1, 2, or 3).
No explanation, no extra text.
""")
    return judge_prompt | llm


@st.cache_resource
def get_heal_chain(llm):
    """
    Self-healing applied ONLY to the final chosen SQL.
    Receives the current SQL + a description of the issue and returns a fixed query.
    """
    heal_prompt = PromptTemplate.from_template("""
You are an expert PostgreSQL data analyst fixing or improving a SQL query.

Database schema:
{schema}

Original user question:
{question}

Issue with the current query:
{issue}

Current SQL:
{current_sql}

Write ONLY a corrected/improved SQL query that properly answers the question.
- Use double quotes around table and column names.
- Return only valid SELECT SQL — no explanation, no markdown.
""")
    return heal_prompt | llm


@st.cache_resource
def get_answer_chain(llm):
    answer_prompt = PromptTemplate.from_template("""
User Question:
{question}

SQL Result:
{sql_result}

Answer the question strictly based on the SQL result.
If the data has ('Results length in DB: x. Only y selected') tell the user how many exists in DB and how many is shown for demonstration.
If the result is empty, say:
"The data does not provide a clear answer to the question."
""")
    return answer_prompt | llm


# ------------------- UTILITIES -------------------

def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()


def df_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable fingerprint of a FULL DataFrame.
    Sorts both columns and rows so order-only differences don't break equality.
    """
    if df.empty:
        return "__empty__"
    try:
        df_sorted = df.reindex(sorted(df.columns), axis=1)
        df_sorted = df_sorted.sort_values(by=list(df_sorted.columns)).reset_index(drop=True)
    except Exception:
        df_sorted = df
    return df_sorted.to_csv(index=False)


def execute_sql(sql: str) -> tuple[pd.DataFrame, str | None]:
    """Run a SELECT; return (DataFrame, None) or (empty df, error string)."""
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ------------------- CONSENSUS SELECTION -------------------

def select_consensus(results: list[dict], question: str, judge_chain) -> tuple[dict, str]:
    """
    Compare the FULL DataFrame output of every variant.

    Case A — ≥2 variants share the exact same full result:
        → Return that result. Selection method = "consensus (N/M agreed)".

    Case B — all variants returned different (non-empty) results:
        → Ask the LLM judge to pick the best one.
        → Selection method = "LLM judge (all differed)".

    Only successful, non-empty results participate.
    Falls back to the first result (for healing) if nothing succeeded.
    """
    valid = [r for r in results if not r["df"].empty and r["error"] is None]

    if not valid:
        # Nothing succeeded — return first result so self-healing can attempt a fix
        return results[0], "no valid results (self-heal required)"

    # Count matching full-result fingerprints
    counts = Counter(r["fingerprint"] for r in valid)
    best_fp, best_count = counts.most_common(1)[0]

    # ── Case A: at least 2 variants agree on the full result ──
    if best_count >= 2:
        for r in valid:
            if r["fingerprint"] == best_fp:
                agree_count = sum(1 for x in valid if x["fingerprint"] == best_fp)
                return r, f"consensus ({agree_count}/{len(results)} variants agreed)"

    # ── Case B: all different — let the LLM judge ──
    candidates_text = ""
    for r in valid:
        preview = r["df"].head(10).to_string(index=False)
        candidates_text += (
            f"Result {r['idx']}:\n"
            f"SQL used:\n{r['sql']}\n\n"
            f"Data (up to 10 rows):\n{preview}\n\n"
            f"{'─'*40}\n\n"
        )

    judge_response = judge_chain.invoke({
        "question": question,
        "num_variants": len(valid),
        "candidates": candidates_text,
    })

    # Parse the LLM's 1-based choice
    try:
        chosen_idx = int(re.search(r"\d+", judge_response.content).group())
    except Exception:
        chosen_idx = valid[0]["idx"]  # fallback

    for r in valid:
        if r["idx"] == chosen_idx:
            return r, f"LLM judge chose result {chosen_idx} (all {len(valid)} differed)"

    return valid[0], "LLM judge fallback (index parse failed)"


# ------------------- SELF-HEALING (FINAL RESULT ONLY) -------------------

def self_heal_final(
    chosen_sql: str,
    issue: str,
    question: str,
    schema: str,
    heal_chain,
    max_attempts: int = MAX_HEAL_ATTEMPTS,
) -> tuple[str, pd.DataFrame, bool]:
    """
    Self-healing runs ONLY on the final chosen SQL after consensus selection.

    'issue' is either an execution error string or a plain-language description
    (e.g. "query returned empty results").

    Returns (final_sql, final_df, healed_successfully).
    """
    current_sql = chosen_sql

    for attempt in range(1, max_attempts + 1):
        healed_response = heal_chain.invoke({
            "schema": schema,
            "question": question,
            "issue": issue,
            "current_sql": current_sql,
        })
        healed_sql = clean_sql(healed_response.content)
        df, error = execute_sql(healed_sql)

        if error is None and not df.empty:
            return healed_sql, df, True   # healed successfully

        # Update for the next attempt
        issue = error or "Query still returned empty results."
        current_sql = healed_sql

    return current_sql, pd.DataFrame(), False  # all attempts exhausted

def limit_results(df, limit):
    df_len = len(df)
    exceed = ""
    if df_len > limit:
        df = df[:limit]
        exceed = f"Results length in DB: {df_len}. Only {limit} Selected"
    return f"{exceed}\n{df.to_string(index=False)}"


# ------------------- STREAMLIT APP -------------------

if __name__ == "__main__":
    schema = get_schema()
    llm = get_llm()
    question_variant_chain = get_question_variant_chain(llm)
    sql_generation_chain = get_single_sql_chain(llm)
    judge_chain = get_judge_chain(llm)
    heal_chain = get_heal_chain(llm)
    answer_chain = get_answer_chain(llm)



    # --- INPUT HANDLING ---
    # We use chat_input as the single source of truth for the question
    user_question = st.chat_input("Ask a question about the database:")

    if user_question:
        # 1. Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # 2. Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # 3. Check Cache or Run Logic
        if user_question in st.session_state.sql_cache:
            final_sql, final_df, selection_method = st.session_state.sql_cache[user_question]
        else:
            with st.status("🧠 Processing your question...", expanded=True) as status:
                # --- STEP 1: History & Rephrasing ---
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]])
                
                st.write("🔄 Rephrasing with context...")
                q_variant_resp = question_variant_chain.invoke({
                    "chat_history": history_str,
                    "question": user_question,
                    "num_variants": NUM_VARIANTS
                })
                
                questions = [q.strip() for q in q_variant_resp.content.split("---VARIANT---") if q.strip()]
                all_questions = ([user_question] + questions)[:NUM_VARIANTS]

                # --- STEP 2: SQL Generation ---
                results = []
                for idx, q_var in enumerate(all_questions, 1):
                    st.write(f"⚙️ Variant {idx}: '{q_var[:50]}...'")
                    few_shot_examples = get_few_shot_context(q_var)
                    sql_resp = sql_generation_chain.invoke({
                        "schema": schema,
                        "question": q_var,
                        "few_shot_examples": few_shot_examples
                    })
                    sql = clean_sql(sql_resp.content)
                    df, error = execute_sql(sql)
                    results.append({
                        "idx": idx,
                        "sql": sql,
                        "df": df,
                        "fingerprint": df_fingerprint(df),
                        "error": error,
                    })

                # --- STEP 3: Consensus ---
                st.write("🔍 Selecting best result...")
                chosen, selection_method = select_consensus(results, user_question, judge_chain)
                final_sql = chosen["sql"]
                final_df = chosen["df"]

                # --- STEP 4: Self-Healing ---
                if chosen["error"] or final_df.empty:
                    st.write("🔧 Running self-heal...")
                    issue = chosen["error"] or "Empty results."
                    final_sql, final_df, healed = self_heal_final(
                        final_sql, issue, user_question, schema, heal_chain
                    )
                    selection_method += " (Healed)" if healed else " (Heal Failed)"

                status.update(label=f"Done — {selection_method}", state="complete")

            # Store in cache
            st.session_state.sql_cache[user_question] = (final_sql, final_df, selection_method)

        # --- STEP 5 & 6: Final Answer & Display ---
        if not final_df.empty:
            if user_question in st.session_state.answer_cache:
                final_answer = st.session_state.answer_cache[user_question]
            else:
                final_result = limit_results(final_df, limit=MAX_QUERY_RESULT_ROWS)
                ans_resp = answer_chain.invoke({"question": user_question, "sql_result": final_result})
                final_answer = ans_resp.content
                st.session_state.answer_cache[user_question] = final_answer
        else:
            final_answer = "The data does not provide a clear answer."

        # Display Assistant message
        with st.chat_message("assistant"):
            st.markdown(final_answer)
            st.dataframe(final_df)
            with st.expander("View SQL"):
                st.code(final_sql, language="sql")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer,
            "sql": final_sql,
            "df": final_df
        })
