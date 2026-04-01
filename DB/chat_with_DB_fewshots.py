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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI

load_dotenv()


# ------------------- CONFIG -------------------

DB_URL            = "postgresql://postgres:msFFfEoZeTdyGPmEVddUOOhmdEIUHIAd@gondola.proxy.rlwy.net:55236/railway"
NUM_VARIANTS      = 3   # how many variants to generate per question
MAX_HEAL_ATTEMPTS = 3   # max self-healing retries on the final chosen result

st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with Postgres DB :bar_chart:")

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

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
        azure_endpoint="https://gbgacademy-genai.openai.azure.com/",
        api_key=os.getenv("openai_api"),
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1",
        temperature=0
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
    """Generates N different ways to ask the same question."""
    prompt = PromptTemplate.from_template("""
You are a linguistic expert. Given a user's natural language question, generate exactly {num_variants} semantically equivalent but differently phrased versions.
Focus on varying the terminology (e.g., "total sales" vs "sum of revenue") and structure.

User Question: {question}

Return ONLY the variants, separated by the delimiter: ---VARIANT---
No numbering, no explanation.
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


# ------------------- STREAMLIT APP -------------------

if __name__ == "__main__":
    schema        = get_schema()
    llm           = get_llm()
    #variant_chain = get_variant_chain(llm)
    question_variant_chain = get_question_variant_chain(llm)
    sql_generation_chain = get_single_sql_chain(llm)
    judge_chain   = get_judge_chain(llm)
    heal_chain    = get_heal_chain(llm)
    answer_chain  = get_answer_chain(llm)

    # --- SESSION STATE ---
    if "sql_cache" not in st.session_state:
        st.session_state.sql_cache = {}
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = {}

    user_question = st.text_input("Ask a question about the database:")

    if st.button("Get Answer") and user_question:

        if user_question in st.session_state.sql_cache:
            final_sql, final_df, selection_method = st.session_state.sql_cache[user_question]

        else:
            # few_shot_examples = get_few_shot_context(user_question)

            # with st.status("🧠 Processing your question…", expanded=True) as status:

                # # ============================================================
                # # STEP 1 — Generate N SQL variants
                # # ============================================================
                # variant_response = variant_chain.invoke({
                #     "schema": schema,
                #     "question": user_question,
                #     "few_shot_examples": few_shot_examples,
                #     "num_variants": NUM_VARIANTS,
                # })

                # raw_variants = variant_response.content.split("---VARIANT---")
                # sql_variants = [clean_sql(v) for v in raw_variants if clean_sql(v)]
                # sql_variants = [
                #     v for v in sql_variants if v.lower().startswith("select")
                # ][:NUM_VARIANTS]

                # if not sql_variants:
                #     st.error("No valid SELECT variants were generated.")
                #     st.stop()

                # st.write(f"✅ Generated {len(sql_variants)} SQL variant(s)")

                # # ============================================================
                # # STEP 2 — Execute all variants (no self-healing here)
                # # ============================================================
                # results = []
                # for idx, sql in enumerate(sql_variants, 1):
                #     st.write(f"⚙️ Executing variant {idx}…")
                #     df, error = execute_sql(sql)
                #     if error:
                #         st.warning(f"Variant {idx} failed to execute: {error}")
                #     results.append({
                #         "idx": idx,
                #         "sql": sql,
                #         "df": df,
                #         "fingerprint": df_fingerprint(df),
                #         "error": error,
                #     })

                # # ============================================================
                # # STEP 3 — Consensus selection on FULL results
                # # ============================================================
                # st.write("🔍 Comparing full results across all variants…")
                # chosen, selection_method = select_consensus(results, user_question, judge_chain)


            with st.status("🧠 Processing your question…", expanded=True) as status:
                
                # ============================================================
                # STEP 1 — Generate N Question variants
                # ============================================================
                st.write("🔄 Rephrasing your question...")
                q_variant_resp = question_variant_chain.invoke({
                    "question": user_question,
                    "num_variants": NUM_VARIANTS
                })
                
                # Split and clean the questions
                questions = [q.strip() for q in q_variant_resp.content.split("---VARIANT---") if q.strip()]
                # Ensure we have the original + variants up to our limit
                all_questions = ([user_question] + questions)[:NUM_VARIANTS]

                # ============================================================
                # STEP 2 — Generate 1 SQL per question variant
                # ============================================================
                results = []
                for idx, q_var in enumerate(all_questions, 1):
                    st.write(f"⚙️ Processing variant {idx}: '{q_var[:50]}...'")
                    
                    # Get specific few-shots for THIS variation
                    few_shot_examples = get_few_shot_context(q_var)
                    
                    sql_resp = sql_generation_chain.invoke({
                        "schema": schema,
                        "question": q_var,
                        "few_shot_examples": few_shot_examples
                    })
                    
                    sql = clean_sql(sql_resp.content)
                    
                    # Execute immediately
                    df, error = execute_sql(sql)
                    
                    if error:
                        st.warning(f"Variant {idx} failed: {error}")
                    
                    results.append({
                        "idx": idx,
                        "question_variant": q_var,
                        "sql": sql,
                        "df": df,
                        "fingerprint": df_fingerprint(df),
                        "error": error,
                    })

                # ============================================================
                # STEP 3 — Consensus selection (Rest of your code remains the same!)
                # ============================================================
                st.write("🔍 Comparing results across variants…")
                chosen, selection_method = select_consensus(results, user_question, judge_chain)
                st.write(f"✅ Selected via: {selection_method}")

                final_sql = chosen["sql"]
                final_df  = chosen["df"]

                # ============================================================
                # STEP 4 — Self-healing on the FINAL chosen result only
                # ============================================================
                needs_healing = chosen["error"] is not None or final_df.empty

                if needs_healing:
                    st.write("🔧 Chosen result is empty or errored — running self-heal…")
                    issue = chosen["error"] or "The selected query returned no results."
                    final_sql, final_df, healed = self_heal_final(
                        final_sql, issue, user_question, schema, heal_chain
                    )
                    if healed:
                        st.write("✅ Self-healing succeeded")
                        selection_method += " → self-healed ✓"
                    else:
                        st.warning("⚠️ Self-healing exhausted all attempts.")
                        selection_method += " → self-heal failed"

                status.update(
                    label=f"Done — {selection_method}",
                    state="complete"
                )

            # Cache for re-use
            st.session_state.sql_cache[user_question] = (final_sql, final_df, selection_method)

        # ====================================================================
        # STEP 5 — Display final SQL & result
        # ====================================================================
        st.subheader("Final SQL Query")
        st.caption(f"Selection: {selection_method}")
        st.code(final_sql, language="sql")

        st.subheader("Query Result")
        st.dataframe(final_df)

        # ====================================================================
        # STEP 6 — Natural-language answer
        # ====================================================================
        if not final_df.empty:
            if user_question in st.session_state.answer_cache:
                final_answer = st.session_state.answer_cache[user_question]
            else:
                final_answer_response = answer_chain.invoke({
                    "question": user_question,
                    "sql_result": final_df.to_string(),
                })
                final_answer = final_answer_response.content
                st.session_state.answer_cache[user_question] = final_answer

            st.markdown(f"**Answer:** {final_answer}")
        else:
            st.markdown("**Answer:** The data does not provide a clear answer to the question.")
