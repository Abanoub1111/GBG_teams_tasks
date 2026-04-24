import os
import re
import json
import pandas as pd
import streamlit as st
from collections import Counter
from dotenv import load_dotenv

# LangChain & Neo4j Imports
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()

# ------------------- CONFIG -------------------
NUM_VARIANTS = 3
MAX_HEAL_ATTEMPTS = 3
MAX_QUERY_RESULT_ROWS = 15

st.set_page_config(page_title="Neo4j Graph Chatbot", page_icon="🕸️", layout="wide")
st.title("Chat with Neo4j Graph_DB")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cypher_cache" not in st.session_state:
    st.session_state.cypher_cache = {}
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

# ------------------- DATABASE -------------------

@st.cache_resource
def get_graph():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        enhanced_schema=False
    )

def execute_cypher(graph, cypher: str) -> tuple[pd.DataFrame, str | None]:
    try:
        result = graph.query(cypher)
        df = pd.DataFrame(result)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

# ------------------- LLM & CHAINS -------------------

@st.cache_resource
def get_llm():
    return AzureChatOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), temperature=0)

@st.cache_resource
def get_question_variant_chain(_llm):
    prompt = PromptTemplate.from_template("""
You are a linguistic expert. Given the chat history and a new question, 
rephrase it to be standalone and generate {num_variants} equivalent versions.

Chat History: {chat_history}
New Question: {question}

Return ONLY variants separated by ---VARIANT---.
""")
    return prompt | _llm

@st.cache_resource
def get_cypher_generation_chain(_llm):
    prompt = PromptTemplate.from_template("""
You are a Neo4j Cypher expert. 

Graph Schema:
{schema}

User Question: {question}

--- CONSTRAINTS ---
- Use ONLY MATCH, WHERE, RETURN. No CREATE, DELETE, or SET.
- Use valid Cypher syntax for Neo4j.
- Return ONLY the Cypher query. No markdown, no explanations.
""")
    return prompt | _llm

@st.cache_resource
def get_judge_chain(_llm):
    prompt = PromptTemplate.from_template("""
Analyze {num_variants} Cypher results for the question: {question}
Pick the best result (1-{num_variants}) that accurately answers the question.

{candidates}

Return ONLY the integer.
""")
    return prompt | _llm

@st.cache_resource
def get_heal_chain(_llm):
    prompt = PromptTemplate.from_template("""
Fix the following Cypher query based on the error provided.

Graph Schema: {schema}
Question: {question}
Error: {issue}
Current Cypher: {current_cypher}

Return ONLY the fixed Cypher query.
""")
    return prompt | _llm

@st.cache_resource
def get_answer_chain(_llm):
    prompt = PromptTemplate.from_template("""
User Question: {question}
Cypher Result: {result_data}

Provide a natural language answer strictly based on the result.
If empty, say "The graph does not provide a clear answer."
""")
    return prompt | _llm

# ------------------- UTILITIES -------------------

def clean_cypher(text: str) -> str:
    text = re.sub(r"```cypher", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

def df_fingerprint(df: pd.DataFrame) -> str:
    if df.empty: return "__empty__"
    return df.to_csv(index=False)

def select_consensus(results, question, judge_chain):
    valid = [r for r in results if not r["df"].empty and r["error"] is None]
    if not valid: return results[0], "no valid results (healing required)"
    
    counts = Counter(r["fingerprint"] for r in valid)
    best_fp, best_count = counts.most_common(1)[0]

    if best_count >= 2:
        for r in valid:
            if r["fingerprint"] == best_fp:
                return r, f"consensus ({best_count}/{len(results)})"

    candidates_text = ""
    for r in valid:
        candidates_text += f"Result {r['idx']}:\nCypher: {r['cypher']}\nData: {r['df'].head(5).to_string()}\n\n"
    
    resp = judge_chain.invoke({"question": question, "num_variants": len(valid), "candidates": candidates_text})
    try:
        idx = int(re.search(r"\d+", resp.content).group())
        return next(r for r in valid if r["idx"] == idx), f"LLM judge (Index {idx})"
    except:
        return valid[0], "LLM judge fallback"

def self_heal_cypher(graph, cypher, issue, question, schema, heal_chain):
    current = cypher
    for _ in range(MAX_HEAL_ATTEMPTS):
        resp = heal_chain.invoke({"schema": schema, "question": question, "issue": issue, "current_cypher": current})
        fixed = clean_cypher(resp.content)
        df, err = execute_cypher(graph, fixed)
        if err is None and not df.empty:
            return fixed, df, True
        current, issue = fixed, (err or "Empty results")
    return current, pd.DataFrame(), False

# ------------------- MAIN APP -------------------

def main():
    graph = get_graph()
    schema = graph.schema
    llm = get_llm()
    
    # Chains
    q_var_chain = get_question_variant_chain(llm)
    cypher_gen_chain = get_cypher_generation_chain(llm)
    judge_chain = get_judge_chain(llm)
    heal_chain = get_heal_chain(llm)
    ans_chain = get_answer_chain(llm)

    # Display History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "cypher" in m: st.code(m["cypher"], language="cypher")
            if "df" in m: st.dataframe(m["df"])

    if user_input := st.chat_input("Ask about your Graph..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        if user_input in st.session_state.cypher_cache:
            final_cypher, final_df, method = st.session_state.cypher_cache[user_input]
        else:
            with st.status("🧠 Processing...", expanded=True) as status:
                # 1. Rephrase & Variants
                hist = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
                vars_resp = q_var_chain.invoke({"chat_history": hist, "question": user_input, "num_variants": NUM_VARIANTS})
                questions = [user_input] + [q.strip() for q in vars_resp.content.split("---VARIANT---") if q.strip()]
                
                # 2. Cypher Generation
                results = []
                for i, q in enumerate(questions[:NUM_VARIANTS], 1):
                    st.write(f"⚙️ Generating Cypher {i}...")
                    c_resp = cypher_gen_chain.invoke({"schema": schema, "question": q})
                    cypher = clean_cypher(c_resp.content)
                    df, err = execute_cypher(graph, cypher)
                    results.append({"idx": i, "cypher": cypher, "df": df, "fingerprint": df_fingerprint(df), "error": err})

                # 3. Consensus & Healing
                chosen, method = select_consensus(results, user_input, judge_chain)
                final_cypher, final_df = chosen["cypher"], chosen["df"]

                if chosen["error"] or final_df.empty:
                    st.write("🔧 Self-healing active...")
                    issue = chosen["error"] or "Query returned no data."
                    final_cypher, final_df, ok = self_heal_cypher(graph, final_cypher, issue, user_input, schema, heal_chain)
                    method += " (Healed)" if ok else " (Failed)"
                
                status.update(label=f"Complete: {method}", state="complete")
            
            st.session_state.cypher_cache[user_input] = (final_cypher, final_df, method)

        # 4. Final Answer
        ans_resp = ans_chain.invoke({"question": user_input, "result_data": final_df.head(MAX_QUERY_RESULT_ROWS).to_string()})
        ans = ans_resp.content
        
        with st.chat_message("assistant"):
            st.markdown(ans)
            st.dataframe(final_df)
            with st.expander("View Cypher"): st.code(final_cypher, language="cypher")

        st.session_state.messages.append({"role": "assistant", "content": ans, "cypher": final_cypher, "df": final_df})

if __name__ == "__main__":
    main()