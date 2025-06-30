# Full code for Streamlit BigQuery chatbot with fuzzy matching and quarter comparison
import os
import streamlit as st
import pandas as pd
import re
import json
from dotenv import load_dotenv
from google.cloud import bigquery
from openai import OpenAI

# Load environment variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize clients
client = OpenAI(api_key=openai_api_key)
bq_client = bigquery.Client()
TABLE_NAME = "vend-it-data-analysis.MachinesSales.data"

# Streamlit setup
st.set_page_config(page_title="Ask Your Data", layout="centered")
st.title("🧠 Vend IT Data Analyst")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "table_info" not in st.session_state:
    st.session_state.table_info = None

@st.cache_data(ttl=3600)
def get_table_info():
    try:
        table = bq_client.get_table(TABLE_NAME)
        schema_df = pd.DataFrame([{ 'column_name': f.name, 'data_type': f.field_type, 'mode': f.mode, 'description': f.description or 'No description' } for f in table.schema])
        sample_data = bq_client.query(f"SELECT * FROM `{TABLE_NAME}` LIMIT 10").to_dataframe()
        stats_queries = {
            'total_records': f"SELECT COUNT(*) as count FROM `{TABLE_NAME}`",
            'date_range': f"SELECT MIN(transaction_datetime) as min_date, MAX(transaction_datetime) as max_date FROM `{TABLE_NAME}`",
            'unique_products': f"SELECT COUNT(DISTINCT product_name) as count FROM `{TABLE_NAME}`",
            'unique_machines': f"SELECT COUNT(DISTINCT machine_tag) as count FROM `{TABLE_NAME}`",
            'dispense_statuses': f"SELECT dispense_status, COUNT(*) as count FROM `{TABLE_NAME}` GROUP BY dispense_status",
            'payment_methods': f"SELECT payment_method, COUNT(*) as count FROM `{TABLE_NAME}` GROUP BY payment_method"
        }
        stats = {}
        for key, query in stats_queries.items():
            try:
                rows = list(bq_client.query(query).result())
                stats[key] = {row[0] if key in ['dispense_statuses', 'payment_methods'] else list(row.keys())[0]: row[1] for row in rows} if rows else {}
            except:
                stats[key] = {}
        return { 'schema': schema_df, 'sample_data': sample_data, 'stats': stats }
    except Exception as e:
        st.error(f"Error fetching table info: {str(e)}")
        return None

def normalize_quarter_question(question):
    return question.replace("Q1", "January to March").replace("Q2", "April to June").replace("Q3", "July to September").replace("Q4", "October to December")

def apply_fuzzy_patch(sql):
    for col in ["product_name", "machine_tag"]:
        sql = re.sub(fr"{col}\s*=\s*['\"](.*?)['\"]", lambda m: f"LOWER({col}) LIKE '%{m.group(1).lower()}%'", sql, flags=re.IGNORECASE)
    return sql

def run_query_safely(sql):
    if any(x in sql.upper() for x in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE", "MERGE"]):
        raise Exception("Only SELECT queries are allowed.")
    sql = apply_fuzzy_patch(sql.strip().rstrip(";"))
    try:
        return bq_client.query(sql).to_dataframe()
    except Exception as e:
        raise Exception(f"Query failed: {str(e)}")

def generate_intelligent_response(question, table_info, chat_context=""):
    schema_text = "\n".join([f"- {row['column_name']} ({row['data_type']})" for _, row in table_info['schema'].iterrows()])
    sample_text = table_info['sample_data'].head(3).to_string(index=False)
    context_info = f"""
TABLE: `{TABLE_NAME}`

SCHEMA:
{schema_text}

SAMPLE DATA:
{sample_text}

DATA STATISTICS:
- Total records: {table_info['stats'].get('total_records', {}).get('count', 'Unknown')}
- Unique products: {table_info['stats'].get('unique_products', {}).get('count', 'Unknown')}
- Unique machines: {table_info['stats'].get('unique_machines', {}).get('count', 'Unknown')}
- Date range: {table_info['stats'].get('date_range', {})}
- Dispense statuses: {table_info['stats'].get('dispense_statuses', {})}
- Payment methods: {table_info['stats'].get('payment_methods', {})}

RECENT CONVERSATION:
{chat_context}
"""
    prompt = f"""
You are an intelligent data analyst assistant. Based on the user's question and available data:
- Understand the question.
- Answer directly or generate SQL.
- Use conversational tone.

INSTRUCTIONS:
- If text filter: use LOWER(col) LIKE '%value%'
- If quarter: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
- Use EXTRACT(QUARTER FROM date) and EXTRACT(YEAR FROM date)
- For listing: use SELECT DISTINCT col

CONTEXT:
{context_info}

USER QUESTION: "{question}"

Respond in JSON:
{{
  "response_type": "direct_answer" or "sql_query",
  "sql_query": "..." or null,
  "direct_answer": "..." or null,
  "explanation": "..."
}}
"""
    try:
        res = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        try:
            return json.loads(res.choices[0].message.content.strip())
        except:
            return {"response_type": "sql_query", "sql_query": res.choices[0].message.content.strip(), "direct_answer": None, "explanation": "Generated SQL query"}
    except Exception as e:
        return {"response_type": "error", "sql_query": None, "direct_answer": f"Error: {str(e)}", "explanation": "Failed to process."}

def generate_conversational_response(question, df, query_info):
    if df.empty:
        return "I couldn't find any matching data."
    if df.shape == (1, 1):
        val = df.iloc[0, 0]
        col = df.columns[0]
        val_fmt = f"${val:,.2f}" if isinstance(val, (int, float)) and any(k in col.lower() for k in ["amount", "revenue", "total"]) else f"{val:,}" if isinstance(val, (int, float)) else str(val)
        prompt = f"The user asked: \"{question}\"\nThe result is: {val_fmt}\nGenerate a helpful, brief answer."
        try:
            res = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.4)
            return res.choices[0].message.content.strip()
        except:
            return f"The answer is: **{val_fmt}**"
    if len(df) <= 10:
        summary_prompt = f"The user asked: \"{question}\"\n\nData:\n{df.to_string(index=False)}\n\nSummarize key insights."
        try:
            res = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": summary_prompt}], temperature=0.4)
            return res.choices[0].message.content.strip()
        except:
            return f"I found {len(df)} results."
    return f"I found {len(df)} results. Showing top 5:\n\n{df.head(5).to_string(index=False)}"

# UI Section
if st.session_state.table_info is None:
    with st.spinner("Loading data info..."):
        st.session_state.table_info = get_table_info()

if st.session_state.table_info:
    with st.form("question_form", clear_on_submit=True):
        user_question_raw = st.text_input("Ask me anything about your vending machine data:", placeholder="e.g., Compare Q1 2024 with Q1 2025 sales", key="user_input")
        ask_button = st.form_submit_button("Ask", use_container_width=True)

    if ask_button and user_question_raw:
        user_question = normalize_quarter_question(user_question_raw)
        st.session_state.chat_history.append(("user", user_question_raw))
        recent_context = "\n".join([f"{r}: {m}" for r, m in st.session_state.chat_history[-6:-1]])
        with st.spinner("🤔 Thinking about your question..."):
            ai_response = generate_intelligent_response(user_question, st.session_state.table_info, recent_context)
            if ai_response["response_type"] == "direct_answer":
                st.session_state.chat_history.append(("assistant", ai_response["direct_answer"]))
                st.success(f"💬 Answer: {ai_response['direct_answer']}")
                if ai_response.get("explanation"):
                    st.caption(f"ℹ️ {ai_response['explanation']}")
            elif ai_response["response_type"] == "sql_query" and ai_response["sql_query"]:
                try:
                    df = run_query_safely(ai_response["sql_query"])
                    response_text = generate_conversational_response(user_question_raw, df, ai_response)
                    st.session_state.chat_history.append(("assistant", response_text))
                    st.success(f"💬 Answer: {response_text}")
                    if not df.empty and len(df) > 1:
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False)
                        st.download_button("📥 Download Results", csv, f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                    with st.expander("🔍 View SQL Query"):
                        st.code(ai_response["sql_query"], language="sql")
                except Exception as e:
                    st.session_state.chat_history.append(("assistant", str(e)))
                    st.error(str(e))
            else:
                fallback = ai_response.get("direct_answer", "I couldn't process your question.")
                st.session_state.chat_history.append(("assistant", fallback))
                st.warning(fallback)

    if st.session_state.chat_history:
        st.markdown("---\n### 💬 Conversation History")
        for role, msg in st.session_state.chat_history[-10:]:
            st.markdown(f"**{'🧑 You' if role == 'user' else '🤖 Assistant'}:** {msg}\n")
else:
    st.error("❌ Could not connect to BigQuery. Please check configuration.")
