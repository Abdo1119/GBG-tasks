import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
DB_URL = os.environ["DB_URL"]

# LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = "chinook-chatbot"

st.set_page_config(page_title="SQL Chatbot")
st.title("Chat with Postgres DB")


@st.cache_resource
def get_db():
    return SQLDatabase.from_uri(DB_URL)


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )


def get_sql_query(question):
    db = get_db()
    llm = get_llm()
    chain = create_sql_query_chain(llm, db)
    patched_question = question + "\nIMPORTANT: Always wrap ALL table and column names in double quotes because they are MixedCase in PostgreSQL."
    raw = chain.invoke({"question": patched_question})
    clean_sql = raw.replace("```sql", "").replace("```", "").strip()
    if clean_sql.upper().startswith("SQLQUERY:"):
        clean_sql = clean_sql[9:].strip()
    return clean_sql


def get_natural_response(question, data):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
User Question: {question}

Data returned from SQL query: {data}

Task: Answer the user's question based on the data returned from the SQL query.
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "data": data})


# --- Streamlit UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the database...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                sql_query = get_sql_query(question)
                st.code(sql_query, language="sql")

                db = get_db()
                result = db.run(sql_query)

                answer = get_natural_response(question, result)
                st.markdown(answer)

                full_response = f"```sql\n{sql_query}\n```\n\n{answer}"
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
