import json
import os
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
import streamlit as st


@st.cache_data
def load_fewshots():
    fewshots_path = os.path.join(os.path.dirname(__file__), "..", "data", "fewshots.json")
    with open(fewshots_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fewshot_prompt():
    examples = load_fewshots()

    example_prompt = PromptTemplate.from_template(
        "Question: {naturalQuestion}\nSQL: {sqlQuery}"
    )

    fewshot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""You are a PostgreSQL expert. Given a user question, generate a syntactically correct PostgreSQL query.
IMPORTANT: Always wrap ALL table and column names in double quotes because they are MixedCase in PostgreSQL.
Only use the tables and columns listed below.
Return ONLY the SQL query, no explanations.

Database schema:
{table_info}

Here are some example questions and their correct SQL queries:""",
        suffix="Question: {input}\nSQL:",
        input_variables=["input", "table_info"],
    )
    return fewshot_prompt


RESPONSE_PROMPT = ChatPromptTemplate.from_template("""
User Question: {question}

Data returned from SQL query: {data}

Task: Answer the user's question based on the data returned from the SQL query.
Provide a clear, concise answer in natural language.
""")
