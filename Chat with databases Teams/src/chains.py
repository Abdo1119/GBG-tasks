import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)
from src.database import get_cached_table_info, execute_query
from src.prompts import build_fewshot_prompt, RESPONSE_PROMPT
from src.utils import clean_sql, validate_sql_readonly


@st.cache_resource
def get_llm():
    return AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def generate_sql(question: str) -> str:
    """Generate a SQL query from a natural language question."""
    llm = get_llm()
    prompt = build_fewshot_prompt()
    chain = prompt | llm | StrOutputParser()
    table_info = get_cached_table_info()
    raw = chain.invoke({"input": question, "table_info": table_info})
    return clean_sql(raw)


def run_sql(sql: str) -> str:
    """Validate and execute a SQL query. Raises ValueError for write operations."""
    is_valid, error_msg = validate_sql_readonly(sql)
    if not is_valid:
        raise ValueError(error_msg)
    return execute_query(sql)


def generate_response(question: str, data: str) -> str:
    """Generate a natural language response from query results."""
    llm = get_llm()
    chain = RESPONSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question, "data": data})
