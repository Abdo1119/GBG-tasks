import re

FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)

SQL_INJECTION_PATTERN = re.compile(
    r"\b(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|ALTER\s+TABLE|CREATE\s+TABLE|TRUNCATE\s+TABLE)\b",
    re.IGNORECASE,
)


def validate_sql_readonly(sql: str) -> tuple[bool, str]:
    """Validate that a SQL query is read-only. Returns (is_valid, error_message)."""
    match = FORBIDDEN_SQL_KEYWORDS.search(sql)
    if match:
        return False, f"Query rejected: write operation '{match.group()}' is not allowed. This is a read-only database."
    return True, ""


def clean_sql(raw: str) -> str:
    """Clean LLM output to extract the SQL query."""
    sql = raw.replace("```sql", "").replace("```", "").strip()
    if sql.upper().startswith("SQLQUERY:"):
        sql = sql[9:].strip()
    if sql.upper().startswith("SQL:"):
        sql = sql[4:].strip()
    return sql


def validate_question(question: str, max_length: int = 500) -> tuple[bool, str]:
    """Validate user input. Returns (is_valid, error_message)."""
    if not question or not question.strip():
        return False, "Please enter a question."
    if len(question) > max_length:
        return False, f"Question is too long ({len(question)} chars). Maximum is {max_length} characters."
    if SQL_INJECTION_PATTERN.search(question):
        return False, "Your question contains SQL-like syntax that is not allowed. Please rephrase using natural language."
    return True, ""
