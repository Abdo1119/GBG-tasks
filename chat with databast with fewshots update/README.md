# Chat with Database — Few-Shot Update

An enhanced version of the Chinook SQL Chatbot that uses **few-shot prompting** to improve SQL query generation accuracy.

## What Changed (vs Session-3)

- Added `fewshots.json` with **25 curated question-SQL pairs** covering:
  - Basic queries (COUNT, WHERE, ORDER BY)
  - JOINs and aggregations (GROUP BY, HAVING, SUM)
  - Window functions (RANK, PARTITION BY)
  - Recursive CTEs (employee hierarchy traversal)
  - Subqueries and self-joins
- Replaced `create_sql_query_chain` with a custom **FewShotPromptTemplate** pipeline
- The LLM now sees real examples before generating SQL, producing more accurate queries with correct PostgreSQL double-quoting

## Database Schema

![Chinook Schema](Schema.jpg)

## Tech Stack

- **Streamlit** — Web UI with chat interface
- **LangChain** — FewShotPromptTemplate & prompt orchestration
- **Google Gemini 2.5 Flash** — LLM for text-to-SQL and response generation
- **PostgreSQL** (hosted on Railway) — Database engine
- **LangSmith** — Tracing & observability

## Project Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit chatbot with few-shot prompting |
| `fewshots.json` | 25 curated question-SQL example pairs |
| `deploy.py` | Script to upload CSV data into PostgreSQL |
| `Schema.jpg` | Database ER diagram |
| `*.csv` | Chinook dataset (11 tables) |

## How It Works

1. User types a natural language question
2. The app builds a **FewShotPromptTemplate** with the database schema + 25 example Q&A pairs
3. Gemini generates an SQL query guided by the examples
4. The query runs against PostgreSQL
5. The raw result is passed back to Gemini for a natural language answer
6. Both the SQL and answer are displayed in the chat

## Setup & Run

### 1. Install dependencies

```bash
pip install streamlit langchain-google-genai langchain-community langchain-core sqlalchemy psycopg2-binary pandas
```

### 2. Set environment variables

```bash
export GOOGLE_API_KEY=your-google-api-key
export DB_URL=postgresql://user:password@host:port/dbname
export LANGSMITH_API_KEY=your-langsmith-api-key  # optional
```

### 3. Run the chatbot

```bash
streamlit run app.py
```

### 4. Try these example questions

- "How many customers are in the USA?"
- "Find the top 5 customers by total spending"
- "Which country generated the highest revenue?"
- "Count how many tracks exist in each genre"
- "Find the hierarchy depth for each employee"
