# Session 3 — Chat with Database (Chinook)

A Streamlit chatbot that lets you ask natural language questions about the **Chinook** music store database. The app converts your question into an SQL query using an LLM, runs it against a PostgreSQL database, and returns a human-readable answer.

## Database Schema

![Chinook Schema](Schema.jpg)

The Chinook database models a digital music store with the following tables:

| Table | Description |
|-------|-------------|
| **Artist** | Music artists |
| **Album** | Albums linked to artists |
| **Track** | Individual tracks with price, duration, genre, and media type |
| **Genre** | Music genres (Rock, Jazz, etc.) |
| **MediaType** | File formats (MPEG, AAC, etc.) |
| **Playlist / PlaylistTrack** | Playlists and their track associations |
| **Customer** | Store customers |
| **Employee** | Store employees with reporting hierarchy |
| **Invoice / InvoiceLine** | Purchase invoices and line items |

## Tech Stack

- **Streamlit** — Web UI with chat interface
- **LangChain** — SQL query chain & prompt orchestration
- **Google Gemini 2.5 Flash** — LLM for text-to-SQL and response generation
- **PostgreSQL** (hosted on Railway) — Database engine
- **LangSmith** — Tracing & observability

## Project Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit chatbot application |
| `deploy.py` | Script to upload CSV data into PostgreSQL |
| `.env.example` | Template for required environment variables |
| `Schema.jpg` | Database ER diagram |
| `*.csv` | Chinook dataset (11 tables) |

## How It Works

1. User types a question in the chat (e.g., *"How many tracks are in the Rock genre?"*)
2. LangChain builds an SQL query using Gemini
3. The query runs against the PostgreSQL database
4. The raw result is passed back to Gemini to generate a natural language answer
5. Both the SQL query and the answer are displayed in the chat

## Setup & Run

### 1. Install dependencies

```bash
pip install streamlit langchain-google-genai langchain-community langchain-classic langchain-core sqlalchemy psycopg2-binary pandas
```

### 2. Set environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required variables:
- `GOOGLE_API_KEY` — Google AI API key
- `DB_URL` — PostgreSQL connection string
- `LANGSMITH_API_KEY` — LangSmith API key (optional, for tracing)

### 3. Deploy data to PostgreSQL (first time only)

```bash
python deploy.py
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.
