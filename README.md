# Finquiry

Multi-user document search and conversational Q&A application built with FastAPI, Streamlit, LangChain, ChromaDB, and Groq.

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/Tiwari-Tech/Finquiry.git
   cd Finquiry
   python -m venv .venv
   ```

2. Activate the virtual environment and install dependencies:

   ```bash
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Set `GROQ_API_KEY` and `JWT_SECRET_KEY` in `.env`. Embeddings default to local Ollama (`nomic-embed-text`); run `ollama pull nomic-embed-text` or switch `EMBEDDING_PROVIDER` in `backend/services/embedding_service.py`.

## Run

Start the backend API:

```bash
uvicorn backend.main:app --reload
```

Start the Streamlit frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

## Notes

- Image OCR requires [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system.
- Local SQLite (`backend/data/users.db`) and ChromaDB (`data/chroma_db/`) are created automatically at runtime and excluded from git.
