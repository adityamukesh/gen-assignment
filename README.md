# 🎓 Socratic RAG Tutor Chatbot

A lightweight Retrieval-Augmented Generation (RAG) tutor built with **Streamlit**, **ChromaDB**, **PyPDF**, and **Google Gemini (`gemini-2.5-flash`)**. Instead of giving away direct answers, the tutor guides students using hints and follow-up questions grounded strictly in uploaded PDF notes.

## Quickstart

1. **Clone & install dependencies**
   ```bash
   git clone <your-repo-url>
   cd rag-tutor-chatbot
   pip install -r requirements.txt
   ```

2. **Run the app**
   ```bash
   streamlit run app.py
   ```

3. **Use the Tutor**
   - Paste your `GEMINI_API_KEY` in the sidebar (or set it via `export GEMINI_API_KEY="your-key"`).
   - Upload one or more PDF study notes and click **Process PDFs**.
   - Start asking questions in the chat!
# gen-assignment
