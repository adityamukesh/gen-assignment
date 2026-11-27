import os
import uuid
import chromadb
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

st.set_page_config(page_title="RAG AI Tutor", page_icon="🎓", layout="centered")
st.title("🎓 Socratic RAG Tutor")
st.caption("Upload your study PDFs and learn through guided questions.")

# --- Sidebar: API Key & PDF Upload ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key from Google AI Studio",
    )
    uploaded_files = st.file_uploader(
        "Upload Study Material (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    process_btn = st.button("Process PDFs", type="primary", use_container_width=True)
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Helper Functions ---
def extract_and_chunk(files, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    chunks = []
    for file in files:
        reader = PdfReader(file)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        start = 0
        while start < len(full_text):
            chunk = full_text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(f"[{file.name}] {chunk}")
            start += chunk_size - overlap
    return chunks

# --- Session State Initialization ---
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"notes_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_ready" not in st.session_state:
    st.session_state.docs_ready = False

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name=st.session_state.collection_name)

# --- Process Uploaded PDFs ---
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Indexing your notes..."):
            chunks = extract_and_chunk(uploaded_files)
            if chunks:
                ids = [f"chunk_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
                collection.upsert(documents=chunks, ids=ids)
                st.session_state.docs_ready = True
                st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s)!")
            else:
                st.sidebar.error("No extractable text found in the uploaded PDFs.")

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle Student Prompt ---
TUTOR_INSTRUCTION = """
You are a patient, encouraging AI Tutor.
Rules:
1. Base your explanations strictly on the provided [COURSE CONTEXT].
2. If the answer is missing from the context, politely state that your current notes do not cover it.
3. Do NOT just give away final answers immediately. Explain the underlying concept briefly, give a helpful hint, and ask a short follow-up question to check the student's understanding.
"""

if prompt := st.chat_input("Ask a question about your notes..."):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
        st.stop()
    if not st.session_state.docs_ready:
        st.warning("Upload and click 'Process PDFs' in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve top 3 relevant chunks
    results = collection.query(query_texts=[prompt], n_results=3)
    docs = results["documents"][0] if results["documents"] else []
    context = "\n---\n".join(docs)

    # Build conversation history for Gemini
    client = genai.Client(api_key=api_key)
    history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in st.session_state.messages[:-1]
    ]

    augmented_prompt = f"""[COURSE CONTEXT]
{context}

[STUDENT MESSAGE]
{prompt}"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat = client.chats.create(
                model="gemini-2.5-flash",
                history=history,
                config=types.GenerateContentConfig(
                    system_instruction=TUTOR_INSTRUCTION,
                    temperature=0.4,
                ),
            )
            response = chat.send_message(augmented_prompt)
            st.markdown(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text})
