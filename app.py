import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="SHL Test Recommender", layout="wide")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("SHL_Scraped_Data.csv")

df = load_data()

# Create vector store from SHL data
@st.cache_resource
def load_vector_store():
    texts = df['context'].astype(str)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    documents = []
    for idx, text in texts.items():
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"row_index": idx}))
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(documents, embeddings)
    return db.as_retriever()

retriever = load_vector_store()
llm = ChatOllama(model="mistral").with_config({"max_tokens": 512})

# Extract clean job description text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        main_div = soup.find("div", {"class": lambda c: c and ("job" in c.lower() or "description" in c.lower())})
        if main_div:
            text = main_div.get_text(separator=" ", strip=True)
        else:
            paragraphs = soup.find_all("p")
            text = max(paragraphs, key=lambda p: len(p.get_text(strip=True)), default="").get_text(strip=True)

        return text[:3000]  # limit
    except Exception:
        return ""

# UI
st.title("🔍 SHL Individual Test Recommender")

input_mode = st.radio("Select Input Type", ["Natural Language Query", "Job Description URL"], horizontal=True)
user_input = st.text_input("Enter your query or job post URL")

if user_input:
    if input_mode == "Job Description URL":
        with st.spinner("Extracting job description..."):
            user_input = extract_text_from_url(user_input)
            if len(user_input.split()) < 30:
                st.warning("⚠️ The extracted job description seems too short. Try another URL or paste text manually.")
                st.stop()

    with st.spinner("Finding most relevant assessments..."):
        docs = retriever.invoke(user_input)
        row_indices = list({doc.metadata["row_index"] for doc in docs})
        top_df = df.loc[row_indices].head(10)

        context_text = ""
        for _, row in top_df.iterrows():
            context_text += f"""
Name: {row['Name']}
Remote Testing: {row['Remote Testing']}
Adaptive/IRT: {row['Adaptive/IRT']}
Duration: {row['Completion Time']}
Test Types: {row['Test Types']}
URL: {row['URL']}
---
"""

        system_prompt = f"""
You are a helpful assistant. From the below context, recommend the top 10 relevant individual assessments in **Markdown table** format. Each row should include:
- **Name** (as a [clickable link](URL)),
- **Remote Testing** (Yes/No),
- **Adaptive/IRT** (Yes/No),
- **Completion Time**,
- **Test Types**.

Only use the provided context. Do not hallucinate. If unsure, say you don't know.

Context:
{context_text}
"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])

        st.markdown("### 🧠 Recommended Assessments")
        st.markdown(response.content)
