import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="SHL Test Recommender", layout="wide")

# Load your CSV
@st.cache_data
def load_data():
    df = pd.read_csv("SHL_Scraped_Data.csv")
    df['text'] = df.apply(lambda row: f"passage: {row['Name']} {row['Job Level']} {row['Test Types']} {row['Languages']} {row['Completion Time']}", axis=1)
    return df

df = load_data()

# Create or load vector store
@st.cache_resource
def load_vector_store():
    texts = df['text'].astype(str)
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

# Setup model
llm = ChatOllama(model="llama3")  # You can change model here

# Streamlit UI
st.title("🔍 SHL Individual Test Recommender")

query = st.text_input("Enter your requirement (e.g. 'Test for entry-level role in sales')")

if query:
    with st.spinner("Finding most relevant assessments..."):
        docs = retriever.invoke(query)
        row_indices = list({doc.metadata["row_index"] for doc in docs})  # ✅ Avoid duplicates

        top_df = df.loc[row_indices].head(10)  # ✅ safer than iloc
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

        # System prompt
        system_prompt = f"""
You are a helpful assistant. From the below context, recommend the top 10 relevant individual assessments in *Markdown table* format. Each row should include:
- *Name* (as a [clickable link](URL)),
- *Remote Testing* (Yes/No),
- *Adaptive/IRT* (Yes/No),
- *Completion Time*,
- *Test Types*.

Only use the provided context. Do not hallucinate. If unsure, say you don't know.

Context:
{context_text}
"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])

        # Display the results
        st.markdown("### 🧠 Recommended Assessments")
        
        # Display response as markdown
        st.markdown(response.content)

