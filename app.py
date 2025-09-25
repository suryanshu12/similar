import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# ------------------- Load .env -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- Config -------------------
TOP_K_RETRIEVE = 50
FINAL_RESULTS = 10
SYNONYMS = {
    "wireless": ["inductive", "contactless"],
    "charging": ["power transfer", "energy transfer"],
}

# ------------------- Local file paths -------------------
EXCEL_PATH = "patent_data.xlsx"
JSON_PATH = "combined_texts.json"
NPY_PATH = "patent_embeddings_mpnet.npy"

# ------------------- Load Data -------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_excel(EXCEL_PATH)
    df.rename(columns={
        "Patent_Number": "patent_number",
        "Title": "title",
        "Abstract": "abstract",
        "Claims": "claims",
        "Description": "description"
    }, inplace=True)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        combined_texts = json.load(f)

    embeddings = np.load(NPY_PATH, allow_pickle=True)
    return df, combined_texts, embeddings

df, combined_texts, embeddings = load_data()

# ------------------- FAISS Setup -------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# ------------------- Models -------------------
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device="cpu")

# ------------------- Helper Functions -------------------
def clean_text(text):
    return re.sub('<.*?>', '', str(text))

def expand_query_with_synonyms(query):
    tokens = query.lower().split()
    expanded_tokens = []
    for token in tokens:
        expanded_tokens.append(token)
        if token in SYNONYMS:
            expanded_tokens.extend(SYNONYMS[token])
    return " ".join(expanded_tokens)

def search(query, top_k=TOP_K_RETRIEVE):
    expanded_query = expand_query_with_synonyms(query)
    query_embedding = embedder.encode([expanded_query], normalize_embeddings=True)
    D, I = index.search(query_embedding, top_k)

    candidates = []
    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        combined_text = combined_texts[idx]
        candidates.append({
            "idx": idx,
            "text": combined_text,
            "metadata": row,
            "faiss_score": D[0][i]
        })

    cross_inp = [(query, c['text']) for c in candidates]
    rerank_scores = reranker.predict(cross_inp)
    for c, score in zip(candidates, rerank_scores):
        c['rerank_score'] = score

    candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    results = []
    for i, c in enumerate(candidates[:FINAL_RESULTS]):
        meta = c['metadata']
        abstract = clean_text(meta.get('abstract', ''))
        sentences = re.split(r'(?<=[.!?]) +', abstract)
        best_sentence = ""
        if sentences:
            sentence_embeddings = embedder.encode(sentences, normalize_embeddings=True)
            q_emb = embedder.encode([query], normalize_embeddings=True)
            sims = util.cos_sim(sentence_embeddings, q_emb).squeeze(1)
            best_idx = int(sims.argmax())
            best_sentence = sentences[best_idx]

        results.append({
            "idx": c['idx'],
            "index": i + 1,
            "similarity": c['faiss_score'] * 100,
            "rerank_score": c['rerank_score'],
            "most_similar_sentence": best_sentence,
            "title": clean_text(meta.get('title', '')),
            "abstract": abstract,
            "patent_number": meta.get('patent_number', ''),
            "publication_date": meta.get('publication_date', ''),
            "application_number": meta.get('application_number', ''),
            "inventors": meta.get('inventors', ''),
            "assignee": meta.get('assignee', ''),
            "claims": meta.get('claims', ''),
            "description": meta.get('description', '')
        })
    return results

# ------------------- OpenAI RAG: Best Solution -------------------
def generate_best_solution(query, results):
    context_text = "\n\n".join([
        f"Patent {r['patent_number']}:\nTitle: {r['title']}\nAbstract: {r['abstract']}\nClaims: {r['claims'][:500]}\nDescription: {r['description'][:500]}"
        for r in results
    ])
    prompt = f"""
    You are a patent research assistant.
    Based on the following patents, identify the best solution that answers the user's question.
    Include the supporting patent number(s) in your answer.
    If no patent matches, say "I don't know".

    Context:
    {context_text}

    Question:
    {query}

    Best Solution:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert patent researcher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    if "I don't know" in answer and results:
        top = results[0]
        answer = f"I could not generate a confident direct answer. However, the most relevant patent seems to be {top['patent_number']} - {top['title']}."
    return answer

# ------------------- Follow-up Q&A -------------------
def ask_question_about_top_patents(question, top_patents):
    context_text = "\n\n".join([
        f"Patent {p['patent_number']}:\nTitle: {p['title']}\nAbstract: {p['abstract']}\nClaims: {p['claims']}\nDescription: {p['description']}"
        for p in top_patents
    ])
    prompt = f"""
    You are a patent expert.
    Based on the following top patents, answer the user's question.
    Use ONLY the information from these patents. If not found, say "I don't know".

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert patent researcher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ------------------- Streamlit UI -------------------
st.set_page_config(layout="wide")
st.title("🔍 Bay Patent Search")

query_col, icon_col = st.columns([9, 1])
with query_col:
    query = st.text_input("Search", placeholder="Enter your search query here...", label_visibility="collapsed")
with icon_col:
    st.write("")
    search_triggered = st.button("🔍")

all_results = []
if query or search_triggered:
    with st.spinner("Searching..."):
        all_results = search(query)

    with st.spinner("Generating best solution using AI"):
        best_solution = generate_best_solution(query, all_results)

    st.subheader("💡 AI Solution")
    st.markdown(best_solution)

    st.markdown(f"**Top {FINAL_RESULTS} results shown (from {TOP_K_RETRIEVE} retrieved)**")
    for result in all_results:
        st.markdown(f"**Why this result?**\n"
                    f"→ FAISS Similarity Score: {result['similarity']:.2f}%\n"
                    f"→ Rerank Score: {result['rerank_score']:.4f}\n"
                    f"→ Most Relevant Sentence: “{result['most_similar_sentence']}”")
        st.markdown(f"### {result['index']}. {result['title']}")
        st.markdown(f"**Abstract:** {result['abstract']}")
        st.markdown(f"""
        - **Patent Number**: {result['patent_number']}  
        - **Publication Date**: {result['publication_date']}  
        - **Application Number**: {result['application_number']}  
        - **Inventors**: {result['inventors']}  
        - **Assignee**: {result['assignee']}
        """)
        st.markdown("---")

# ------------------- Follow-up Q&A UI -------------------
if all_results:
    st.subheader("❓ Ask a question about these top patents")
    followup_question = st.text_input(
        "Enter your question here...",
        placeholder="Ask something specific about the top patents..."
    )
    ask_question_triggered = st.button("Ask GPT about top patents")

    if ask_question_triggered and followup_question:
        with st.spinner("GPT is answering based on top patents..."):
            answer = ask_question_about_top_patents(followup_question, all_results)
        st.markdown("### 📝 Answer from GPT")
        st.markdown(answer)
