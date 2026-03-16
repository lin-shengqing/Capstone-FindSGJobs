import os
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
import streamlit as st

# --- 0. ENVIRONMENT SETUP ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="FindSGJobs AI: SCTP Prototype", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; }
    .rag-advice-box {
        background: linear-gradient(135deg, #e8f4f8, #d6eaf8);
        border-left: 4px solid #2980b9;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 12px;
        font-size: 0.97rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADERS & AI MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def fetch_jobs_data():
    api_url = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    try:
        with open('sctp_courses.json', 'r') as f:
            sctp_courses = json.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading sctp_courses.json: {e}")
        sctp_courses = {}

    prelaoded_roles = []
    try:
        if os.path.exists('roles.json'):
            with open('roles.json', 'r') as f:
                prelaoded_roles = json.load(f)
    except Exception:
        prelaoded_roles = []

    api_jobs = []
    is_live = False
    api_status_msg = "Offline"

    try:
        response = requests.get(api_url, headers=headers, timeout=1)
        response.raise_for_status()
        raw_json = response.json()

        source_data = []
        if isinstance(raw_json, dict) and 'data' in raw_json:
            inner_data = raw_json['data']
            if isinstance(inner_data, dict) and 'result' in inner_data:
                source_data = inner_data['result']

        if source_data:
            is_live = True
            for item in source_data:
                job_info = item.get('job', {})
                role = job_info.get('Title', 'Unknown Role')
                skills_text = job_info.get('keywords', 'Details in description')
                desc = job_info.get('JobDescription', '')

                course_info = {"Course": "SCTP: General Career Transition", "Provider": "SkillsFuture Singapore"}
                for key in sorted(sctp_courses.keys(), key=len, reverse=True):
                    if key.lower() in str(role).lower():
                        course_info = sctp_courses[key]
                        break

                api_jobs.append({
                    "Role": role, "Skills": skills_text, "Description": desc,
                    "Course": course_info["Course"], "Provider": course_info["Provider"]
                })
    except Exception as e:
        api_status_msg = str(e)
        is_live = False

    combined_df = pd.DataFrame(api_jobs + prelaoded_roles)
    return combined_df, is_live, api_status_msg, len(api_jobs), len(prelaoded_roles)


# --- 3. RAG: BUILD FAISS VECTOR INDEX ---
@st.cache_resource
def build_faiss_index(_model, _jobs_df):
    """
    RAG - RETRIEVAL COMPONENT:
    Builds a FAISS flat inner-product index over all job documents.
    Each document is a concatenation of Role + Skills + Description so that
    the index captures rich semantic content for retrieval.
    """
    documents = []
    for _, row in _jobs_df.iterrows():
        doc = f"Role: {row['Role']}. Skills: {row['Skills']}. Description: {row['Description']}"
        documents.append(doc)

    # Encode all documents into embeddings
    embeddings = _model.encode(documents, convert_to_numpy=True, show_progress_bar=False)

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, documents


# --- 4. RAG: WEIGHTED RE-RANKING ---
def weighted_rerank(profile_text, top_k_indices, jobs_df, model):
    """
    After FAISS retrieves Top-K candidates, apply weighted re-ranking
    on Title / Skills / Description similarities for higher precision.
    """
    subset_df = jobs_df.iloc[top_k_indices].copy()

    user_emb = model.encode(profile_text, convert_to_tensor=True)

    title_embs = model.encode(subset_df['Role'].tolist(), convert_to_tensor=True)
    skill_embs = model.encode(subset_df['Skills'].tolist(), convert_to_tensor=True)
    desc_embs  = model.encode(subset_df['Description'].tolist(), convert_to_tensor=True)

    title_scores = util.cos_sim(user_emb, title_embs)[0]
    skill_scores = util.cos_sim(user_emb, skill_embs)[0]
    desc_scores  = util.cos_sim(user_emb, desc_embs)[0]

    raw_scores = (title_scores * 0.5) + (skill_scores * 0.3) + (desc_scores * 0.2)
    calibrated = [min(float(s) * 2.2, 0.98) for s in raw_scores]

    subset_df['Score'] = [round(s * 100, 1) for s in calibrated]
    return subset_df.sort_values(by="Score", ascending=False).head(3)


# --- 5. RAG: LLM GENERATION via Hugging Face Inference Providers ---
def generate_career_advice(profile_text: str, top_jobs: pd.DataFrame) -> str:
    """
    RAG - GENERATION COMPONENT:
    Constructs a prompt from the retrieved job context and the candidate profile,
    then calls the HF Inference Providers API (InferenceClient) to generate
    personalised career advice. Uses auto provider selection so HF picks the
    fastest available backend for the chosen model.
    """
    if not hf_token:
        return "⚠️ HF_TOKEN not set. Please add it to your .env file to enable AI-generated advice."

    # Build context string from retrieved documents
    context_lines = []
    for _, row in top_jobs.iterrows():
        context_lines.append(
            f"- {row['Role']} (Match: {row['Score']}%): requires {row['Skills']}. "
            f"Recommended upskilling: {row['Course']} @ {row['Provider']}."
        )
    context = "\n".join(context_lines)

    system_msg = (
        "You are a helpful and encouraging career advisor in Singapore "
        "specialising in mid-career transitions. Be concise and actionable."
    )
    user_msg = (
        f"A candidate has the following professional background:\n{profile_text[:800]}\n\n"
        f"Based on their profile, the top matching job roles are:\n{context}\n\n"
        "In 2-3 concise sentences, give this candidate personalised career transition advice. "
        "Focus on their strengths, the most relevant role, and one concrete next step."
    )

    # Models to try — using widely supported open weights on HF free tier
    models_to_try = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ]

    last_error = ""
    for model in models_to_try:
        try:
            # Let huggingface_hub use its default inference routing
            client = InferenceClient(api_key=hf_token)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=250,
                temperature=0.7,
            )
            advice = completion.choices[0].message.content.strip()
            if advice:
                return f"[{model.split('/')[1]}] {advice}"
        except Exception as e:
            last_error = str(e)
            continue

    return f"⚠️ HF API over capacity or model unsupported. Last error: {last_error}"


# --- 6. INITIALIZATION ---
with st.sidebar:
    st.title("⚙️ System Status")
    with st.status("Initializing RAG Pipeline...", expanded=False) as status:
        st.write("Loading Sentence Transformer Model...")
        model = load_model()
        st.write('Model "all-MiniLM-L6-v2" Loaded ✅')

        st.write("Loading job data...")
        jobs_df, is_live, api_error, api_count, seed_count = fetch_jobs_data()
        st.write(f"{len(jobs_df)} roles loaded ✅")

        st.write("Building FAISS vector index...")
        faiss_index, job_documents = build_faiss_index(model, jobs_df)
        st.write(f"FAISS index built ({faiss_index.ntotal} vectors) ✅")

        status.update(label="RAG Pipeline Ready!", state="complete", expanded=False)

    with st.expander("📊 Data Breakdown", expanded=False):
        st.write(f"**Total Roles in Index:** {len(jobs_df)}")
        st.write(f"**Vector Dimensions:** 384 (MiniLM)")
        st.write(f"**Index Type:** FAISS Flat Inner-Product")
        st.write(f"**LLM:** Mistral-7B-Instruct (HF API)")

    if not hf_token:
        st.warning("HF_TOKEN missing. LLM-generated advice will be disabled.")

# --- 7. NAVIGATION ---
st.sidebar.title("🛠️ Project Controls")
nav = st.sidebar.radio("Navigation", ["AI Recommendation Engine", "Project Description"])

# --- 8. PAGES ---
if nav == "AI Recommendation Engine":
    st.title("🎯 AI-Driven Job Recommendation & SCTP Skill Gap Analyzer")
    st.caption("Powered by RAG: FAISS Semantic Retrieval + Mistral-7B Generation")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Candidate Input")
        upload = st.file_uploader("Upload Resume (PDF)", type="pdf")
        user_input = st.text_area("Or Paste Experience Manually", height=250)
        run_btn = st.button("Analyze My Career Profile")

        st.markdown("### Example job experience:")
        st.markdown(""" I am a highly motivated professional looking to transition into Data Science and AI.  
                    I have 13 years of experience in software development, Site Reliability Engineering.  
                    I am proficient in ASP.NET Microsoft SQL and have recently completed a basic course in Python and Machine Learning.  
                    I am also interested in Big Data and Machine Learning. """)

    if run_btn:
        profile_text = ""
        if upload:
            try:
                reader = PdfReader(upload)
                for page in reader.pages:
                    profile_text += page.extract_text()
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        else:
            profile_text = user_input

        if profile_text:
            with st.spinner("🔍 Step 1/2 — RAG Retrieval: searching FAISS index..."):
                # --- RAG RETRIEVAL: encode query and search FAISS ---
                query_emb = model.encode([profile_text], convert_to_numpy=True)
                faiss.normalize_L2(query_emb)

                TOP_K = 15  # retrieve wider pool, then re-rank
                distances, indices = faiss_index.search(query_emb, TOP_K)
                top_k_indices = indices[0].tolist()

                # --- WEIGHTED RE-RANKING on the retrieved subset ---
                results = weighted_rerank(profile_text, top_k_indices, jobs_df, model)

            with c2:
                st.subheader("🏆 Top Matched Roles")
                for _, row in results.iterrows():
                    with st.expander(f"⭐ {row['Role']} ({row['Score']}% Match)", expanded=True):
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=row['Score'],
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"}}
                        ))
                        fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                        req_skills = [s.strip().lower() for s in str(row['Skills']).split(",")]
                        missing = [s.upper() for s in req_skills if s not in profile_text.lower() and len(s) > 2]

                        if missing[:3]:
                            st.error(f"**Skill Gaps:** {', '.join(missing[:3])}")
                            st.success(f"**Upskilling Pathway:** {row['Course']} @ {row['Provider']}")
                        else:
                            st.success("✅ Your profile is a strong technical match!")

            # --- RAG GENERATION: call Mistral-7B for personalised advice ---
            with st.spinner("🤖 Step 2/2 — RAG Generation: Mistral-7B crafting personalised advice..."):
                advice = generate_career_advice(profile_text, results)

            st.markdown("---")
            st.subheader("🧠 AI Career Advisor (RAG Generated)")
            st.markdown(
                f'<div class="rag-advice-box">💬 {advice}</div>',
                unsafe_allow_html=True
            )

elif nav == "Project Description":
    st.title("🎯 AI-Driven Job Recommendation & SCTP Skill Gap Analyzer")
    st.markdown("### Use the **🛠️ Project Controls** on the left menu to navigate.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **RAG Pipeline Architecture:**
        * **Ingestion:** Curated seed catalog embedded into a FAISS vector index at startup.
        * **Retrieval:** Candidate profile encoded & searched against FAISS index (Top-K retrieval).
        * **Re-Ranking:** Weighted cosine similarity (Title 50% · Skills 30% · Desc 20%).
        """)
        st.markdown("""
        **LLM Generation Layer (AI Career Advisor):**
        * A prompt is constructed from the retrieved data and sent to the **Hugging Face Inference API**.
        * To ensure stability, the system uses a dynamic fallback array of top open-weight models:
            1. **`Llama-3.1-8B-Instruct`** (Primary) - Fast & highly capable.
            2. **`Qwen2.5-72B-Instruct`** (Fallback 1) - Massive, GPT-4 level reasoning.
            3. **`Mistral-Nemo-Instruct-2407`** (Fallback 2) - Highly efficient 12B model.
        """)
        st.title("🛡️ Implementation & Data Ethics")
        st.markdown("""
        * **Transparency:** Skill gap reporting provides actionable feedback.
        * **Privacy:** Vectorised matching ensures PII is never stored.
        * **PDPA Aligned:** Resume processed in-memory only.
        * **Bias Reduction:** Mathematical (vector-based) matching only.
        """)
        st.title("Description")
        st.markdown("""An SCTP participant might struggle to find a job due to **skill gaps** and **insufficient relevant experience**, high competition for roles, or a mismatch between their qualifications and employer needs.

---

## Skill and Experience Mismatches

* **Skill Gaps:** Participants may not have the specific skills required for the jobs they are applying for.
* **Lack of Experience:** Employers frequently cite a lack of relevant experience as a primary reason for being unable to fill positions, particularly for individuals with new or changing careers.
* **Qualification Mismatch:** Some roles may require a degree or specific certifications that participants don't possess.

---

## Job Search and Application Issues

* **Outdated Materials:** An outdated resume or cover letter can make a candidate seem unprepared.
* **Poor Interview Skills:** Ineffective communication or poor performance during job interviews can hinder a candidate's chances.
* **Ineffective Strategies:** Relying only on job boards or a lack of a clear job search strategy can limit opportunities.
* **Competition:** A highly competitive job market can make it difficult for individuals to secure roles.

---

## External and Systemic Factors

* **Economic Conditions:** Broader economic factors can influence employment outcomes.
* **Networking:** A lack of professional contacts and internal referrals can disadvantage job seekers.
* **Unclear Expectations:** Confusing application processes or unclear job descriptions can make it difficult for candidates to understand what employers are looking for.
                    """)
