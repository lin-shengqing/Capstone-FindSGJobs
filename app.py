import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
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
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADERS & AI MODEL (with Spinners) ---
@st.cache_resource
def load_model():
    # This might take a moment on the first run
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def fetch_jobs_data():
    api_url = "https://www.findsgjobs.com/apis/job/searchable"
    sctp_map = {
        "Data Scientist": {"Course": "SCTP: Advanced Data Science & AI", "Provider": "NTU PACE"},
        "Analyst": {"Course": "SCTP: Data Analytics & BI", "Provider": "SIT / SUSS"},
        "Engineer": {"Course": "SCTP: AI Engineering Specialist", "Provider": "SUTD Academy"},
        "Cloud": {"Course": "SCTP: Cloud Infrastructure Engineering", "Provider": "NTU / TP"}
    }

    try:
        # Reduced timeout to 5s to prevent hanging
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        raw_data = response.json()
        
        jobs_list = []
        source_data = raw_data.get('data', raw_data) if isinstance(raw_data, dict) else raw_data
        
        for item in source_data[:50]:
            role = item.get('job_title', item.get('title', 'Unknown Role'))
            skills_text = item.get('skills', item.get('requirements', 'Details in description'))
            desc = item.get('job_description', item.get('description', ''))
            
            course_info = sctp_map["Analyst"] # Default
            for key, val in sctp_map.items():
                if key.lower() in role.lower():
                    course_info = val
                    break

            jobs_list.append({
                "Role": role,
                "Skills": skills_text,
                "Description": desc,
                "Course": course_info["Course"],
                "Provider": course_info["Provider"]
            })
        return pd.DataFrame(jobs_list)
    
    except Exception as e:
        # Fallback data if API fails or times out
        return pd.DataFrame([
            {"Role": "Data Scientist", "Skills": "Python, ML, SQL", "Description": "AI Research", "Course": "SCTP: DS&AI", "Provider": "NTU"},
            {"Role": "AI Engineer", "Skills": "LLMs, Python, NLP", "Description": "GenAI Dev", "Course": "SCTP: AI Specialist", "Provider": "SUTD"},
            {"Role": "Data Analyst", "Skills": "Excel, SQL, Tableau", "Description": "Business Insights", "Course": "SCTP: Data Analytics", "Provider": "SIT"}
        ])

# --- 3. INITIALIZATION ---
# Show a status message while things load to avoid the "Blank Screen"
with st.sidebar:
    st.title("⚙️ System Status")
    with st.status("Initializing AI & Data...", expanded=True) as status:
        st.write("Loading Transformer Model...")
        model = load_model()
        st.write("Connecting to FindSGJobs API...")
        jobs_df = fetch_jobs_data()
        status.update(label="System Ready!", state="complete", expanded=False)

    if not hf_token:
        st.warning("HF_TOKEN missing. Performance may be throttled.")

# --- 4. NAVIGATION ---
st.sidebar.title("🛠️ Project Controls")
nav = st.sidebar.radio("Navigation", ["Introduction", "AI Recommendation Engine", "Impact & Ethics"])

# --- 5. PAGES ---
if nav == "Introduction":
    st.title("🎯 Bridging the Singapore Skill Gap")
    st.write("### The SCTP Capstone Project")
    st.info(f"Loaded {len(jobs_df)} active roles for analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **System Features:**
        * **Live Integration:** Real-time job feeds from FindSGJobs.
        * **Semantic Matching:** AI understands context, not just keywords.
        * **Gap Analysis:** Immediate mapping to 2026 SCTP curriculum.
        """)
    with col2:
        df_funnel = pd.DataFrame({
            'Stages': ["Total Applicants", "Skills Match", "Interview Ready", "Hired"],
            'Count': [1000, 450, 120, 30]
        })
        fig = px.funnel(df_funnel, x='Count', y='Stages')
        st.plotly_chart(fig, use_container_width=True)

elif nav == "AI Recommendation Engine":
    st.title("🚀 Live AI Matching Demo")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Candidate Input")
        upload = st.file_uploader("Upload Resume (PDF)", type="pdf")
        user_input = st.text_area("Or Paste Experience Manually", height=200)
        run_btn = st.button("Analyze My Career Profile")

    if run_btn:
        profile_text = ""
        if upload:
            reader = PdfReader(upload)
            for page in reader.pages:
                profile_text += page.extract_text()
        else:
            profile_text = user_input

        if profile_text:
            with st.spinner("AI is calculating vector similarities..."):
                # Combine title and description for richer context
                job_strings = (jobs_df['Role'] + " " + jobs_df['Description']).tolist()
                user_emb = model.encode(profile_text, convert_to_tensor=True)
                job_embs = model.encode(job_strings, convert_to_tensor=True)
                
                # Compute Similarity
                scores = util.cos_sim(user_emb, job_embs)[0].tolist()
                jobs_df['Score'] = [round(float(s) * 100, 1) for s in scores]
                results = jobs_df.sort_values(by="Score", ascending=False).head(3)
                
            with c2:
                st.subheader("Top Live Matches")
                for _, row in results.iterrows():
                    with st.expander(f"⭐ {row['Role']} ({row['Score']}% Match)", expanded=True):
                        # Gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = row['Score'],
                            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"}},
                            domain = {'x': [0, 1], 'y': [0, 1]}
                        ))
                        fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Skill Gaps
                        req_skills = [s.strip().lower() for s in str(row['Skills']).split(",")]
                        missing = [s.upper() for s in req_skills if s not in profile_text.lower() and len(s) > 2]
                        
                        if missing[:3]:
                            st.error(f"**Gaps:** {', '.join(missing[:3])}")
                            st.success(f"**Pathway:** {row['Course']} @ {row['Provider']}")
                        else:
                            st.success("Your profile is a strong technical match!")
        else:
            st.warning("Please provide a resume or profile text.")

elif nav == "Impact & Ethics":
    st.title("🛡️ Implementation & Data Ethics")
    st.markdown("""
    * **Scalability:** Real-time API consumption.
    * **PDPA:** No persistent storage of PII (Personally Identifiable Information).
    * **Fairness:** Matching is based on semantic capability, reducing human bias in shortlisting.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*9HEn98S-0Z_0_KkAtf_v9A.png", caption="AI Vector Space Mapping")