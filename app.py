import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import json

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

# --- 2. DATA LOADERS & AI MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def fetch_jobs_data():
    api_url = "https://www.findsgjobs.com/apis/job/searchable"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    # --- 1. LOAD HARDCODED ROLES FROM JSON FILE ---
    hardcoded_roles = []
    try:
        with open('roles.json', 'r') as f:
            hardcoded_roles = json.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading roles.json: {e}")
        # Emergency minimal fallback if file is missing
        hardcoded_roles = [{"Role": "System Admin", "Skills": "Linux", "Description": "IT Support", "Course": "SCTP: IT Support", "Provider": "TP"}]

    # --- 2. FETCH LIVE API DATA ---
    api_jobs = []
    api_status_msg = "Online"
    is_live = False
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        raw_json = response.json()
        
        source_data = []
        if isinstance(raw_json, dict) and 'data' in raw_json:
            inner_data = raw_json['data']
            if isinstance(inner_data, dict) and 'result' in inner_data:
                source_data = inner_data['result']
        
        if source_data:
            is_live = True
            # Mapping API fields to our internal format
            for item in source_data:
                job_info = item.get('job', {})
                role = job_info.get('Title', 'Unknown Role')
                skills_text = job_info.get('keywords', 'Details in description')
                desc = job_info.get('JobDescription', '')
                
                # Logic to map API roles to SCTP (simplified)
                api_jobs.append({
                    "Role": role,
                    "Skills": skills_text,
                    "Description": desc,
                    "Course": "SCTP: Professional Transition Program", # Default for API
                    "Provider": "Multiple Providers"
                })
    except Exception as e:
        api_status_msg = f"Error: {str(e)[:50]}"
        is_live = False

    combined_df = pd.DataFrame(api_jobs + hardcoded_roles)
    return combined_df, is_live, api_status_msg, len(api_jobs), len(hardcoded_roles)

# --- 3. SYSTEM INITIALIZATION ---
with st.sidebar:
    st.title("⚙️ System Status")
    with st.status("Initializing AI & Data...", expanded=True) as status:
        st.write("Loading Transformer Model...")
        model = load_model()
        st.write("Fetching Hybrid Job Feed...")
        jobs_df, is_live, api_error, api_count, seed_count = fetch_jobs_data()
        
        if is_live:
            st.success(f"Successfully pulled {api_count} roles from Live API.")
            status.update(label="Live Connection Active!", state="complete", expanded=False)
        else:
            st.warning("API Switched to Fallback Mode.")
            status.update(label="Offline Mode Active", state="error", expanded=True)

    with st.expander("📊 Data Breakdown", expanded=True):
        st.write(f"**Total Roles Loaded:** {len(jobs_df)}")
        st.write(f"**API Status:** {'🟢 Online' if is_live else '🔴 Offline'}")
        if not is_live:
            st.caption(f"Reason: {api_error}")
        st.write(f"**Live API Roles:** {api_count}")
        st.write(f"**Hardcoded Seed Roles:** {seed_count}")

    if not hf_token:
        st.warning("HF_TOKEN missing. Performance may be throttled.")

# --- 4. NAVIGATION ---
st.sidebar.title("🛠️ Project Controls")
nav = st.sidebar.radio("Navigation", ["Introduction", "AI Recommendation Engine", "Impact & Ethics"])

# --- 5. PAGES ---
if nav == "Introduction":
    st.title("🎯 Bridging the Singapore Skill Gap")
    st.write("### The SCTP Capstone Project")
    
    if is_live:
        st.caption("🟢 Connected to Live FindSGJobs Feed + Seed Catalog")
    else:
        st.caption("🟡 Running on local fallback dataset only")

    st.info(f"Loaded {len(jobs_df)} active roles for analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **System Features:**
        * **Hybrid Ingestion:** Merging live API data with a diverse curated seed catalog.
        * **Fault Tolerance:** Automatic fallback to local data if the API is unreachable.
        * **Semantic Matching:** NLP-driven understanding of candidate suitability.
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
    
    example_resume = """I am a highly motivated professional looking to transition into Data Science. 
I have 3 years of experience in project management. I am proficient in Excel and have 
recently completed a basic course in Python and SQL. I am interested in data 
visualization using Tableau and building predictive models."""

    with c1:
        st.subheader("Candidate Input")
        upload = st.file_uploader("Upload Resume (PDF)", type="pdf")
        user_input = st.text_area("Or Paste Experience Manually", value=example_resume, height=250)
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
                job_strings = (jobs_df['Role'] + " " + jobs_df['Description']).tolist()
                user_emb = model.encode(profile_text, convert_to_tensor=True)
                job_embs = model.encode(job_strings, convert_to_tensor=True)
                
                scores = util.cos_sim(user_emb, job_embs)[0].tolist()
                jobs_df['Score'] = [round(float(s) * 100, 1) for s in scores]
                results = jobs_df.sort_values(by="Score", ascending=False).head(3)
                
            with c2:
                st.subheader(f"Top Matches Found")
                for _, row in results.iterrows():
                    with st.expander(f"⭐ {row['Role']} ({row['Score']}% Match)", expanded=True):
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = row['Score'],
                            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"}},
                            domain = {'x': [0, 1], 'y': [0, 1]}
                        ))
                        fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                        
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
    * **Resilience:** Demonstrated hybrid architecture ensures system availability.
    * **Transparency:** Skill gap reporting provides actionable feedback to the user.
    * **Privacy:** Vectorized matching ensures PII is never stored or transmitted.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*9HEn98S-0Z_0_KkAtf_v9A.png", caption="AI Vector Space Mapping")