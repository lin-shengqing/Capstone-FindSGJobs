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

    sctp_map = {
        "Data Scientist": {"Course": "SCTP: Advanced Data Science & AI", "Provider": "NTU PACE"},
        "Analyst": {"Course": "SCTP: Data Analytics & BI", "Provider": "SIT / SUSS"},
        "Engineer": {"Course": "SCTP: AI Engineering Specialist", "Provider": "SUTD Academy"},
        "Cloud": {"Course": "SCTP: Cloud Infrastructure Engineering", "Provider": "NTU / TP"}
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        raw_json = response.json()
        
        # --- FIXED DATA PARSING FOR FINDSGJOBS STRUCTURE ---
        # Structure: raw_json -> 'data' (dict) -> 'result' (list)
        source_data = []
        if isinstance(raw_json, dict) and 'data' in raw_json:
            inner_data = raw_json['data']
            if isinstance(inner_data, dict) and 'result' in inner_data:
                source_data = inner_data['result']
        
        if not source_data:
            raise ValueError("Could not locate 'data -> result' list in API response")

        # Now we slice the LIST
        final_list = source_data[:50] 
        
        jobs_list = []
        for item in final_list:
            # Accessing the nested 'job' dictionary within each result item
            job_info = item.get('job', {})
            
            role = job_info.get('Title', 'Unknown Role')
            # Using 'keywords' or 'JobDescription' as the basis for AI matching
            skills_text = job_info.get('keywords', 'Details in description')
            desc = job_info.get('JobDescription', '')
            
            # Match to an SCTP course based on keywords in the title
            course_info = sctp_map["Analyst"] # Default
            for key, val in sctp_map.items():
                if key.lower() in str(role).lower():
                    course_info = val
                    break

            jobs_list.append({
                "Role": role,
                "Skills": skills_text,
                "Description": desc,
                "Course": course_info["Course"],
                "Provider": course_info["Provider"]
            })
            
        return pd.DataFrame(jobs_list), "API"
    
    except Exception as e:
        # Fallback dataset
        fallback_df = pd.DataFrame([
            {"Role": "Data Scientist", "Skills": "Python, ML, SQL", "Description": "AI Research", "Course": "SCTP: DS&AI", "Provider": "NTU"},
            {"Role": "AI Engineer", "Skills": "LLMs, Python, NLP", "Description": "GenAI Dev", "Course": "SCTP: AI Specialist", "Provider": "SUTD"},
            {"Role": "Data Analyst", "Skills": "Excel, SQL, Tableau", "Description": "Business Insights", "Course": "SCTP: Data Analytics", "Provider": "SIT"}
        ])
        return fallback_df, f"Fallback (Error: {str(e)})"

# --- 3. INITIALIZATION ---
with st.sidebar:
    st.title("⚙️ System Status")
    with st.status("Initializing AI & Data...", expanded=True) as status:
        st.write("Loading Transformer Model...")
        model = load_model()
        st.write("Connecting to FindSGJobs API...")
        jobs_df, source_type = fetch_jobs_data()
        
        if "API" in source_type:
            st.success(f"Successfully pulled {len(jobs_df)} roles from Live API.")
        else:
            st.warning("API Switched to Fallback Mode.")
            
        status.update(label="System Ready!", state="complete", expanded=False)

    with st.expander("Data Connection Details"):
        st.write(f"**Data Source:** {source_type}")
        st.write(f"**API URL:** https://www.findsgjobs.com/apis/job/searchable")
        if "API" in source_type:
            st.write("🟢 Connection Stable")
        else:
            st.write("🔴 Using Offline Recovery Mode")

    if not hf_token:
        st.warning("HF_TOKEN missing. Performance may be throttled.")

# --- 4. NAVIGATION ---
st.sidebar.title("🛠️ Project Controls")
nav = st.sidebar.radio("Navigation", ["Introduction", "AI Recommendation Engine", "Impact & Ethics"])

# --- 5. PAGES ---
if nav == "Introduction":
    st.title("🎯 Bridging the Singapore Skill Gap")
    st.write("### The SCTP Capstone Project")
    
    if "API" in source_type:
        st.caption("🟢 Connected to Live FindSGJobs Feed")
    else:
        st.caption("🟡 Running on local fallback dataset")

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
    
    example_resume = """I am a highly motivated professional looking to transition into Data Science. 
I have 3 years of experience in project management. I am proficient in Excel and have 
recently completed a basic course in Python and SQL. I am interested in data 
visualization using Tableau and building predictive models."""

    with c1:
        st.subheader("Candidate Input")
        upload = st.file_uploader("Upload Resume (PDF)", type="pdf")
        
        user_input = st.text_area(
            "Or Paste Experience Manually", 
            value=example_resume,
            height=250,
            help="You can edit this text or paste your own resume content."
        )
        
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
                # Using 'Description' which we mapped from 'JobDescription' in the API
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
    * **Scalability:** Real-time API consumption demonstrated via FindSGJobs endpoints.
    * **PDPA:** No persistent storage of PII (Personally Identifiable Information).
    * **Fairness:** Matching is based on semantic capability, reducing human bias in shortlisting.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*9HEn98S-0Z_0_KkAtf_v9A.png", caption="AI Vector Space Mapping")