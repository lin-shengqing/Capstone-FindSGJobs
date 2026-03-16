# 🚀 AI-Driven Job Recommendation & SCTP Skill Gap Analyzer

## 📌 Project Overview
A functional AI prototype built to solve the job-matching friction in the Singapore market. The application identifies technical gaps between a candidate's resume and current job requirements, then maps those gaps to specific **2026 SCTP training modules**.

---

## 🎯 Problem Statement
Singaporean job seekers, especially mid-career professionals, often face:
* **Hidden Skill Gaps:** Difficulty identifying exactly which technical delta prevents them from landing a role.
* **Contextual Mismatches:** Standard keyword-based search fails to recognize transferable skills from previous industries.
* **Upskilling Fragmentation:** Difficulty mapping job requirements to the correct SkillsFuture Career Transition Programme (SCTP).

---

## 🧠 Technical Methodology

### 1. Semantic Vector Search
Unlike keyword matching, this tool utilizes the **`all-MiniLM-L6-v2`** Sentence-Transformer model. It converts resume text and job descriptions into high-dimensional vectors, measuring the **Cosine Similarity** between them to understand intent and context.

### 2. Weighted Similarity Scoring
To ensure high precision, the recommendation engine employs a weighted ranking algorithm:
* **Job Title (50%):** Prioritizes the candidate's career direction.
* **Technical Skills (30%):** Matches specific tools (Python, SQL, etc.).
* **Job Description (20%):** Captures broad professional context.

### 3. Score Calibration (UX Optimization)
Raw semantic similarity coefficients typically fall between 0.3 and 0.5. The app applies a **linear calibration layer** to map these raw scores to a 0–100% human-legible scale, ensuring the "Match Percentage" aligns with user expectations while maintaining ranking integrity.

### 4. Data Ingestion
The system architecture is decoupled for resilience:
* **Seed Database:** Utilizes a custom `roles.json` containing 100+ diverse industry roles to ensure high-quality recommendations even when offline. 
* **Course Database:** Utilizes a custom `sctp_course.json` containing 20+ diverse SCTP courses to fill up the skill gaps and insufficient relevant experience. 

---

### 📁 File Structure

* **app.py**: Core application logic, NLP pipeline, and Streamlit UI.
* **roles.json**: Curated database of 100 diverse job roles and requirements.
* **sctp_courses.json**: Mapping of job categories to actual SCTP course providers (NTU, SUTD, SIT, etc.).
* **requirements.txt**: Python dependencies (Sentence-Transformers, PyPDF, Plotly).

---

### 🛡️ Ethics & Data Privacy

---

* **PDPA Aligned**: All resume processing is done in-memory. No personal data or uploaded PDFs are stored permanently.
* **Bias Reduction**: Matching is purely mathematical (vector-based), ignoring demographic indicators such as age, gender, or race.
* **Transparency**: Clearly identifies **"Skill Gaps"** to provide users with actionable feedback rather than a simple "Yes/No" result.
* **Industry Partner**: FindSGJobs

## 🛠️ Installation (Conda Workflow)

1. **Create the environment**:
   ```bash
   conda env create -f environment.yml
2. **Activate the Environment**:
   ```bash
   conda activate capstone_findsgjobs
3. **Add .env file for Hugging Face HF_TOKEN**:
   ```bash
    HF_TOKEN=hf_your_actual_token_here
For Render, add HF_TOKEN in Add Environment Variable

4. **Run the streamlit app**:
   ```bash
   streamlit run app.py
