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

## 🧠 Technical Methodology (RAG Pipeline)

The system operates as a full **Retrieval-Augmented Generation (RAG)** pipeline, moving beyond simple keyword matching to provide semantic understanding and personalized AI generation.

### 1. Vector Store & Semantic Retrieval
Instead of keyword matching, this tool utilizes the **`all-MiniLM-L6-v2`** Sentence-Transformer model. At startup, it converts all curated job requirements into high-dimensional vectors and stores them in a **FAISS (Facebook AI Similarity Search)** index. When a candidate uploads a resume, the system queries this FAISS index to retrieve the most semantically relevant roles via Inner-Product (Cosine Similarity).

### 2. Weighted Similarity Re-Scoring
To ensure high precision on the retrieved candidates, the engine applies a weighted ranking algorithm:
* **Job Title (50%):** Prioritizes the candidate's career direction.
* **Technical Skills (30%):** Matches specific tools (Python, SQL, etc.).
* **Job Description (20%):** Captures broad professional context.

### 3. Score Calibration (UX Optimization)
Raw semantic similarity coefficients typically fall between 0.3 and 0.5. The app applies a **linear calibration layer** to map these raw scores to a 0–100% human-legible scale, ensuring the "Match Percentage" aligns with user expectations.

### 4. LLM Generation Layer (AI Career Advisor)
Once the top matches and skill gaps are identified, the system constructs a context-rich prompt and calls the **Hugging Face Inference Providers API** (`huggingface_hub`). A generative Large Language Model analyzes the candidate's specific background against the retrieved job requirements and streams back personalized, actionable career transition advice.

**Model Selection:**
The system uses a dynamic fallback array of open-weight LLMs hosted on Hugging Face's free Serverless Inference API to ensure stability and high reasoning quality:
1. **`meta-llama/Llama-3.1-8B-Instruct`** (Primary): Highly capable, fast, and excellent at following complex system prompts.
2. **`Qwen/Qwen2.5-72B-Instruct`** (Fallback 1): A massive, exceptionally powerful model that rivals proprietary models in reasoning.
3. **`mistralai/Mistral-Nemo-Instruct-2407`** (Fallback 2): A highly efficient 12B parameter model built jointly by Mistral and Nvidia.

### 5. Data Ingestion
* **Seed Database:** Utilizes a custom `roles.json` containing 100+ diverse industry roles.
* **Course Database:** Utilizes `sctp_courses.json` containing SCTP courses to fulfill skill gaps. 

---

### 📁 File Structure

* **app.py**: Core application logic, RAG pipeline, FAISS indexing, and Streamlit UI.
* **roles.json**: Curated database of 100 diverse job roles and requirements.
* **sctp_courses.json**: Mapping of job categories to actual SCTP course providers (NTU, SUTD, SIT, etc.).
* **requirements.txt**: Python dependencies (Sentence-Transformers, PyPDF, FAISS, huggingface-hub).

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
