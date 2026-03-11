# 🚀 AI Job Matcher & SCTP Skill Gap Analyzer
**Capstone Project FindSGJobs**

## 📌 Project Overview
A functional AI prototype built to solve the job-matching friction in the Singapore market. The application identifies technical gaps between a candidate's resume and current job requirements, then maps those gaps to specific **2026 SCTP training modules**.

## 🛠️ Installation (Conda Workflow)

1. **Create the environment**:
   ```bash
   conda env create -f environment.yml

2. **Activate the Environment**:
   ```bash
   conda activate capstone_findsgjobs

3. **Add .env file for HF_TOKEN**:
   ```bash
    HF_TOKEN=hf_your_actual_token_here

For Render, add HF_TOKEN in Add Environment Variable

4. **Run the app**:
   ```bash
   streamlit run app.py