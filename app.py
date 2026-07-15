import streamlit as st

st.set_page_config(page_title="TwinFit AI", page_icon="📄", layout="wide")
st.title("📄 TwinFit AI — Resume Intelligence Platform")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.info("### 👤 Candidate Mode\nUpload your resume PDF and paste a job description to see your match score and prediction.")
    st.page_link("pages/1_Candidate_Mode.py", label="Go to Candidate Mode →")

with col2:
    st.info("### 🏢 Recruiter Dashboard\nUpload multiple resumes, rank candidates by match score, filter by minimum score and top N.")
    st.page_link("pages/2_Recruiter_Dashboard.py", label="Go to Recruiter Dashboard →")

st.markdown("---")
st.caption("Model: Siamese Neural Network · Embeddings: all-MiniLM-L6-v2")