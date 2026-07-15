import os
import tempfile
import streamlit as st
import requests

st.set_page_config(page_title="Candidate Mode", page_icon="👤", layout="wide")

st.title("👤 Candidate Mode")
st.write("Upload your resume and paste a job description to get your match result.")
st.markdown("---")

MAX_MB = 5

st.subheader("1. Upload Your Resume (PDF)")
uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

st.subheader("2. Paste the Job Description")
jd_text = st.text_area("Job Description", height=200, placeholder="Paste the full job description here...")

if st.button("🔍 Predict Match", type="primary"):
    if uploaded is None:
        st.error("Please upload a resume PDF.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()
    if uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"File too large. Maximum size is {MAX_MB} MB.")
        st.stop()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Analyzing your resume..."):
            with open(tmp_path, "rb") as f:
                response = requests.post(
                    "http://localhost:8000/predict",
                    files={"pdf": ("resume.pdf", f, "application/pdf")},
                    data={"jd_text": jd_text.strip()},
                )
            result = response.json()

    except Exception as e:
        st.error(f"Could not process the PDF: {e}")
        st.stop()

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    st.markdown("---")
    score      = result["match_score"]
    prediction = result["prediction"]
    confidence = round(abs(score - 50) * 2, 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Match Score", f"{score}%")

    with col2:
        st.metric("Model Confidence", f"{confidence}%",
                help="Distance from the 50% decision boundary × 2.")

    with col3:
        if prediction == "Good Fit":
            st.success("✅ Good Fit")
        else:
            st.error("❌ No Fit")

    st.caption(
        f"Match Score ≥ 50% → **Good Fit**, below 50% → **No Fit**. "
        f"Confidence: {'Strong' if confidence >= 60 else 'Moderate' if confidence >= 30 else 'Weak'}."
    )
