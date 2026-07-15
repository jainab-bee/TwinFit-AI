import os
import sys
import io
import pandas as pd
import streamlit as st
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="Recruiter Dashboard", page_icon="🏢", layout="wide")

st.title("🏢 Recruiter Dashboard")
st.write("Upload multiple resumes, rank candidates, and filter by score.")
st.markdown("---")

st.subheader("1. Upload Resume PDFs")
uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

st.subheader("2. Paste the Job Description")
jd_text = st.text_area("Job Description", height=200,
                        placeholder="Paste the full job description here...")

st.subheader("3. Filters")
col_slider, col_topn = st.columns(2)
with col_slider:
    min_score = st.slider("Minimum Match Score (%)", min_value=0, max_value=100, value=50, step=5)
with col_topn:
    top_n = st.number_input("Show Top N Candidates", min_value=1, max_value=100, value=10, step=1)

if st.button("🚀 Screen All Resumes", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one resume PDF.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    with st.spinner(f"Screening {len(uploaded_files)} resume(s)..."):
        results= requests.post(
            "http://localhost:8000/recruiter",
            files=[("pdfs", (f.name, f, "application/pdf")) for f in uploaded_files],
            data={"jd_text": jd_text.strip()},
        )
        all_results = results.json()

    scored  = [r for r in all_results if r["score"] is not None]
    errored = [r for r in all_results if r["score"] is None]

    filtered = [r for r in scored if r["score"] >= min_score]
    filtered = filtered[:int(top_n)]

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📁 Total Uploaded",      len(uploaded_files))
    c2.metric("✅ Qualified Candidates", len(filtered))
    c3.metric("🏆 Highest Score",
            f"{scored[0]['score']}%" if scored else "—")
    c4.metric("📊 Average Score",
            f"{round(sum(r['score'] for r in scored) / len(scored), 1)}%" if scored else "—")

    st.markdown("---")
    st.subheader(f"📋 Ranked Candidates  (min {min_score}% · top {top_n})")

    if not filtered:
        st.warning("No candidates meet the current filters. Try lowering the minimum score.")
    else:
        rows = []
        for r in filtered:
            rows.append({
                "Rank":            r.get("rank", "—"),
                "Candidate Name":  r["name"],
                "Match Score (%)": r["score"],
                "Prediction":      "✅ Good Fit" if r["prediction"] == "Good Fit" else "❌ No Fit",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=buf.getvalue().encode("utf-8"),
            file_name="recruiter_screening_results.csv",
            mime="text/csv",
        )

    if errored:
        st.markdown("---")
        st.subheader("⚠️ Skipped Files")
        for r in errored:
            st.warning(f"**{r['name']}** — {r['error']}")
