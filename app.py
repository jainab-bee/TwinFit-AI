import os
import sys
import csv
import io
import time
import tempfile
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.models.predict import load_models, predict_resume_match
from src.preprocessing.parser import extract_text_from_pdf
from src.skills import find_missing_skills

st.set_page_config(
    page_title="TwinFit AI ",
    page_icon="📄",
    layout="wide"
)

MAX_PDF_SIZE_MB = 5
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024

AI_HIGH_THRESHOLD = 60
COVERAGE_HIGH_THRESHOLD = 80
COVERAGE_MED_THRESHOLD = 50


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data(show_spinner=False)
def run_analysis(resume_text, jd_text):
    score_result = predict_resume_match(resume_text, jd_text, *get_models())
    skill_result = find_missing_skills(resume_text, jd_text)
    return score_result, skill_result


with st.sidebar:
    st.title("ℹ️ How This Works")
    st.markdown("""
**Step 1 — Provide your resume**
Upload a PDF or paste plain text directly.

**Step 2 — Paste the job description**
Copy the full job posting text into the box.

**Step 3 — Click Predict Match**
The app runs two independent checks:
- **AI Similarity Score** — how closely your resume text mirrors the JD's language (neural network)
- **Skill Coverage** — how many of the JD's listed skills appear in your resume (rule-based)

The final verdict combines both. They can sometimes disagree — the reason line will explain why.

---

**Verdict guide**
| AI Score | Coverage | Verdict |
|----------|----------|---------|
| High (≥60%) | High (≥80%) | ✅ Good Fit |
| High | Medium (50–79%) | ✅ Good Fit (minor gaps) |
| High | Low (<50%) | ⚠️ Partial Fit |
| Low (<60%) | High (≥80%) | ℹ️ Skills Match — Check Experience |
| Low | Medium | ⚠️ Partial Fit — Mixed Signals |
| Low | Low | ⚠️ No Fit |

---

**Tips for best results**
- Use a text-selectable PDF (not a scanned image)
- Paste the full job description, not just a summary
- Mirror the JD's exact skill names in your resume
""")
    st.divider()
    st.caption("Model: Siamese neural network + all-MiniLM-L6-v2")


st.title("📄 TwinFit AI ")
st.write("Find out how well your resume matches a job description — and exactly what skills are missing.")


def validate_pdf(uploaded_file):
    if uploaded_file.size > MAX_PDF_SIZE_BYTES:
        return False, (
            f"This file is {uploaded_file.size / (1024*1024):.1f} MB, "
            f"which is over the {MAX_PDF_SIZE_MB} MB limit. "
            "Please upload a smaller PDF."
        )
    header = uploaded_file.read(4)
    uploaded_file.seek(0)
    if header != b"%PDF":
        return False, (
            "This doesn't look like a valid PDF file. "
            "Please upload a proper PDF and try again."
        )
    return True, None


def compute_skill_coverage(matched_count, total_jd_skills):
    if total_jd_skills == 0:
        return 0
    return round((matched_count / total_jd_skills) * 100, 1)


def get_verdict(ai_score, coverage_pct):
    high_ai = ai_score >= AI_HIGH_THRESHOLD
    high_cov = coverage_pct >= COVERAGE_HIGH_THRESHOLD
    med_cov = COVERAGE_MED_THRESHOLD <= coverage_pct < COVERAGE_HIGH_THRESHOLD

    if high_ai and high_cov:
        return "✅ Good Fit", "Strong skill match and high AI confidence.", "success"

    if high_ai and med_cov:
        return "✅ Good Fit (minor gaps)", "AI is confident and most skills are present — a few gaps to address.", "success"

    if high_ai and not high_cov and not med_cov:
        return "⚠️ Partial Fit", "AI is confident, but significant required skills are missing.", "warning"

    if not high_ai and high_cov:
        return "ℹ️ Skills Match — Check Experience", "Skills align well, but AI score is low — possibly due to experience level or resume formatting.", "info"

    if not high_ai and med_cov:
        return "⚠️ Partial Fit — Mixed Signals", "Some skills are present, but overall match is uncertain.", "warning"

    return "⚠️ No Fit", "Low skill overlap and low AI confidence.", "warning"


def render_skill_chips(skills, color):
    if not skills:
        st.write("None found.")
        return
    badge_style = (
        f"display:inline-block; background:{color}22; color:{color}; "
        "border:1px solid; border-radius:20px; "
        "padding:3px 12px; margin:3px; font-size:0.85rem; font-weight:500;"
    )
    chips_html = " ".join(
        f'<span style="{badge_style}">{skill}</span>'
        for skill in skills
    )
    st.markdown(chips_html, unsafe_allow_html=True)


def build_csv_report(score_result, skill_result):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["=== SUMMARY ==="])
    writer.writerow(["Match Score (%)", score_result["match_score"]])
    writer.writerow(["Prediction", score_result["prediction"]])
    writer.writerow([])

    writer.writerow(["=== SKILL ANALYSIS ==="])
    writer.writerow(["Skill", "Status", "Requirement Type"])
    for entry in skill_result.get("skill_analysis", []):
        writer.writerow([
            entry.get("jd_skill", ""),
            entry.get("status", ""),
            entry.get("requirement_type", "")
        ])

    writer.writerow([])
    writer.writerow(["=== RESUME SKILLS DETECTED ==="])
    writer.writerow(["Skill", "Source Section"])
    for item in skill_result.get("resume_skills", []):
        writer.writerow([item.get("skill", ""), item.get("source_section", "")])

    return output.getvalue().encode("utf-8")


def show_results(score_result, skill_result):
    ai_score = score_result["match_score"]

    matched_skills = [item["jd_skill"] for item in skill_result.get("matched_skills", [])]
    missing_mandatory = skill_result.get("missing_mandatory_skills", [])
    missing_preferred = skill_result.get("missing_preferred_skills", [])
    all_missing = list(dict.fromkeys(missing_mandatory + missing_preferred))

    total_jd_skills = len(matched_skills) + len(all_missing)
    skill_coverage = compute_skill_coverage(len(matched_skills), total_jd_skills)
    verdict_label, reason, verdict_style = get_verdict(ai_score, skill_coverage)

    st.divider()
    col_ai, col_coverage, col_verdict = st.columns(3)

    with col_ai:
        st.metric(
            label="AI Similarity Score",
            value=f"{ai_score}%",
            help=f"Threshold: >= {AI_HIGH_THRESHOLD}% is High. Measures semantic similarity between resume and JD."
        )
    with col_coverage:
        st.metric(
            label="Skill Coverage",
            value=f"{skill_coverage}%",
            help=(
                f"High: >= {COVERAGE_HIGH_THRESHOLD}% | "
                f"Medium: {COVERAGE_MED_THRESHOLD}–{COVERAGE_HIGH_THRESHOLD - 1}% | "
                f"Low: < {COVERAGE_MED_THRESHOLD}%."
            )
        )
    with col_verdict:
        if verdict_style == "success":
            st.success(verdict_label)
        elif verdict_style == "info":
            st.info(verdict_label)
        else:
            st.warning(verdict_label)

    st.caption(f"**Why this verdict?** {reason}")

    col_matched, col_missing = st.columns(2)
    with col_matched:
        st.subheader(f"✅ Matched Skills ({len(matched_skills)})")
        render_skill_chips(matched_skills, "#27ae60")
    with col_missing:
        st.subheader(f"❌ Missing Skills ({len(all_missing)})")
        render_skill_chips(all_missing, "#e74c3c")

    if matched_skills or all_missing:
        st.divider()
        st.subheader("📊 Skill Overlap Breakdown")
        df = pd.DataFrame(
            {"Skills": [len(matched_skills), len(missing_mandatory), len(missing_preferred)]},
            index=["Matched", "Missing (Mandatory)", "Missing (Preferred)"]
        )
        st.bar_chart(df)

    st.divider()
    st.download_button(
        label="⬇️ Download Full Report (CSV)",
        data=build_csv_report(score_result, skill_result),
        file_name="resume_match_report.csv",
        mime="text/csv",
        help="Download a CSV with the full skill-by-skill breakdown."
    )


st.subheader("1. Provide Your Resume")
input_mode = st.radio(
    "How would you like to provide your resume?",
    options=["Upload a PDF", "Paste text directly"],
    horizontal=True
)

uploaded_file = None
pasted_text = ""

if input_mode == "Upload a PDF":
    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],
        help=f"Max file size: {MAX_PDF_SIZE_MB} MB. Must be a text-based (non-scanned) PDF."
    )
else:
    pasted_text = st.text_area(
        "Paste your resume text here",
        height=250,
        placeholder="Copy and paste the text from your resume..."
    )

st.subheader("2. Paste the Job Description")
jd_text = st.text_area(
    "Job Description",
    height=220,
    placeholder="Paste the full job description here..."
)

if st.button("🔍 Predict Match", type="primary"):

    if input_mode == "Upload a PDF" and uploaded_file is None:
        st.error("Please upload a resume PDF before running the analysis.")
        st.stop()

    if input_mode == "Paste text directly" and not pasted_text.strip():
        st.error("Please paste your resume text before running the analysis.")
        st.stop()

    if not jd_text.strip():
        st.error("Please paste a job description before running the analysis.")
        st.stop()

    progress_area = st.empty()

    def update_progress(step1_done, step2_done, step3_done):
        mark = lambda done: "✅" if done else "⏳"
        progress_area.markdown(
            f"**Progress**  \n"
            f"{mark(step1_done)} Step 1: Extracting resume text  \n"
            f"{mark(step2_done)} Step 2: Matching skills against job description  \n"
            f"{mark(step3_done)} Step 3: Scoring overall fit"
        )

    update_progress(False, False, False)

    resume_text = ""

    if input_mode == "Upload a PDF":
        is_valid, error_msg = validate_pdf(uploaded_file)
        if not is_valid:
            progress_area.empty()
            st.error(error_msg)
            st.stop()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            resume_text = extract_text_from_pdf(tmp_path)
        except Exception:
            progress_area.empty()
            st.error(
                "We couldn't read any text from this PDF. "
                "It might be a scanned image, password-protected, or corrupted. "
                "Try pasting the text directly using the other input mode."
            )
            st.stop()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        resume_text = pasted_text.strip()

    if not resume_text:
        progress_area.empty()
        st.error(
            "No readable text was found in the resume. "
            "If you uploaded a scanned PDF, try the 'Paste text directly' option instead."
        )
        st.stop()

    update_progress(True, False, False)

    try:
        score_result, skill_result = run_analysis(resume_text, jd_text)
    except Exception:
        progress_area.empty()
        st.error(
            "Something went wrong while analyzing the resume. "
            "Please try again. If the problem continues, check that the model files are in the `saved_models/` folder."
        )
        st.stop()

    update_progress(True, True, False)
    update_progress(True, True, True)
    time.sleep(0.4)
    progress_area.empty()

    show_results(score_result, skill_result)