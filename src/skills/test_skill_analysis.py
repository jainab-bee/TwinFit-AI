import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.skills.sementic_meaning import find_missing_skills, extract_skills

# ============================================================
# HELPERS
# ============================================================

_passed = 0
_failed = 0


def _check(condition: bool, label: str, reason: str = "") -> None:
    global _passed, _failed
    if condition:
        print("  [PASS] " + label)
        _passed += 1
    else:
        msg = "  [FAIL] " + label
        if reason:
            msg += "  -- " + reason
        print(msg)
        _failed += 1


def _run_test(name: str) -> None:
    print("\n" + "=" * 60)
    print("TEST: " + name)
    print("=" * 60)


def _skill_names(skill_list) -> set:
    if not skill_list:
        return set()
    if isinstance(skill_list[0], dict):
        return {s["skill"] for s in skill_list}
    return set(skill_list)


def _matched_skill_names(result) -> set:
    return {m.get("matched_resume_skill") or m.get("jd_skill") for m in result["matched_skills"]}


def _jd_skill_names(result) -> set:
    return _skill_names(result["jd_required_skills"])


def _resume_skill_names(result) -> set:
    return _skill_names(result["resume_skills"])


# ============================================================
# TEST 1 -- Exact Match
# ============================================================
_run_test("1: Exact Match -- Python, Docker, scikit-learn")

resume1 = "Python, Docker, scikit-learn"
jd1 = """
Must have Python and Docker.
Experience with scikit-learn is required.
"""

result1 = find_missing_skills(resume1, jd1)
rs1 = _resume_skill_names(result1)
jd1_names = _jd_skill_names(result1)

_check("Python" in rs1, "Python found in resume skills", "Found: " + str(rs1))
_check("Docker" in rs1, "Docker found in resume skills", "Found: " + str(rs1))
_check("scikit-learn" in rs1, "scikit-learn found in resume skills", "Found: " + str(rs1))
_check("Python" in jd1_names, "Python in JD skills")
_check("Docker" in jd1_names, "Docker in JD skills")
_check("scikit-learn" in jd1_names, "scikit-learn in JD skills")

analysis1 = {e["jd_skill"]: e["status"] for e in result1["skill_analysis"]}
_check(
    analysis1.get("Python") in ("exact_match", "alias_match"),
    "Python matched",
    "status=" + str(analysis1.get("Python"))
)
_check(
    analysis1.get("Docker") in ("exact_match", "alias_match"),
    "Docker matched",
    "status=" + str(analysis1.get("Docker"))
)
_check(
    analysis1.get("scikit-learn") in ("exact_match", "alias_match"),
    "scikit-learn matched",
    "status=" + str(analysis1.get("scikit-learn"))
)
_check(
    not result1["missing_mandatory_skills"],
    "No mandatory missing skills",
    "Missing: " + str(result1["missing_mandatory_skills"])
)


# ============================================================
# TEST 2 -- Related But Missing (TensorFlow)
# ============================================================
_run_test("2: Related But Missing -- TensorFlow not satisfied by ML + NNs")

resume2 = "Machine Learning, Neural Networks"
jd2 = "TensorFlow is required."

result2 = find_missing_skills(resume2, jd2)
analysis2 = {e["jd_skill"]: e for e in result2["skill_analysis"]}

_check("Machine Learning" in _resume_skill_names(result2), "Machine Learning found in resume")
_check("Neural Networks" in _resume_skill_names(result2), "Neural Networks found in resume")
_check("TensorFlow" in _jd_skill_names(result2), "TensorFlow in JD skills")

tf_entry = analysis2.get("TensorFlow", {})
_check(
    tf_entry.get("status") in ("missing", "related_match"),
    "TensorFlow NOT counted as matched",
    "status=" + str(tf_entry.get("status"))
)
_check(
    tf_entry.get("matched_resume_skill") is None,
    "matched_resume_skill is None for TensorFlow",
    "got: " + str(tf_entry.get("matched_resume_skill"))
)
related_tf = set(tf_entry.get("related_resume_skills", []))
_check(
    bool(related_tf & {"Machine Learning", "Neural Networks"}),
    "Machine Learning / Neural Networks shown as related skills for TensorFlow",
    "related: " + str(related_tf)
)
_check(
    "TensorFlow" in result2["missing_mandatory_skills"],
    "TensorFlow is in missing_mandatory_skills",
    "missing: " + str(result2["missing_mandatory_skills"])
)


# ============================================================
# TEST 3 -- Alias Matching
# ============================================================
_run_test("3: Alias Matching -- sklearn, postgres, nodejs")

resume3 = "sklearn, postgres, nodejs"
jd3 = "scikit-learn, PostgreSQL, Node.js"

result3 = find_missing_skills(resume3, jd3)
analysis3 = {e["jd_skill"]: e for e in result3["skill_analysis"]}

_check("scikit-learn" in _jd_skill_names(result3), "scikit-learn in JD skills")
_check("PostgreSQL" in _jd_skill_names(result3), "PostgreSQL in JD skills")
_check("Node.js" in _jd_skill_names(result3), "Node.js in JD skills")

_check(
    analysis3.get("scikit-learn", {}).get("status") in ("exact_match", "alias_match"),
    "scikit-learn matched via alias (sklearn)",
    "status=" + str(analysis3.get("scikit-learn", {}).get("status"))
)
_check(
    analysis3.get("PostgreSQL", {}).get("status") in ("exact_match", "alias_match"),
    "PostgreSQL matched via alias (postgres)",
    "status=" + str(analysis3.get("PostgreSQL", {}).get("status"))
)
_check(
    analysis3.get("Node.js", {}).get("status") in ("exact_match", "alias_match"),
    "Node.js matched via alias (nodejs)",
    "status=" + str(analysis3.get("Node.js", {}).get("status"))
)
_check(
    not result3["missing_mandatory_skills"] and not result3["missing_preferred_skills"],
    "No missing skills for alias test",
    "missing: " + str(result3["missing_skills"])
)


# ============================================================
# TEST 4 -- Noise Rejection
# ============================================================
_run_test("4: Noise Rejection -- job posting prose should not produce noise skills")

noisy_text = """
We are looking for a Machine Learning Engineer.
The candidate will train models and work with the engineering team.
"""

noisy_skills = extract_skills(noisy_text)
noisy_names_lower = {s["skill"].lower() for s in noisy_skills}

FORBIDDEN_NOISE = {
    "machine learning engineer", "candidate", "train", "model",
    "engineer", "engineering", "training", "learning",
}

for noise in FORBIDDEN_NOISE:
    _check(
        noise not in noisy_names_lower,
        "'" + noise + "' NOT extracted as a skill",
        "Extracted skills: " + str(noisy_names_lower)
    )

_check(
    "machine learning engineer" not in noisy_names_lower,
    "'machine learning engineer' not extracted as a skill"
)


# ============================================================
# TEST 5 -- Heading Rejection
# ============================================================
_run_test("5: Heading Rejection -- TECHNICAL SKILLS / Languages / Tools headings")

heading_text = """
TECHNICAL SKILLS
Languages
Tools
"""

heading_skills = extract_skills(heading_text)
heading_names_lower = {s["skill"].lower() for s in heading_skills}

FORBIDDEN_HEADINGS = {"technical skills", "languages", "tools"}
for h in FORBIDDEN_HEADINGS:
    _check(
        h not in heading_names_lower,
        "'" + h + "' NOT extracted as a skill from headings",
        "Extracted: " + str(heading_names_lower)
    )


# ============================================================
# TEST 6 -- Framework Distinction (PyTorch != TensorFlow)
# ============================================================
_run_test("6: Framework Distinction -- PyTorch does not satisfy TensorFlow requirement")

resume6 = "PyTorch"
jd6 = "TensorFlow is mandatory."

result6 = find_missing_skills(resume6, jd6)
analysis6 = {e["jd_skill"]: e for e in result6["skill_analysis"]}

_check("PyTorch" in _resume_skill_names(result6), "PyTorch found in resume")
_check("TensorFlow" in _jd_skill_names(result6), "TensorFlow found in JD")

tf_entry6 = analysis6.get("TensorFlow", {})
_check(
    tf_entry6.get("status") in ("missing", "related_match"),
    "TensorFlow is NOT marked as matched (PyTorch != TensorFlow)",
    "status=" + str(tf_entry6.get("status"))
)
_check(
    tf_entry6.get("matched_resume_skill") is None,
    "matched_resume_skill is None (no exact/alias match)",
    "got: " + str(tf_entry6.get("matched_resume_skill"))
)
related6 = set(tf_entry6.get("related_resume_skills", []))
_check(
    "PyTorch" in related6,
    "PyTorch listed as related skill for TensorFlow",
    "related: " + str(related6)
)
_check(
    "TensorFlow" in result6["missing_mandatory_skills"],
    "TensorFlow in missing_mandatory_skills",
    "missing: " + str(result6["missing_mandatory_skills"])
)


# ============================================================
# SUMMARY
# ============================================================
total = _passed + _failed
print("\n" + "=" * 60)
print("RESULTS: " + str(_passed) + "/" + str(total) + " passed, " + str(_failed) + " failed")
print("=" * 60)

if _failed == 0:
    print("ALL TESTS PASSED")
else:
    print("ATTENTION: " + str(_failed) + " test(s) failed -- see [FAIL] lines above.")
    sys.exit(1)
