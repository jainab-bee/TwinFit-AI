import inspect
import os
import re
import sys
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import predict_from_pdf

MAX_MB = 5


def extract_name_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[_\-\.]+", " ", stem)
    noise = {"resume", "cv", "curriculum", "vitae", "final", "new", "updated", "v1", "v2"}
    words = [w for w in stem.split() if w.lower() not in noise]
    return " ".join(words).title() if words else stem.title()


def _get_file_name(uploaded_file) -> str:
    return getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", None) or "uploaded_resume.pdf"


def _get_file_size(uploaded_file) -> int:
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        return size

    file_obj = getattr(uploaded_file, "file", None)
    if file_obj is not None and hasattr(file_obj, "tell") and hasattr(file_obj, "seek"):
        try:
            current_pos = file_obj.tell()
            file_obj.seek(0, os.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(current_pos)
            return size
        except Exception:
            return 0

    return 0


async def screen_single(uploaded_file, jd_text: str) -> dict:
    name = extract_name_from_filename(_get_file_name(uploaded_file))
    size = _get_file_size(uploaded_file)
    if size > MAX_MB * 1024 * 1024:
        return {"name": name, "score": None, "prediction": None, "error": f"File too large (>{MAX_MB} MB)"}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = uploaded_file.read()
            if inspect.isawaitable(content):
                content = await content
            tmp.write(content)
            tmp_path = tmp.name
        result = predict_from_pdf(tmp_path, jd_text)
        return {"name": name, "score": result["match_score"], "prediction": result["prediction"], "error": None}
    except Exception as error:
        return {"name": name, "score": None, "prediction": None, "error": str(error)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def screen_all(uploaded_files: list, jd_text: str) -> list:
    results = [await screen_single(file_obj, jd_text) for file_obj in uploaded_files]
    scored = sorted([result for result in results if result["score"] is not None], key=lambda item: item["score"], reverse=True)
    errored = [result for result in results if result["score"] is None]

    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    return scored + errored
