import os
import sys
sys.path.append(os.getcwd())

from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

from src.preprocessing.clean_txt import clean_text
from src.preprocessing.parser import extract_text_from_pdf
from src.models.custom_layers import AbsoluteDifference

MODEL_PATH = "saved_models/best_keras_siamese_binary_model.keras"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_models():
    keras_model = load_model(MODEL_PATH, custom_objects={"AbsoluteDifference": AbsoluteDifference})
    sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return keras_model, sentence_model


def predict_resume_match(resume_text, jd_text, keras_model, sentence_model):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    resume_emb = sentence_model.encode( [resume_clean],convert_to_numpy=True,normalize_embeddings=True)
    jd_emb = sentence_model.encode( [jd_clean],convert_to_numpy=True,normalize_embeddings=True)
    probability = keras_model.predict([resume_emb, jd_emb])[0][0]
    return {
        "match_score": round(float(probability) * 100, 2),
        "prediction": "Good Fit" if probability >= 0.5 else "No Fit"
    }


def predict_from_pdf(pdf_path, jd_text):
    keras_model, sentence_model = load_models()
    resume_text = extract_text_from_pdf(pdf_path)
    return predict_resume_match(resume_text, jd_text, keras_model, sentence_model)
