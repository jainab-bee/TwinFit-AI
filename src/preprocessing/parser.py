import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

    except Exception as e:
        raise ValueError(f"PDF parsing failed: {e}")

    return text.strip()