import os
import sys
import tempfile
from typing import Annotated

from fastapi import ( FastAPI,UploadFile,File,Form,HTTPException,status,)

sys.path.append( os.path.abspath(os.path.dirname(__file__)) )

from src.models.predict import predict_from_pdf
from src.recruiter.bulk_screening import screen_all

app = FastAPI(title="TwinFit AI API",version="1.0.0")

def validate_pdf(pdf: UploadFile) -> None:
    if not pdf.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{pdf.filename} is not a PDF file.",
        )

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "api": "TwinFit AI API",
    }

@app.post("/predict")
async def predict(pdf: Annotated[UploadFile,File(description="Upload candidate resume PDF"),],jd_text: Annotated[str,Form(description="Enter job description"),],):
    if not jd_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a job description.",
        )
    validate_pdf(pdf)
    tmp_path = None

    try:
        pdf_content = await pdf.read()
        if not pdf_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF is empty.",
            )

        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf",) as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name

        result = predict_from_pdf(tmp_path,jd_text.strip())
        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error

    finally:
        await pdf.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/recruiter")
async def recruiter_predict(pdfs: Annotated[list[UploadFile],File(description="Upload multiple resume PDFs"),],jd_text: Annotated[str,Form(description="Enter job description"), ],):
    if not pdfs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload at least one resume PDF.",
        )

    if not jd_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a job description.",
        )

    for pdf in pdfs:
        validate_pdf(pdf)

    try:
        result = await screen_all( pdfs, jd_text.strip())

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error

    finally:
        for pdf in pdfs:
            await pdf.close()