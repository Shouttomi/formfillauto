import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from extractor import extract_challans_from_pdf

app = FastAPI(title="Challan PDF Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(status_code=400, content={"error": "Only PDF files accepted"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        challans = extract_challans_from_pdf(tmp_path)
        return {"challans": challans, "count": len(challans)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/")
def root():
    return {"message": "Challan PDF Extractor API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
