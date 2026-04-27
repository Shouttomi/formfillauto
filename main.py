import os
import asyncio
import tempfile
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from extractor import extract_challans_from_pdf

SELF_URL = os.getenv("RENDER_EXTERNAL_URL", "https://challan-extractor.onrender.com")


async def keep_alive():
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(300)  # 5 minutes
        try:
            await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(f"{SELF_URL}/health", timeout=10)
            )
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(keep_alive())
    yield


app = FastAPI(title="Challan PDF Extractor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EXAMPLE_RESPONSE = {
    "count": 2,
    "challans": [
        {
            "date": "14/04/2026",
            "challan_date": "14/04/2026",
            "challan_no": "1042",
            "firm": "JAI MATA DI FASHIONS PVT. LTD.",
            "lot_book_type": "",
            "party": "LUCKY FABRICS",
            "party_address": "123 TEXTILE MARKET, SURAT - 395003",
            "master_ac": "",
            "agent": "",
            "gstin_no": "24AABCL1234A1Z5",
            "gstin_map": {"LUCKY FABRICS": "24AABCL1234A1Z5", "JAI MATA DI FASHIONS PVT. LTD.": "27AABCL5678B1Z3"},
            "pan_no": "AABCL1234A",
            "group": "",
            "marka_help": "LF-VRUNDAVAN",
            "lot_no": "",
            "quality": "VRUNDAVAN PRINT",
            "hsn_code": "540710",
            "taka": 40,
            "meter": 1240.5,
            "fas_rate": 85.0,
            "amount": 105442.5,
            "dyed_print": "",
            "weight": 312.0,
            "total": 0.0,
            "lr_no": "LR/2026/0412",
            "lr_date": "",
            "chadhti": 0.0,
            "width": 0.0,
            "transpoter": "",
            "remark": "Party: KRISHNA TEXTILES, Bill No: B-291, Pur No: 5503",
            "weaver": "",
            "item": "",
            "pu_bill_no": "5503",
            "table": [
                {"srl_no": 1, "meter": 31.2},
                {"srl_no": 2, "meter": 30.8},
                {"srl_no": 3, "meter": 31.5},
                {"srl_no": 4, "meter": 30.0},
                {"srl_no": 5, "meter": 31.0}
            ]
        },
        {
            "date": "14/04/2026",
            "challan_date": "14/04/2026",
            "challan_no": "3318",
            "firm": "JAI MATA DI FASHIONS PVT. LTD.",
            "lot_book_type": "",
            "party": "SUDARSHAN GARMENTS",
            "party_address": "PLOT 7, RING ROAD, SURAT - 395002",
            "master_ac": "",
            "agent": "",
            "gstin_no": "24ABCDS9876Z1Z2",
            "gstin_map": {"SUDARSHAN GARMENTS": "24ABCDS9876Z1Z2"},
            "pan_no": "ABCDS9876Z",
            "group": "",
            "marka_help": "",
            "lot_no": "",
            "quality": "HEAVY GEORGETTE 60GMS",
            "hsn_code": "5407",
            "taka": 20,
            "meter": 620.0,
            "fas_rate": 110.0,
            "amount": 68200.0,
            "dyed_print": "",
            "weight": 186.0,
            "total": 0.0,
            "lr_no": "",
            "lr_date": "",
            "chadhti": 0.0,
            "width": 44.0,
            "transpoter": "",
            "remark": "",
            "weaver": "RAMESH WEAVING MILLS",
            "item": "HEAVY GEORGETTE 60GMS",
            "pu_bill_no": "PB-1109",
            "table": [
                {"srl_no": 1, "meter": 31.0},
                {"srl_no": 2, "meter": 31.5},
                {"srl_no": 3, "meter": 30.5},
                {"srl_no": 4, "meter": 31.0}
            ]
        }
    ]
}


@app.post(
    "/extract",
    responses={
        200: {
            "description": "Challans extracted successfully",
            "content": {
                "application/json": {
                    "example": EXAMPLE_RESPONSE
                }
            },
        },
        400: {
            "description": "Invalid file type",
            "content": {
                "application/json": {
                    "example": {"error": "Only PDF files accepted"}
                }
            },
        },
        500: {
            "description": "Extraction failed",
            "content": {
                "application/json": {
                    "example": {"error": "Could not extract text from PDF"}
                }
            },
        },
    },
)
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


@app.get("/example", summary="Get example API response")
def example_response():
    """Returns a sample response so frontend developers know what to expect."""
    return EXAMPLE_RESPONSE
