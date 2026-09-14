"""
backend/app.py — SignalScope API.

Run:
    uvicorn backend.app:app --reload --port 8000

Endpoints:
    POST /api/predict          core task (Section 3.1)      -> label + confidence
    POST /api/explain          Module A                      -> heat-map + grounded text
    POST /api/attribution      Module B                      -> generator family
    POST /api/provenance       Module D                      -> C2PA/EXIF combined verdict
    POST /api/multimodal       Module E                      -> image+caption consistency
    POST /api/full-scan        F: one call running everything enabled, for the UI
    GET  /api/health

CORS is open for local dev; tighten origins before any real deployment.
"""

import base64
import os
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

from predict import predict_image          # noqa: E402
from gradcam import explain as explain_fn  # noqa: E402
from attribution import heuristic_family   # noqa: E402
from provenance import combine_with_visual_verdict  # noqa: E402

CHECKPOINT = os.environ.get("SIGNALSCOPE_CHECKPOINT", "checkpoints/signalscope.pt")

app = FastAPI(title="SignalScope API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # dev only -- restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    return tmp.name


def _checkpoint_or_503():
    if not os.path.exists(CHECKPOINT):
        raise HTTPException(
            status_code=503,
            detail=(f"No trained checkpoint at {CHECKPOINT}. Run "
                    "`python model/train.py` first (see README)."),
        )


@app.get("/api/health")
def health():
    return {"status": "ok", "checkpoint_loaded": os.path.exists(CHECKPOINT)}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    _checkpoint_or_503()
    path = _save_upload(file)
    try:
        return predict_image(path, CHECKPOINT)
    finally:
        os.unlink(path)


@app.post("/api/explain")
def explain(file: UploadFile = File(...)):
    _checkpoint_or_503()
    path = _save_upload(file)
    heatmap_path = path + "_heatmap.jpg"
    try:
        pred = predict_image(path, CHECKPOINT)
        exp = explain_fn(path, CHECKPOINT, out_heatmap_path=heatmap_path)
        heatmap_b64 = None
        if os.path.exists(heatmap_path):
            with open(heatmap_path, "rb") as f:
                heatmap_b64 = base64.b64encode(f.read()).decode()
        return {
            "prediction": pred,
            "localisation": exp["localisation"],
            "explanation": exp["explanation"],
            "heatmap_base64": heatmap_b64,
        }
    finally:
        for p in (path, heatmap_path):
            if os.path.exists(p):
                os.unlink(p)


@app.post("/api/attribution")
def attribution(file: UploadFile = File(...)):
    path = _save_upload(file)
    try:
        return heuristic_family(path)
    finally:
        os.unlink(path)


@app.post("/api/provenance")
def provenance(file: UploadFile = File(...)):
    _checkpoint_or_503()
    path = _save_upload(file)
    try:
        pred = predict_image(path, CHECKPOINT)
        return combine_with_visual_verdict(pred["p_fake"], path)
    finally:
        os.unlink(path)


@app.post("/api/multimodal")
def multimodal(file: UploadFile = File(...), caption: str = Form(...)):
    from multimodal import image_text_consistency
    path = _save_upload(file)
    try:
        return image_text_consistency(path, caption)
    finally:
        os.unlink(path)


@app.post("/api/full-scan")
def full_scan(file: UploadFile = File(...), caption: str = Form(None)):
    """Convenience endpoint for the frontend: runs core + every bonus module
    that doesn't error out, and returns everything in one response."""
    _checkpoint_or_503()
    path = _save_upload(file)
    heatmap_path = path + "_heatmap.jpg"
    result = {}
    try:
        result["prediction"] = predict_image(path, CHECKPOINT)

        try:
            exp = explain_fn(path, CHECKPOINT, out_heatmap_path=heatmap_path)
            heatmap_b64 = None
            if os.path.exists(heatmap_path):
                with open(heatmap_path, "rb") as f:
                    heatmap_b64 = base64.b64encode(f.read()).decode()
            result["explanation"] = {
                "localisation": exp["localisation"],
                "explanation": exp["explanation"],
                "heatmap_base64": heatmap_b64,
            }
        except Exception as e:
            result["explanation"] = {"error": str(e)}

        try:
            result["attribution"] = heuristic_family(path)
        except Exception as e:
            result["attribution"] = {"error": str(e)}

        try:
            result["provenance"] = combine_with_visual_verdict(
                result["prediction"]["p_fake"], path)
        except Exception as e:
            result["provenance"] = {"error": str(e)}

        if caption:
            try:
                from multimodal import image_text_consistency
                result["multimodal"] = image_text_consistency(path, caption)
            except Exception as e:
                result["multimodal"] = {"error": str(e)}

        return result
    finally:
        for p in (path, heatmap_path):
            if os.path.exists(p):
                os.unlink(p)
