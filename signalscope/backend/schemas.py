from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PredictResponse(BaseModel):
    label: str
    confidence: float
    p_fake: float
    p_real: float
    presentation: str


class ExplainResponse(BaseModel):
    prediction: PredictResponse
    localisation: Dict[str, Any]
    explanation: str
    heatmap_base64: Optional[str] = None


class AttributionResponse(BaseModel):
    family: str
    confidence: float
    method: str


class ProvenanceResponse(BaseModel):
    visual_p_fake: float
    combined_p_fake: float
    adjustment_applied: float
    exif: Dict[str, Any]
    c2pa: Dict[str, Any]
    reasoning: List[str]


class MultimodalResponse(BaseModel):
    similarity: float
    verdict: str
    note: str


class FullReportResponse(BaseModel):
    prediction: PredictResponse
    explanation: Optional[ExplainResponse] = None
    attribution: Optional[AttributionResponse] = None
    provenance: Optional[ProvenanceResponse] = None
    multimodal: Optional[MultimodalResponse] = None
