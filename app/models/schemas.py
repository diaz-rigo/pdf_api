from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# ===============================
# RESPUESTA AL SUBIR PDF
# ===============================
class PDFUploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    pages: int
    extracted_text_path: str
    message: str


# ===============================
# PETICIÓN DE BÚSQUEDA
# ===============================
class SearchRequest(BaseModel):
    term: str
    case_sensitive: bool = False
    use_regex: bool = False


# ===============================
# RESULTADO INDIVIDUAL
# ===============================
class SearchResult(BaseModel):
    page: int
    position: int
    context: str
    snippet: str


# ===============================
# RESPUESTA DE BÚSQUEDA
# ===============================
class SearchResponse(BaseModel):
    term: str
    total_matches: int
    results: List[SearchResult]
    pdf_id: str
    execution_time: float


# ===============================
# INFORMACIÓN GENERAL DEL PDF
# ===============================
class PDFInfo(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    size: int
    pages: int
    has_text: bool
    text_file_size: Optional[int] = None


# ===============================
# 🔥 ANALYSIS (EVITA TU ERROR)
# ===============================
class PDFAnalysis(BaseModel):
    pdf_id: str
    pages: int
    used_ocr: bool
    extracted_text_path: Optional[str] = None
