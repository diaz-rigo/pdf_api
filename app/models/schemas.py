from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PDFUploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    pages: int
    extracted_text_path: str
    message: str
    
class SearchRequest(BaseModel):
    term: str
    case_sensitive: bool = False
    use_regex: bool = False
    
class SearchResult(BaseModel):
    page: int
    position: int
    context: str
    snippet: str
    
class SearchResponse(BaseModel):
    term: str
    total_matches: int
    results: List[SearchResult]
    pdf_id: str
    execution_time: float
    
class PDFInfo(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    size: int
    pages: int
    has_text: bool
    text_file_size: Optional[int] = None