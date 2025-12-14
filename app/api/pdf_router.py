from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import os
import time
from typing import Optional
from app.services.pdf_service import PDFService
from app.models.schemas import (
    PDFUploadResponse, 
    SearchRequest, 
    SearchResponse,
    PDFInfo
)
from app.core.config import settings

router = APIRouter(prefix="/api/pdf", tags=["pdf"])
pdf_service = PDFService()

# Almacenamiento temporal en memoria (en producción usar base de datos)
pdf_storage = {}

@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...), use_ocr: bool = Query(True)):
    """Sube un PDF y extrae su texto"""
    try:
        # Validar que sea PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
        # Leer archivo
        file_bytes = await file.read()
        
        # Generar ID único
        pdf_id = pdf_service.generate_pdf_id(file.filename, file_bytes)
        
        # Guardar PDF
        pdf_path = os.path.join(settings.UPLOAD_FOLDER, f"{pdf_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
        
        # Extraer texto
        text, pages, used_ocr = pdf_service.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        
        # Guardar texto extraído
        text_path = pdf_service.save_extracted_text(text, pdf_id)
        
        # Almacenar en memoria
        pdf_storage[pdf_id] = {
            'filename': file.filename,
            'pdf_path': pdf_path,
            'text_path': text_path,
            'pages': pages,
            'upload_time': time.time(),
            'text': text,
            'size': len(file_bytes)
        }
        
        return PDFUploadResponse(
            id=pdf_id,
            filename=file.filename,
            size=len(file_bytes),
            pages=pages,
            extracted_text_path=text_path,
            message=f"PDF procesado exitosamente. {'Se usó OCR' if used_ocr else 'No se requirió OCR'}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {str(e)}")

@router.post("/{pdf_id}/search", response_model=SearchResponse)
async def search_pdf(pdf_id: str, search_request: SearchRequest):
    """Busca un término en el PDF"""
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    start_time = time.time()
    
    # Cargar texto
    pdf_data = pdf_storage[pdf_id]
    text = pdf_data['text']
    
    # Realizar búsqueda
    results = pdf_service.search_in_text(
        text, 
        search_request.term, 
        search_request.case_sensitive
    )
    
    # Limitar resultados para respuesta
    limited_results = results[:100]  # Máximo 100 resultados
    
    execution_time = time.time() - start_time
    
    return SearchResponse(
        term=search_request.term,
        total_matches=len(results),
        results=limited_results,
        pdf_id=pdf_id,
        execution_time=execution_time
    )

@router.get("/{pdf_id}/text")
async def get_pdf_text(pdf_id: str):
    """Obtiene el texto completo del PDF"""
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    text_path = pdf_storage[pdf_id]['text_path']
    
    if os.path.exists(text_path):
        return FileResponse(
            text_path, 
            media_type='text/plain',
            filename=f"{pdf_id}_texto.txt"
        )
    else:
        raise HTTPException(status_code=404, detail="Texto no encontrado")

@router.get("/{pdf_id}/info", response_model=PDFInfo)
async def get_pdf_info(pdf_id: str):
    """Obtiene información del PDF"""
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    pdf_data = pdf_storage[pdf_id]
    
    # Verificar tamaño del archivo de texto
    text_file_size = None
    if os.path.exists(pdf_data['text_path']):
        text_file_size = os.path.getsize(pdf_data['text_path'])
    
    return PDFInfo(
        id=pdf_id,
        filename=pdf_data['filename'],
        upload_date=datetime.fromtimestamp(pdf_data['upload_time']),
        size=pdf_data['size'],
        pages=pdf_data['pages'],
        has_text=True if pdf_data['text'].strip() else False,
        text_file_size=text_file_size
    )

@router.get("/list")
async def list_pdfs():
    """Lista todos los PDFs procesados"""
    return {
        "count": len(pdf_storage),
        "pdfs": [
            {
                "id": pdf_id,
                "filename": data['filename'],
                "pages": data['pages'],
                "size": data['size']
            }
            for pdf_id, data in pdf_storage.items()
        ]
    }

@router.delete("/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """Elimina un PDF y sus archivos asociados"""
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    pdf_data = pdf_storage[pdf_id]
    
    # Eliminar archivos
    try:
        if os.path.exists(pdf_data['pdf_path']):
            os.remove(pdf_data['pdf_path'])
        if os.path.exists(pdf_data['text_path']):
            os.remove(pdf_data['text_path'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando archivos: {str(e)}")
    
    # Eliminar de memoria
    del pdf_storage[pdf_id]
    
    return {"message": f"PDF {pdf_id} eliminado exitosamente"}

@router.post("/quick-search")
async def quick_search(
    file: UploadFile = File(...),
    search_term: str = Query(...),
    use_ocr: bool = Query(True)
):
    """Búsqueda rápida sin guardar el PDF permanentemente"""
    try:
        start_time = time.time()
        
        # Leer archivo
        file_bytes = await file.read()
        
        # Guardar temporalmente
        temp_id = f"temp_{hash(file_bytes) % 1000000}"
        temp_path = os.path.join(settings.UPLOAD_FOLDER, f"{temp_id}.pdf")
        
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        
        # Extraer texto
        text, pages, _ = pdf_service.extract_text_from_pdf(temp_path, use_ocr=use_ocr)
        
        # Buscar
        results = pdf_service.search_in_text(text, search_term)
        
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            term=search_term,
            total_matches=len(results),
            results=results[:50],  # Máximo 50 resultados para búsqueda rápida
            pdf_id="temp",
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda rápida: {str(e)}")