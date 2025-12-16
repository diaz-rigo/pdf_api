# from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
# from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
# import os
# import time
# import json
# from typing import Optional, List, Dict
# from datetime import datetime
# import asyncio
# from app.services.pdf_service import PDFService
# from app.models.schemas import (
#     PDFUploadResponse, 
#     SearchRequest, 
#     SearchResponse,
#     PDFInfo,
#     PDFAnalysis,
#     ProcessingStatus
# )
# from app.core.config import settings
# import tempfile
# import hashlib

# router = APIRouter(prefix="/api/pdf", tags=["pdf"])
# pdf_service = PDFService()

# # Almacenamiento mejorado con información de estado y caché
# pdf_storage = {}
# processing_queue = {}
# text_cache = {}  # Cache de texto para búsquedas frecuentes

# @router.post("/upload", response_model=PDFUploadResponse)
# async def upload_pdf(
#     file: UploadFile = File(...), 
#     use_ocr: bool = Query(True),
#     language: str = Query("spa"),
#     batch_size: Optional[int] = Query(None, ge=1, le=50)
# ):
#     """Sube un PDF y extrae su texto con optimización para grandes archivos"""
#     try:
#         start_time = time.time()
        
#         # Validar que sea PDF
#         if not file.filename.lower().endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
#         # Leer archivo
#         file_bytes = await file.read()
        
#         # Validar tamaño
#         if len(file_bytes) > settings.MAX_FILE_SIZE:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Archivo demasiado grande. Máximo permitido: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
#             )
        
#         # Generar ID único
#         pdf_id = pdf_service.generate_pdf_id(file.filename, file_bytes)
        
#         # Analizar PDF para optimización
#         analysis = pdf_service.analyze_pdf_structure_from_bytes(file_bytes)
        
#         # Guardar PDF
#         pdf_path = os.path.join(settings.UPLOAD_FOLDER, f"{pdf_id}.pdf")
#         with open(pdf_path, "wb") as f:
#             f.write(file_bytes)
        
#         # Extraer texto con parámetros optimizados
#         actual_batch_size = batch_size or analysis.get('suggested_batch_size', 10)
        
#         text, pages, used_ocr = pdf_service.extract_text_from_pdf(
#             pdf_path, 
#             use_ocr=use_ocr, 
#             language=language,
#             batch_size=actual_batch_size
#         )
        
#         # Guardar texto extraído
#         text_path = pdf_service.save_extracted_text(text, pdf_id)
        
#         # Calcular estadísticas
#         processing_time = time.time() - start_time
#         text_size = len(text.encode('utf-8'))
        
#         # Almacenar en memoria con metadatos optimizados
#         pdf_storage[pdf_id] = {
#             'filename': file.filename,
#             'pdf_path': pdf_path,
#             'text_path': text_path,
#             'pages': pages,
#             'upload_time': time.time(),
#             'size': len(file_bytes),
#             'text_size': text_size,
#             'used_ocr': used_ocr,
#             'processing_time': processing_time,
#             'language': language,
#             'batch_size': actual_batch_size,
#             'analysis': analysis
#         }
        
#         # Cachear texto para búsquedas rápidas (solo si no es demasiado grande)
#         if text_size < 10 * 1024 * 1024:  # 10MB máximo para cache
#             text_cache[pdf_id] = text
        
#         return PDFUploadResponse(
#             id=pdf_id,
#             filename=file.filename,
#             size=len(file_bytes),
#             pages=pages,
#             extracted_text_path=text_path,
#             processing_time_seconds=round(processing_time, 2),
#             used_ocr=used_ocr,
#             analysis=analysis,
#             message=f"PDF procesado exitosamente en {processing_time:.1f}s. "
#                    f"{'Se usó OCR' if used_ocr else 'Texto extraído directamente'}."
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error procesando PDF: {str(e)}")

# @router.post("/upload-large", response_model=ProcessingStatus)
# async def upload_large_pdf(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     use_ocr: bool = Query(True),
#     language: str = Query("spa")
# ):
#     """Sube un PDF grande (>100 páginas) para procesamiento en background"""
#     try:
#         # Validar
#         if not file.filename.lower().endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
#         file_bytes = await file.read()
        
#         # Generar ID
#         pdf_id = pdf_service.generate_pdf_id(file.filename, file_bytes)
        
#         # Guardar temporalmente
#         temp_path = os.path.join(tempfile.gettempdir(), f"{pdf_id}.pdf")
#         with open(temp_path, "wb") as f:
#             f.write(file_bytes)
        
#         # Inicializar estado de procesamiento
#         processing_queue[pdf_id] = {
#             'status': 'queued',
#             'progress': 0,
#             'total_pages': 0,
#             'processed_pages': 0,
#             'start_time': time.time(),
#             'filename': file.filename
#         }
        
#         # Agregar tarea en background
#         background_tasks.add_task(
#             process_large_pdf_background,
#             temp_path,
#             pdf_id,
#             use_ocr,
#             language
#         )
        
#         return ProcessingStatus(
#             pdf_id=pdf_id,
#             status="queued",
#             message="PDF en cola de procesamiento",
#             check_status_url=f"/api/pdf/status/{pdf_id}",
#             estimated_wait_time="Variable según tamaño"
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error iniciando procesamiento: {str(e)}")

# async def process_large_pdf_background(pdf_path: str, pdf_id: str, use_ocr: bool, language: str):
#     """Procesa PDF grande en background"""
#     try:
#         processing_queue[pdf_id]['status'] = 'processing'
        
#         # Procesar PDF
#         text, total_pages, used_ocr = pdf_service.extract_text_from_pdf(
#             pdf_path, 
#             use_ocr=use_ocr, 
#             language=language,
#             batch_size=20
#         )
        
#         # Actualizar progreso
#         processing_queue[pdf_id]['progress'] = 100
#         processing_queue[pdf_id]['total_pages'] = total_pages
#         processing_queue[pdf_id]['processed_pages'] = total_pages
        
#         # Guardar permanentemente
#         permanent_path = os.path.join(settings.UPLOAD_FOLDER, f"{pdf_id}.pdf")
#         os.rename(pdf_path, permanent_path)
        
#         # Guardar texto
#         text_path = pdf_service.save_extracted_text(text, pdf_id)
        
#         # Almacenar en memoria
#         pdf_storage[pdf_id] = {
#             'filename': processing_queue[pdf_id]['filename'],
#             'pdf_path': permanent_path,
#             'text_path': text_path,
#             'pages': total_pages,
#             'upload_time': time.time(),
#             'size': os.path.getsize(permanent_path),
#             'text_size': len(text.encode('utf-8')),
#             'used_ocr': used_ocr,
#             'processing_time': time.time() - processing_queue[pdf_id]['start_time']
#         }
        
#         # Actualizar estado
#         processing_queue[pdf_id]['status'] = 'completed'
#         processing_queue[pdf_id]['result'] = {
#             'pdf_id': pdf_id,
#             'text_path': text_path,
#             'total_pages': total_pages
#         }
        
#         # Cachear texto
#         text_cache[pdf_id] = text
        
#     except Exception as e:
#         processing_queue[pdf_id]['status'] = 'failed'
#         processing_queue[pdf_id]['error'] = str(e)
        
#     finally:
#         # Limpiar después de 1 hora
#         await asyncio.sleep(3600)
#         if pdf_id in processing_queue:
#             del processing_queue[pdf_id]

# @router.get("/status/{pdf_id}")
# async def get_processing_status(pdf_id: str):
#     """Consulta el estado de procesamiento de un PDF grande"""
#     if pdf_id not in processing_queue:
#         if pdf_id in pdf_storage:
#             return {
#                 "pdf_id": pdf_id,
#                 "status": "already_processed",
#                 "stored_in": "pdf_storage"
#             }
#         raise HTTPException(status_code=404, detail="PDF no encontrado en cola de procesamiento")
    
#     status_info = processing_queue[pdf_id]
    
#     # Calcular tiempo transcurrido
#     elapsed_time = time.time() - status_info['start_time']
    
#     return ProcessingStatus(
#         pdf_id=pdf_id,
#         status=status_info['status'],
#         progress=status_info.get('progress', 0),
#         elapsed_time_seconds=round(elapsed_time, 2),
#         total_pages=status_info.get('total_pages', 0),
#         processed_pages=status_info.get('processed_pages', 0),
#         message=f"Estado: {status_info['status'].upper()}"
#     )

# @router.post("/{pdf_id}/search", response_model=SearchResponse)
# async def search_pdf(pdf_id: str, search_request: SearchRequest):
#     """Busca un término en el PDF con caché y optimizaciones"""
#     start_time = time.time()
    
#     # Verificar si existe
#     if pdf_id not in pdf_storage:
#         raise HTTPException(status_code=404, detail="PDF no encontrado")
    
#     # Cargar texto desde caché o archivo
#     if pdf_id in text_cache:
#         text = text_cache[pdf_id]
#         cache_hit = True
#     else:
#         pdf_data = pdf_storage[pdf_id]
#         if os.path.exists(pdf_data['text_path']):
#             # Verificar si está comprimido
#             if pdf_data['text_path'].endswith('.gz'):
#                 import gzip
#                 with gzip.open(pdf_data['text_path'], 'rt', encoding='utf-8') as f:
#                     text = f.read()
#             else:
#                 with open(pdf_data['text_path'], 'r', encoding='utf-8') as f:
#                     text = f.read()
            
#             # Cachear si no es demasiado grande
#             if len(text.encode('utf-8')) < 5 * 1024 * 1024:  # 5MB
#                 text_cache[pdf_id] = text
#             cache_hit = False
#         else:
#             raise HTTPException(status_code=404, detail="Texto no encontrado")
    
#     # Realizar búsqueda optimizada
#     results = pdf_service.search_in_text(
#         text, 
#         search_request.term, 
#         search_request.case_sensitive,
#         context_chars=search_request.context_chars or 100
#     )
    
#     execution_time = time.time() - start_time
    
#     # Aplicar paginación si se solicita
#     if search_request.page and search_request.page_size:
#         page = search_request.page - 1
#         page_size = search_request.page_size
#         start_idx = page * page_size
#         end_idx = start_idx + page_size
#         paginated_results = results[start_idx:end_idx]
#     else:
#         paginated_results = results[:search_request.max_results or 100]
    
#     return SearchResponse(
#         term=search_request.term,
#         total_matches=len(results),
#         results=paginated_results,
#         pdf_id=pdf_id,
#         execution_time=execution_time,
#         cache_hit=cache_hit,
#         page=search_request.page or 1,
#         page_size=search_request.page_size or len(paginated_results),
#         total_pages=len(pdf_storage[pdf_id].get('pages', []))
#     )

# @router.get("/{pdf_id}/text")
# async def get_pdf_text(pdf_id: str, stream: bool = Query(False)):
#     """Obtiene el texto completo del PDF con opción de streaming"""
#     if pdf_id not in pdf_storage:
#         raise HTTPException(status_code=404, detail="PDF no encontrado")
    
#     text_path = pdf_storage[pdf_id]['text_path']
    
#     if not os.path.exists(text_path):
#         raise HTTPException(status_code=404, detail="Texto no encontrado")
    
#     # Streaming para archivos grandes
#     if stream and os.path.getsize(text_path) > 1 * 1024 * 1024:  # > 1MB
#         def iterfile():
#             with open(text_path, 'rb') as f:
#                 while chunk := f.read(1024 * 1024):  # Chunks de 1MB
#                     yield chunk
        
#         return StreamingResponse(
#             iterfile(),
#             media_type='text/plain',
#             headers={'Content-Disposition': f'attachment; filename="{pdf_id}_texto.txt"'}
#         )
#     else:
#         return FileResponse(
#             text_path, 
#             media_type='text/plain',
#             filename=f"{pdf_id}_texto.txt"
#         )

# @router.get("/{pdf_id}/info", response_model=PDFInfo)
# async def get_pdf_info(pdf_id: str):
#     """Obtiene información detallada del PDF"""
#     if pdf_id not in pdf_storage:
#         raise HTTPException(status_code=404, detail="PDF no encontrado")
    
#     pdf_data = pdf_storage[pdf_id]
    
#     # Obtener información adicional del archivo
#     text_file_size = 0
#     if os.path.exists(pdf_data['text_path']):
#         text_file_size = os.path.getsize(pdf_data['text_path'])
    
#     # Calcular densidad de texto
#     text_density = 0
#     if pdf_data.get('text_size', 0) > 0 and pdf_data['pages'] > 0:
#         text_density = pdf_data['text_size'] / pdf_data['pages']
    
#     return PDFInfo(
#         id=pdf_id,
#         filename=pdf_data['filename'],
#         upload_date=datetime.fromtimestamp(pdf_data['upload_time']),
#         size=pdf_data['size'],
#         pages=pdf_data['pages'],
#         has_text=pdf_data.get('text_size', 0) > 0,
#         text_file_size=text_file_size,
#         text_size=pdf_data.get('text_size', 0),
#         used_ocr=pdf_data.get('used_ocr', False),
#         processing_time_seconds=round(pdf_data.get('processing_time', 0), 2),
#         language=pdf_data.get('language', 'spa'),
#         text_density=round(text_density, 2),
#         analysis=pdf_data.get('analysis', {})
#     )

# @router.get("/{pdf_id}/analysis", response_model=PDFAnalysis)
# async def get_pdf_analysis(pdf_id: str):
#     """Obtiene análisis detallado del PDF"""
#     if pdf_id not in pdf_storage:
#         raise HTTPException(status_code=404, detail="PDF no encontrado")
    
#     pdf_data = pdf_storage[pdf_id]
    
#     if not os.path.exists(pdf_data['pdf_path']):
#         raise HTTPException(status_code=404, detail="Archivo PDF no encontrado")
    
#     # Realizar análisis
#     analysis = pdf_service.analyze_pdf_structure(pdf_data['pdf_path'])
    
#     # Agregar información de procesamiento
#     analysis.update({
#         'processing_method': 'ocr' if pdf_data.get('used_ocr') else 'text_extraction',
#         'actual_processing_time': pdf_data.get('processing_time', 0),
#         'recommended_for_future': {
#             'use_ocr': analysis.get('likely_scanned', True),
#             'batch_size': analysis.get('suggested_batch_size', 10),
#             'dpi': analysis.get('recommended_dpi', 200)
#         }
#     })
    
#     return PDFAnalysis(**analysis)

# @router.get("/list")
# async def list_pdfs(
#     page: int = Query(1, ge=1),
#     page_size: int = Query(20, ge=1, le=100),
#     sort_by: str = Query("upload_time", regex="^(upload_time|filename|size|pages)$"),
#     order: str = Query("desc", regex="^(asc|desc)$")
# ):
#     """Lista todos los PDFs procesados con paginación y ordenamiento"""
#     total = len(pdf_storage)
    
#     # Convertir a lista para ordenar
#     pdf_list = list(pdf_storage.items())
    
#     # Ordenar
#     reverse = order == "desc"
#     if sort_by == "filename":
#         pdf_list.sort(key=lambda x: x[1]['filename'].lower(), reverse=reverse)
#     elif sort_by == "size":
#         pdf_list.sort(key=lambda x: x[1]['size'], reverse=reverse)
#     elif sort_by == "pages":
#         pdf_list.sort(key=lambda x: x[1]['pages'], reverse=reverse)
#     else:  # upload_time por defecto
#         pdf_list.sort(key=lambda x: x[1]['upload_time'], reverse=reverse)
    
#     # Paginar
#     start_idx = (page - 1) * page_size
#     end_idx = start_idx + page_size
#     paginated_items = pdf_list[start_idx:end_idx]
    
#     return {
#         "total": total,
#         "page": page,
#         "page_size": page_size,
#         "total_pages": (total + page_size - 1) // page_size,
#         "pdfs": [
#             {
#                 "id": pdf_id,
#                 "filename": data['filename'],
#                 "pages": data['pages'],
#                 "size": data['size'],
#                 "size_mb": round(data['size'] / (1024 * 1024), 2),
#                 "upload_date": datetime.fromtimestamp(data['upload_time']).isoformat(),
#                 "has_text": data.get('text_size', 0) > 0,
#                 "used_ocr": data.get('used_ocr', False)
#             }
#             for pdf_id, data in paginated_items
#         ]
#     }

# @router.delete("/{pdf_id}")
# async def delete_pdf(pdf_id: str):
#     """Elimina un PDF y sus archivos asociados"""
#     if pdf_id not in pdf_storage:
#         raise HTTPException(status_code=404, detail="PDF no encontrado")
    
#     pdf_data = pdf_storage[pdf_id]
    
#     # Eliminar archivos
#     files_deleted = 0
#     errors = []
    
#     files_to_delete = [
#         (pdf_data['pdf_path'], 'PDF'),
#         (pdf_data['text_path'], 'texto'),
#     ]
    
#     # Verificar si hay archivo comprimido
#     if os.path.exists(pdf_data['text_path'] + '.gz'):
#         files_to_delete.append((pdf_data['text_path'] + '.gz', 'texto comprimido'))
    
#     for file_path, file_type in files_to_delete:
#         try:
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#                 files_deleted += 1
#         except Exception as e:
#             errors.append(f"Error eliminando {file_type}: {str(e)}")
    
#     # Eliminar de memoria y caché
#     if pdf_id in pdf_storage:
#         del pdf_storage[pdf_id]
#     if pdf_id in text_cache:
#         del text_cache[pdf_id]
    
#     return {
#         "message": f"PDF {pdf_id} eliminado",
#         "files_deleted": files_deleted,
#         "errors": errors if errors else None
#     }

# @router.post("/quick-search")
# async def quick_search(
#     file: UploadFile = File(...),
#     search_term: str = Query(...),
#     use_ocr: bool = Query(True),
#     language: str = Query("spa"),
#     max_results: int = Query(50, ge=1, le=200)
# ):
#     """Búsqueda rápida sin guardar el PDF permanentemente"""
#     try:
#         start_time = time.time()
        
#         # Leer archivo
#         file_bytes = await file.read()
        
#         # Limitar tamaño para búsqueda rápida
#         if len(file_bytes) > 50 * 1024 * 1024:  # 50MB máximo para búsqueda rápida
#             raise HTTPException(
#                 status_code=400, 
#                 detail="Para archivos mayores a 50MB use el endpoint /upload"
#             )
        
#         # Guardar temporalmente
#         temp_id = hashlib.md5(file_bytes).hexdigest()[:16]
#         temp_path = os.path.join(tempfile.gettempdir(), f"quick_{temp_id}.pdf")
        
#         with open(temp_path, "wb") as f:
#             f.write(file_bytes)
        
#         # Extraer texto con procesamiento rápido
#         text, pages, _ = pdf_service.extract_text_from_pdf(
#             temp_path, 
#             use_ocr=use_ocr,
#             language=language,
#             batch_size=10  # Batch más pequeño para respuesta rápida
#         )
        
#         # Buscar
#         results = pdf_service.search_in_text(
#             text, 
#             search_term,
#             context_chars=80  # Contexto más corto para búsqueda rápida
#         )
        
#         # Limpiar archivo temporal
#         try:
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)
#         except:
#             pass
        
#         execution_time = time.time() - start_time
        
#         return SearchResponse(
#             term=search_term,
#             total_matches=len(results),
#             results=results[:max_results],
#             pdf_id=f"temp_{temp_id}",
#             execution_time=execution_time,
#             pages=pages,
#             message=f"Búsqueda completada en {execution_time:.2f}s"
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error en búsqueda rápida: {str(e)}")

# @router.post("/batch-search")
# async def batch_search(
#     pdf_ids: List[str] = Query(...),
#     search_term: str = Query(...),
#     case_sensitive: bool = Query(False)
# ):
#     """Busca un término en múltiples PDFs simultáneamente"""
#     try:
#         start_time = time.time()
        
#         results_by_pdf = {}
#         errors = []
        
#         for pdf_id in pdf_ids[:10]:  # Máximo 10 PDFs por búsqueda
#             if pdf_id in pdf_storage:
#                 try:
#                     # Usar texto en caché si está disponible
#                     if pdf_id in text_cache:
#                         text = text_cache[pdf_id]
#                     else:
#                         pdf_data = pdf_storage[pdf_id]
#                         if os.path.exists(pdf_data['text_path']):
#                             if pdf_data['text_path'].endswith('.gz'):
#                                 import gzip
#                                 with gzip.open(pdf_data['text_path'], 'rt', encoding='utf-8') as f:
#                                     text = f.read()
#                             else:
#                                 with open(pdf_data['text_path'], 'r', encoding='utf-8') as f:
#                                     text = f.read()
#                         else:
#                             errors.append(f"Texto no encontrado para {pdf_id}")
#                             continue
                    
#                     # Realizar búsqueda
#                     search_results = pdf_service.search_in_text(
#                         text, 
#                         search_term, 
#                         case_sensitive
#                     )
                    
#                     results_by_pdf[pdf_id] = {
#                         'filename': pdf_storage[pdf_id]['filename'],
#                         'matches': len(search_results),
#                         'results': search_results[:20]  # Máximo 20 por PDF
#                     }
                    
#                 except Exception as e:
#                     errors.append(f"Error buscando en {pdf_id}: {str(e)}")
#             else:
#                 errors.append(f"PDF {pdf_id} no encontrado")
        
#         total_matches = sum(data['matches'] for data in results_by_pdf.values())
#         execution_time = time.time() - start_time
        
#         return {
#             "search_term": search_term,
#             "total_pdfs_searched": len(pdf_ids),
#             "total_matches": total_matches,
#             "results_by_pdf": results_by_pdf,
#             "errors": errors if errors else None,
#             "execution_time": execution_time
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error en búsqueda por lotes: {str(e)}")

# @router.get("/stats")
# async def get_stats():
#     """Obtiene estadísticas del servicio"""
#     total_pdfs = len(pdf_storage)
#     total_pages = sum(data['pages'] for data in pdf_storage.values())
#     total_size = sum(data['size'] for data in pdf_storage.values())
    
#     # PDFs con OCR
#     pdfs_with_ocr = sum(1 for data in pdf_storage.values() if data.get('used_ocr', False))
    
#     # Distribución por tamaño
#     size_distribution = {
#         'small': sum(1 for data in pdf_storage.values() if data['size'] < 1024 * 1024),  # < 1MB
#         'medium': sum(1 for data in pdf_storage.values() if 1024 * 1024 <= data['size'] < 10 * 1024 * 1024),  # 1-10MB
#         'large': sum(1 for data in pdf_storage.values() if data['size'] >= 10 * 1024 * 1024),  # >= 10MB
#     }
    
#     return {
#         "total_pdfs": total_pdfs,
#         "total_pages": total_pages,
#         "total_size_mb": round(total_size / (1024 * 1024), 2),
#         "average_pages_per_pdf": round(total_pages / max(total_pdfs, 1), 2),
#         "pdfs_with_ocr": pdfs_with_ocr,
#         "pdfs_without_ocr": total_pdfs - pdfs_with_ocr,
#         "size_distribution": size_distribution,
#         "text_cache_size": len(text_cache),
#         "processing_queue_size": len(processing_queue)
#     }