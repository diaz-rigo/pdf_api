# import fitz  # PyMuPDF
# import re
# import os
# import uuid
# import hashlib
# from typing import List, Tuple, Optional
# from datetime import datetime
# from .ocr_service import OCRService
# from app.core.config import settings

# class PDFService:
#     def __init__(self):
#         self.ocr_service = OCRService(settings.TESSERACT_CMD)
    
#     def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = True, language: str = 'spa'):
#         doc = fitz.open(pdf_path)
#         text = ""
#         used_ocr = False
#         total_pages = len(doc)  # 👈 guardar antes

#         for page_num in range(total_pages):
#             page = doc.load_page(page_num)
#             page_text = page.get_text()

#             if not page_text.strip() and use_ocr:
#                 try:
#                     pix = page.get_pixmap(dpi=200)
#                     img_bytes = pix.tobytes("png")
#                     page_text = self.ocr_service.extract_text_from_image(
#                         img_bytes,
#                         language
#                     )
#                     used_ocr = True
#                 except Exception as e:
#                     print(f"OCR falló en página {page_num + 1}: {e}")
#                     page_text = ""

#             text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"

#         doc.close()
#         return text, total_pages, used_ocr

    
#     def search_in_text(self, text: str, search_term: str, case_sensitive: bool = False) -> List[dict]:
#         """Busca un término en el texto y devuelve coincidencias con contexto"""
#         if not case_sensitive:
#             search_term = search_term.lower()
#             text_lower = text.lower()
#         else:
#             text_lower = text
        
#         results = []
        
#         # Encontrar todas las coincidencias
#         for match in re.finditer(re.escape(search_term), text_lower):
#             start = max(0, match.start() - 100)
#             end = min(len(text), match.end() + 100)
            
#             context = text[start:end]
#             snippet = text[match.start():match.end()]
            
#             # Calcular página
#             page = 1
#             page_markers = [m.start() for m in re.finditer(r'--- Página \d+ ---', text)]
#             for marker in page_markers:
#                 if marker < match.start():
#                     page = int(re.search(r'--- Página (\d+) ---', text[marker:marker+20]).group(1))
            
#             results.append({
#                 'page': page,
#                 'position': match.start(),
#                 'context': context,
#                 'snippet': snippet
#             })
        
#         return results
    
#     def save_extracted_text(self, text: str, pdf_id: str) -> str:
#         """Guarda el texto extraído en un archivo"""
#         filename = f"{pdf_id}.txt"
#         filepath = os.path.join(settings.EXTRACTED_FOLDER, filename)
        
#         with open(filepath, 'w', encoding='utf-8') as f:
#             f.write(text)
        
#         return filepath
    
#     def generate_pdf_id(self, filename: str, file_bytes: bytes) -> str:
#         """Genera un ID único para el PDF basado en nombre y contenido"""
#         content_hash = hashlib.md5(file_bytes).hexdigest()[:8]
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         return f"{timestamp}_{content_hash}_{filename[:20]}".replace(" ", "_")
    
#     def get_pdf_info(self, pdf_path: str) -> dict:
#         """Obtiene información del PDF"""
#         doc = fitz.open(pdf_path)
#         info = {
#             'pages': len(doc),
#             'size': os.path.getsize(pdf_path),
#             'has_text': False
#         }
        
#         # Verificar si tiene texto extraíble
#         for page_num in range(min(3, len(doc))):  # Revisa las primeras 3 páginas
#             page = doc.load_page(page_num)
#             if page.get_text().strip():
#                 info['has_text'] = True
#                 break
        
#         doc.close()
#         return info