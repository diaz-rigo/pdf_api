# C:\Angular\OCR\pdf_api\app\services\pdf_service.py

import fitz  # PyMuPDF
import re
import os
import uuid
import hashlib
import logging
import concurrent.futures
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from .ocr_service import OCRService
from app.core.config import settings
import tempfile
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.ocr_service = OCRService(settings.TESSERACT_CMD)
        self.temp_dir = tempfile.gettempdir()
        
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = True, 
                             language: str = 'spa', batch_size: int = 20) -> Tuple[str, int, bool]:
        """
        Extrae texto de PDF con optimización para grandes documentos
        
        Args:
            pdf_path: Ruta al archivo PDF
            use_ocr: Usar OCR si es necesario
            language: Idioma para OCR
            batch_size: Número de páginas por lote
        
        Returns:
            Tupla con (texto, total_páginas, usó_ocr)
        """
        doc = None
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            used_ocr = False
            
            logger.info(f"Iniciando extracción de {total_pages} páginas")
            
            # Estrategia según número de páginas
            if total_pages > 100:
                return self._extract_large_pdf(doc, total_pages, use_ocr, language, batch_size)
            else:
                return self._extract_small_pdf(doc, total_pages, use_ocr, language)
                
        except Exception as e:
            logger.error(f"Error extrayendo texto: {e}")
            return "", 0, False
        finally:
            if doc:
                doc.close()
            gc.collect()  # Liberar memoria
    
    def _extract_small_pdf(self, doc: fitz.Document, total_pages: int, 
                          use_ocr: bool, language: str) -> Tuple[str, int, bool]:
        """Extrae PDFs pequeños (< 100 páginas)"""
        text_parts = []
        used_ocr = False
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            # Usar OCR solo si es necesario
            if not page_text.strip() and use_ocr:
                page_text = self._ocr_page(page, page_num, language)
                if page_text:
                    used_ocr = True
            
            text_parts.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
            
            # Log cada 10 páginas
            if (page_num + 1) % 10 == 0:
                logger.info(f"Procesadas {page_num + 1}/{total_pages} páginas")
        
        return "\n".join(text_parts), total_pages, used_ocr
    
    def _extract_large_pdf(self, doc: fitz.Document, total_pages: int, 
                          use_ocr: bool, language: str, batch_size: int) -> Tuple[str, int, bool]:
        """Extrae PDFs grandes con procesamiento por lotes y paralelización"""
        text_parts = [""] * total_pages
        used_ocr = False
        
        # Procesar por lotes
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            
            logger.info(f"Procesando lote {batch_start + 1}-{batch_end} de {total_pages}")
            
            # Procesar páginas del lote actual
            for page_num in range(batch_start, batch_end):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if not page_text.strip() and use_ocr:
                    page_text = self._ocr_page(page, page_num, language)
                    if page_text:
                        used_ocr = True
                
                text_parts[page_num] = f"\n--- Página {page_num + 1} ---\n{page_text}"
            
            # Liberar memoria del lote
            gc.collect()
        
        return "".join(text_parts), total_pages, used_ocr
    
    def _ocr_page(self, page: fitz.Page, page_num: int, language: str) -> str:
        """Ejecuta OCR en una página específica"""
        try:
            # Optimizar DPI según contenido
            dpi = 200  # DPI base
            
            # Ajustar DPI basado en tamaño de página
            rect = page.rect
            if rect.width > 1000 or rect.height > 1000:
                dpi = 150  # DPI más bajo para páginas grandes
            
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            
            return self.ocr_service.extract_text_from_image(img_bytes, language)
            
        except Exception as e:
            logger.error(f"OCR falló en página {page_num + 1}: {e}")
            return ""
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, use_ocr: bool = True, 
                                   language: str = 'spa') -> Tuple[str, int, bool]:
        """Extrae texto directamente desde bytes del PDF"""
        try:
            # Guardar temporalmente para procesar
            temp_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex}.pdf")
            
            with open(temp_path, 'wb') as f:
                f.write(pdf_bytes)
            
            # Procesar archivo temporal
            result = self.extract_text_from_pdf(temp_path, use_ocr, language)
            
            # Limpiar archivo temporal
            try:
                os.remove(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando bytes PDF: {e}")
            return "", 0, False
    
    def search_in_text(self, text: str, search_term: str, 
                      case_sensitive: bool = False, context_chars: int = 100) -> List[dict]:
        """Busca un término en el texto con índice de páginas precalculado"""
        if not text or not search_term:
            return []
        
        # Precalcular posiciones de páginas para búsqueda más rápida
        page_positions = []
        page_matches = list(re.finditer(r'--- Página (\d+) ---', text))
        
        for match in page_matches:
            page_num = int(match.group(1))
            page_positions.append((page_num, match.start()))
        
        results = []
        search_text = text if case_sensitive else text.lower()
        search_term_adj = search_term if case_sensitive else search_term.lower()
        
        # Buscar todas las ocurrencias
        for match in re.finditer(re.escape(search_term_adj), search_text):
            start_pos = match.start()
            
            # Encontrar página correspondiente (búsqueda binaria)
            page_num = 1
            for i in range(len(page_positions) - 1):
                if page_positions[i][1] <= start_pos < page_positions[i + 1][1]:
                    page_num = page_positions[i][0]
                    break
            else:
                if page_positions:
                    page_num = page_positions[-1][0]
            
            # Extraer contexto
            start = max(0, start_pos - context_chars)
            end = min(len(text), match.end() + context_chars)
            
            context = text[start:end]
            snippet = text[match.start():match.end()]
            
            results.append({
                'page': page_num,
                'position': start_pos,
                'context': context,
                'snippet': snippet,
                'score': self._calculate_relevance_score(context, search_term)
            })
        
        # Ordenar por relevancia
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _calculate_relevance_score(self, context: str, search_term: str) -> float:
        """Calcula puntuación de relevancia para resultados de búsqueda"""
        # Implementación básica - puede mejorarse
        term_count = context.lower().count(search_term.lower())
        context_len = len(context)
        
        # Más puntos por más ocurrencias y contexto más corto (más específico)
        score = (term_count * 10) + (100 / max(1, context_len / 100))
        return score
    
    def save_extracted_text(self, text: str, pdf_id: str) -> str:
        """Guarda el texto extraído comprimido si es muy grande"""
        filename = f"{pdf_id}.txt"
        filepath = os.path.join(settings.EXTRACTED_FOLDER, filename)
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Comprimir si el texto es muy grande (> 10MB)
        if len(text) > 10 * 1024 * 1024:
            import gzip
            filepath += '.gz'
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(text)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
        
        logger.info(f"Texto guardado en {filepath}")
        return filepath
    
    def generate_pdf_id(self, filename: str, file_bytes: bytes) -> str:
        """Genera un ID único optimizado"""
        content_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\-_]', '_', filename[:30])
        
        return f"{timestamp}_{content_hash}_{safe_name}"
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Obtiene información detallada del PDF"""
        try:
            doc = fitz.open(pdf_path)
            info = {
                'pages': len(doc),
                'size': os.path.getsize(pdf_path),
                'size_mb': round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
                'has_text': False,
                'estimated_processing_time': 0,
                'page_sizes': []
            }
            
            # Estimar tiempo de procesamiento
            avg_page_time = 0.5  # segundos por página (estimado)
            info['estimated_processing_time'] = round(info['pages'] * avg_page_time / 60, 2)  # en minutos
            
            # Verificar primeras páginas y tamaño
            sample_pages = min(5, len(doc))
            total_text_chars = 0
            
            for page_num in range(sample_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    info['has_text'] = True
                    total_text_chars += len(page_text)
                
                # Tamaño de página
                rect = page.rect
                info['page_sizes'].append({
                    'width': round(rect.width),
                    'height': round(rect.height)
                })
            
            # Calcular densidad de texto
            if sample_pages > 0:
                info['text_density'] = round(total_text_chars / sample_pages, 2)
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo info PDF: {e}")
            return {}
    
    def analyze_pdf_structure(self, pdf_path: str) -> dict:
        """Analiza la estructura del PDF para optimizar procesamiento"""
        doc = fitz.open(pdf_path)
        
        analysis = {
            'total_pages': len(doc),
            'likely_scanned': False,
            'recommended_dpi': 200,
            'suggested_batch_size': 10,
            'processing_strategy': 'hybrid'  # 'text_only', 'ocr_only', 'hybrid'
        }
        
        # Muestrear páginas
        sample_indices = [0, len(doc)//2, -1]
        text_pages = 0
        
        for idx in sample_indices:
            if idx < len(doc):
                page = doc.load_page(idx)
                page_text = page.get_text()
                
                if len(page_text.strip()) > 100:
                    text_pages += 1
        
        # Determinar estrategia
        if text_pages == len(sample_indices):
            analysis['processing_strategy'] = 'text_only'
            analysis['recommended_dpi'] = 150
        elif text_pages == 0:
            analysis['processing_strategy'] = 'ocr_only'
            analysis['likely_scanned'] = True
            analysis['recommended_dpi'] = 300
            analysis['suggested_batch_size'] = 5  # Lotes más pequeños para OCR intensivo
        else:
            analysis['processing_strategy'] = 'hybrid'
        
        # Ajustar tamaño de lote según número de páginas
        if analysis['total_pages'] > 200:
            analysis['suggested_batch_size'] = 15
        elif analysis['total_pages'] > 500:
            analysis['suggested_batch_size'] = 20
        
        doc.close()
        return analysis