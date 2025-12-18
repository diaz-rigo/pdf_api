# C:\Angular\OCR\pdf_api\app\services\pdf_service.py
import re
import fitz  # PyMuPDF
import re
import os
import uuid
import hashlib
import logging
import io
import concurrent.futures
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from .ocr_service import OCRService
from app.core.config import settings
import tempfile
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.ocr_service = OCRService(settings.TESSERACT_CMD)
        self.temp_dir = tempfile.gettempdir()
        # Pool de hilos para procesamiento paralelo de documentos
        self.doc_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pdf_proc_")
        # Pool de hilos para procesamiento paralelo dentro de un documento
        self.page_executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_PAGES, 
                                               thread_name_prefix="page_proc_")
        # Lock para sincronización
        self.processing_lock = threading.Lock()
    
    # ========== MÉTODO FALTANTE AÑADIDO ==========
    def generate_pdf_id(self, filename: str, file_bytes: bytes) -> str:
        """Genera un ID único optimizado para el PDF"""
        try:
            # Hash del contenido
            content_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Nombre seguro del archivo
            safe_name = re.sub(r'[^\w\-_]', '_', Path(filename).stem[:30])
            
            # ID compuesto
            pdf_id = f"{timestamp}_{content_hash}_{safe_name}"
            
            logger.debug(f"ID generado para {filename}: {pdf_id}")
            return pdf_id
            
        except Exception as e:
            logger.error(f"Error generando PDF ID: {e}")
            # Fallback a UUID si hay error
            return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # ========== MÉTODOS EXISTENTES ==========
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = True, 
                             language: str = 'spa', batch_size: int = 20,
                             generate_ocr_pdf: bool = True) -> Tuple[str, int, bool, Optional[str]]:
        """
        Extrae texto de PDF y opcionalmente genera PDF con capa de texto OCR
        
        Args:
            pdf_path: Ruta al archivo PDF
            use_ocr: Usar OCR si es necesario
            language: Idioma para OCR
            batch_size: Número de páginas por lote
            generate_ocr_pdf: Generar PDF con capa de texto OCR
        
        Returns:
            Tupla con (texto, total_páginas, usó_ocr, ruta_pdf_ocr)
        """
        doc = None
        ocr_pdf_path = None
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            used_ocr = False
            
            logger.info(f"Iniciando extracción de {total_pages} páginas: {pdf_path}")
            
            # Análisis previo para optimizar estrategia
            analysis = self._analyze_for_optimization(doc, total_pages)
            
            # Determinar estrategia basada en análisis
            if total_pages > 100 or analysis['likely_complex']:
                text, used_ocr = self._extract_large_pdf_optimized(doc, total_pages, 
                                                                  use_ocr, language, 
                                                                  analysis)
            else:
                text, used_ocr = self._extract_small_pdf_optimized(doc, total_pages, 
                                                                  use_ocr, language)
            
            # Generar PDF con capa OCR si se usó OCR
            if used_ocr and generate_ocr_pdf:
                ocr_pdf_path = self._generate_ocr_pdf(doc, text, pdf_path, total_pages)
                
            return text, total_pages, used_ocr, ocr_pdf_path
                
        except Exception as e:
            logger.error(f"Error extrayendo texto: {e}")
            return "", 0, False, None
        finally:
            if doc:
                doc.close()
            gc.collect()
    
    def _analyze_for_optimization(self, doc: fitz.Document, total_pages: int) -> Dict[str, Any]:
        """Analiza el PDF para determinar estrategia óptima"""
        analysis = {
            'likely_complex': False,
            'text_density': 0,
            'avg_page_size': 0,
            'likely_scanned': False
        }
        
        # Muestrear páginas estratégicamente
        sample_indices = [0, min(10, total_pages-1), min(50, total_pages-1), -1]
        sample_indices = list(set(sample_indices))  # Eliminar duplicados
        
        total_text_chars = 0
        total_area = 0
        
        for idx in sample_indices:
            if 0 <= idx < total_pages:
                page = doc.load_page(idx)
                page_text = page.get_text()
                total_text_chars += len(page_text.strip())
                
                rect = page.rect
                total_area += rect.width * rect.height
        
        # Calcular densidad de texto
        if len(sample_indices) > 0:
            analysis['text_density'] = total_text_chars / len(sample_indices)
            analysis['avg_page_size'] = total_area / len(sample_indices)
            
            # Determinar si es escaneado/complejo
            if analysis['text_density'] < 100:  # Menos de 100 caracteres por página en promedio
                analysis['likely_scanned'] = True
                analysis['likely_complex'] = True
        
        # Marcar como complejo si tiene muchas páginas
        if total_pages > 200:
            analysis['likely_complex'] = True
            
        return analysis
    
    def _extract_small_pdf_optimized(self, doc: fitz.Document, total_pages: int, 
                                   use_ocr: bool, language: str) -> Tuple[str, bool]:
        """Extrae PDFs pequeños optimizado"""
        text_parts = []
        used_ocr = False
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            # Usar OCR solo si es necesario
            if not page_text.strip() and use_ocr:
                page_text = self._ocr_page_optimized(page, page_num, language)
                if page_text:
                    used_ocr = True
            
            text_parts.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
        
        return "\n".join(text_parts), used_ocr
    
    def _extract_large_pdf_optimized(self, doc: fitz.Document, total_pages: int, 
                                   use_ocr: bool, language: str, 
                                   analysis: Dict) -> Tuple[str, bool]:
        """Extrae PDFs grandes con procesamiento paralelo optimizado"""
        text_parts = [""] * total_pages
        used_ocr = False
        
        # Determinar tamaño de lote dinámico basado en análisis
        batch_size = self._calculate_dynamic_batch_size(total_pages, analysis)
        
        logger.info(f"Procesando {total_pages} páginas en lotes de {batch_size}")
        
        # Procesar por lotes con paralelismo
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Procesando lote {batch_start + 1}-{batch_end} de {total_pages}")
            
            # Cargar todas las páginas del lote primero
            pages = []
            for page_num in range(batch_start, batch_end):
                pages.append(doc.load_page(page_num))
            
            # Procesar páginas del lote en paralelo
            futures = []
            for i, page in enumerate(pages):
                page_num = batch_start + i
                future = self.page_executor.submit(
                    self._process_single_page, 
                    page, page_num, use_ocr, language
                )
                futures.append((page_num, future))
            
            # Recoger resultados
            for page_num, future in futures:
                try:
                    page_text, page_used_ocr = future.result(timeout=settings.OCR_TIMEOUT_PER_PAGE)
                    text_parts[page_num] = f"\n--- Página {page_num + 1} ---\n{page_text}"
                    if page_used_ocr:
                        used_ocr = True
                except Exception as e:
                    logger.error(f"Error procesando página {page_num + 1}: {e}")
                    text_parts[page_num] = f"\n--- Página {page_num + 1} ---\n[Error de procesamiento]"
            
            # Liberar memoria del lote
            pages.clear()
            gc.collect()
            
            # Pequeña pausa para evitar sobrecarga
            if batch_end % 100 == 0:
                import time
                time.sleep(0.1)
        
        return "".join(text_parts), used_ocr
    
    def _process_single_page(self, page: fitz.Page, page_num: int, 
                           use_ocr: bool, language: str) -> Tuple[str, bool]:
        """Procesa una sola página (para paralelización)"""
        page_text = page.get_text()
        used_ocr = False
        
        if not page_text.strip() and use_ocr:
            page_text = self._ocr_page_optimized(page, page_num, language)
            used_ocr = bool(page_text.strip())
        
        return page_text, used_ocr
    
    def _calculate_dynamic_batch_size(self, total_pages: int, analysis: Dict) -> int:
        """Calcula tamaño de lote dinámico basado en características del PDF"""
        base_size = 20
        
        # Ajustar por número de páginas
        if total_pages > 500:
            base_size = 15
        if total_pages > 1000:
            base_size = 10
        
        # Ajustar por complejidad
        if analysis['likely_scanned']:
            base_size = max(5, base_size // 2)  # Lotes más pequeños para OCR intensivo
        
        return min(base_size, settings.DEFAULT_BATCH_SIZE)
    
    def _ocr_page_optimized(self, page: fitz.Page, page_num: int, language: str) -> str:
        """Ejecuta OCR optimizado en una página específica"""
        try:
            # Optimizar DPI dinámicamente
            rect = page.rect
            page_area = rect.width * rect.height
            
            # DPI más bajo para páginas grandes, más alto para páginas pequeñas
            if page_area > 1000000:  # Páginas muy grandes
                dpi = 150
            elif page_area < 300000:  # Páginas pequeñas
                dpi = 300
            else:
                dpi = 200
            
            # Obtener imagen optimizada
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            img_bytes = pix.tobytes("png")
            
            return self.ocr_service.extract_text_from_image(img_bytes, language)
            
        except Exception as e:
            logger.error(f"OCR falló en página {page_num + 1}: {e}")
            return ""
    def _generate_ocr_pdf(self, original_doc: fitz.Document, extracted_text: str, 
                        original_path: str, total_pages: int) -> str:
        """
        Genera un clon del PDF con capa de texto OCR integrada y SELECTABLE
        PRESERVANDO EL FONDO ORIGINAL EXACTAMENTE COMO iLovePDF
        
        iLovePDF funciona así:
        1. Mantiene el PDF original exactamente igual (mismo fondo, imágenes, etc.)
        2. Añade una capa de texto invisible pero seleccionable encima
        3. El texto sigue el layout original pero es seleccionable y buscable
        """
        ocr_doc = None
        try:
            logger.info("Generando PDF con capa de texto OCR (estilo iLovePDF)...")
            
            # Crear nuevo documento COPIANDO las páginas originales
            ocr_doc = fitz.open()  # Crear documento vacío
            
            # COPIAR METADATOS DEL DOCUMENTO ORIGINAL
            if original_doc.metadata:
                ocr_doc.set_metadata(original_doc.metadata)
            
            # Procesar cada página
            for page_num in range(total_pages):
                # Obtener página original
                original_page = original_doc.load_page(page_num)
                
                # 1. COPIAR LA PÁGINA ORIGINAL COMPLETA (incluyendo imágenes y fondo)
                # Usar insert_pdf para copiar exactamente la página
                ocr_doc.insert_pdf(
                    original_doc,  # Documento de origen
                    from_page=page_num,  # Página de inicio
                    to_page=page_num,    # Página final (solo esta)
                )
                
                # 2. Obtener el texto OCR específico de esta página
                page_text = self._extract_page_text(extracted_text, page_num + 1)
                
                if page_text and page_text.strip():
                    # 3. Limpiar y preparar el texto
                    clean_text = self._clean_text_for_ocr_layer(page_text)
                    
                    # 4. Obtener la página recién insertada
                    new_page = ocr_doc[page_num]
                    
                    # 5. Añadir texto OCR como capa invisible pero seleccionable
                    # Usar un método más directo y efectivo
                    self._add_ocr_text_layer(new_page, clean_text, original_page.rect)
            
            # 6. Generar nombre de archivo
            original_name = Path(original_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ocr_pdf_path = os.path.join(
                settings.UPLOAD_FOLDER,
                f"{original_name}_OCR_{timestamp}.pdf"
            )
            
            # Asegurar que la carpeta existe
            os.makedirs(os.path.dirname(ocr_pdf_path), exist_ok=True)
            
            # 7. Guardar el PDF con optimizaciones
            ocr_doc.save(
                ocr_pdf_path,
                deflate=True,      # Compresión
                garbage=3,         # Limpieza de objetos no usados
                pretty=True,       # Formato legible
                clean=True,        # Limpiar estructura
                linear=True,       # Optimizado para web
                encryption=fitz.PDF_ENCRYPT_NONE,  # Sin encriptación
            )
            
            logger.info(f"✅ PDF con capa OCR generado: {ocr_pdf_path} ({os.path.getsize(ocr_pdf_path)} bytes)")
            
            # 8. Verificar que el archivo es válido
            if os.path.exists(ocr_pdf_path) and os.path.getsize(ocr_pdf_path) > 0:
                # Abrir y cerrar para verificar que no está corrupto
                test_doc = fitz.open(ocr_pdf_path)
                test_doc.close()
                logger.info("✅ PDF verificado y listo para usar")
                return ocr_pdf_path
            else:
                logger.error("❌ PDF generado es inválido o está vacío")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generando PDF con OCR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Intentar método alternativo
            return self._create_simple_ocr_pdf(original_doc, extracted_text, original_path, total_pages)
        finally:
            if ocr_doc:
                ocr_doc.close()
    def _add_ocr_text_layer(self, page: fitz.Page, text: str, page_rect: fitz.Rect):
        """
        Añade texto OCR como capa invisible pero seleccionable en la página
        
        Técnica usada:
        - Texto con color blanco (invisible sobre fondo blanco)
        - Fuente normal y tamaño legible
        - Opacidad total (no transparente, solo blanco)
        """
        try:
            # Configurar márgenes
            margin = 20  # Puntos
            text_rect = fitz.Rect(
                margin,
                margin,
                page_rect.width - margin,
                page_rect.height - margin
            )
            
            # DIVIDIR el texto en bloques más pequeños para mejor manejo
            text_blocks = text.split('\n')
            current_y = text_rect.y0
            
            for block in text_blocks:
                if not block.strip():
                    current_y += 12  # Espacio entre párrafos
                    continue
                    
                # Insertar cada bloque de texto por separado
                # Usar color blanco (1,1,1) - invisible en fondo blanco
                # Usar render_mode=3 para texto invisible pero selectable
                page.insert_text(
                    point=(text_rect.x0, current_y),
                    text=block,
                    fontsize=11,  # Tamaño normal pero invisible
                    fontname="helv",
                    color=(1, 1, 1),  # BLANCO PURO - INVISIBLE
                    render_mode=3,  # Render mode 3: texto invisible
                    # fill_color=(1, 1, 1, 0),  # Fondo transparente (opcional)
                )
                
                current_y += 14  # Interlineado
            
            logger.debug(f"Texto OCR añadido como capa invisible")
            
        except Exception as e:
            logger.warning(f"No se pudo añadir capa de texto OCR: {e}")
            # Método alternativo más simple
            try:
                # Insertar texto como anotación oculta
                annot = page.add_freetext_annot(
                    text_rect,
                    text,
                    fontsize=11,
                    fontname="helv",
                    text_color=(1, 1, 1),  # Texto blanco
                    fill_color=(1, 1, 1, 0),  # Fondo transparente
                    border_color=(1, 1, 1, 0),  # Sin borde
                )
                # Marcar como oculta
                annot.set_flags(fitz.PDF_ANNOT_IS_HIDDEN)
                annot.update()
            except:
                # Último recurso: texto muy pequeño
                page.insert_textbox(
                    text_rect,
                    text,
                    fontsize=0.1,  # Texto casi invisible
                    color=(1, 1, 1),  # Blanco
                    fontname="helv",
                    align=0,
                )
    def _create_simple_ocr_pdf(self, original_doc: fitz.Document, extracted_text: str,
                            original_path: str, total_pages: int) -> str:
        """
        Método alternativo simplificado estilo iLovePDF
        """
        try:
            logger.info("Usando método simple (fallback)...")
            
            # Crear nuevo documento COPIANDO TODO
            ocr_doc = fitz.open()
            
            # Copiar todo el documento original
            ocr_doc.insert_pdf(original_doc)
            
            # Añadir texto OCR a cada página
            for page_num in range(total_pages):
                page = ocr_doc[page_num]
                page_text = self._extract_page_text(extracted_text, page_num + 1)
                
                if page_text and page_text.strip():
                    clean_text = self._clean_text_for_ocr_layer(page_text)
                    
                    # Añadir texto con el método simplificado
                    rect = page.rect
                    margin = 20
                    text_rect = fitz.Rect(
                        margin, margin, 
                        rect.width - margin, 
                        rect.height - margin
                    )
                    
                    # Texto blanco invisible
                    page.insert_textbox(
                        text_rect,
                        clean_text,
                        fontsize=12,
                        color=(1, 1, 1),  # Blanco puro
                        fontname="helv",
                        align=0,  # Alineación izquierda
                        render_mode=3,  # Texto invisible
                    )
            
            # Guardar
            original_name = Path(original_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ocr_pdf_path = os.path.join(
                settings.UPLOAD_FOLDER,
                f"{original_name}_OCR_SIMPLE_{timestamp}.pdf"
            )
            
            os.makedirs(os.path.dirname(ocr_pdf_path), exist_ok=True)
            ocr_doc.save(ocr_pdf_path, deflate=True, clean=True)
            ocr_doc.close()
            
            logger.info(f"PDF simple generado: {ocr_pdf_path}")
            return ocr_pdf_path
            
        except Exception as e:
            logger.error(f"Error en método simple: {e}")
            return None

    def _clean_text_for_ocr_layer(self, text: str) -> str:
        """
        Limpia el texto para la capa OCR
        
        iLovePDF mantiene:
        - Saltos de línea naturales
        - Espaciado coherente
        - Caracteres especiales legibles
        """
        if not text:
            return ""
        
        # 1. Eliminar marcadores de página innecesarios
        text = re.sub(r'--- Página \d+ ---', '', text)
        
        # 2. Limpiar espacios múltiples pero mantener saltos de línea
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 3. Mantener saltos de línea significativos
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Solo mantener líneas no vacías
                cleaned_lines.append(line)
        
        # 4. Unir con saltos de línea pero no demasiados
        result = '\n'.join(cleaned_lines)
        
        # 5. Limitar longitud máxima por seguridad
        if len(result) > 10000:  # Máximo 10k caracteres por página
            result = result[:10000] + "... [texto truncado]"
        
        return result
    def _verify_selectable_text(self, doc: fitz.Document, filepath: str):
        """Verifica que el texto sea seleccionable"""
        try:
            selectable_pages = 0
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text and len(text.strip()) > 10:  # Al menos 10 caracteres
                    selectable_pages += 1
            
            logger.info(f"Verificación: {selectable_pages}/{len(doc)} páginas tienen texto seleccionable")
            
        except Exception as e:
            logger.warning(f"No se pudo verificar texto seleccionable: {e}")

    def _create_fallback_ocr_pdf(self, text: str, original_path: str, total_pages: int) -> str:
        """Crea un PDF alternativo solo con texto cuando falla el método principal"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import simpleSplit
            
            original_name = Path(original_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_path = os.path.join(
                settings.UPLOAD_FOLDER,
                f"{original_name}_OCR_FALLBACK_{timestamp}.pdf"
            )
            
            c = canvas.Canvas(fallback_path, pagesize=letter)
            width, height = letter
            
            # Configurar fuente y tamaño
            c.setFont("Helvetica", 10)
            
            # Dividir texto por páginas
            pages_text = self._split_text_by_pages(text, total_pages)
            
            for page_num, page_text in enumerate(pages_text):
                if page_num > 0:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                
                # Agregar encabezado
                c.drawString(50, height - 50, f"Página {page_num + 1} - Texto extraído")
                c.line(50, height - 55, width - 50, height - 55)
                
                # Agregar texto
                y = height - 80
                lines = page_text.split('\n')
                
                for line in lines:
                    if y < 50:  # Nueva página si nos quedamos sin espacio
                        c.showPage()
                        c.setFont("Helvetica", 10)
                        y = height - 50
                    
                    # Manejar líneas largas
                    if len(line) > 120:
                        sublines = simpleSplit(line, "Helvetica", 10, width - 100)
                        for subline in sublines:
                            c.drawString(50, y, subline)
                            y -= 15
                    else:
                        c.drawString(50, y, line)
                        y -= 15
            
            c.save()
            logger.info(f"PDF fallback generado: {fallback_path}")
            return fallback_path
            
        except Exception as e:
            logger.error(f"Error creando PDF fallback: {e}")
            return None
    def _extract_page_text(self, full_text: str, page_number: int) -> str:
        """
        Extrae el texto de una página específica del texto completo
        """
        try:
            # Buscar el marcador de página
            pattern = rf'--- Página {page_number} ---\n(.*?)(?=\n--- Página|\Z)'
            match = re.search(pattern, full_text, re.DOTALL)
            
            if match:
                return match.group(1).strip()
            else:
                # Alternativa: buscar por número de página
                lines = full_text.split('\n')
                in_target_page = False
                page_text = []
                
                for line in lines:
                    if f'--- Página {page_number} ---' in line:
                        in_target_page = True
                        continue
                    elif in_target_page and line.startswith('--- Página'):
                        break
                    elif in_target_page:
                        page_text.append(line)
                
                return '\n'.join(page_text).strip()
                
        except Exception as e:
            logger.error(f"Error extrayendo texto página {page_number}: {e}")
            return ""
    
    def process_multiple_pdfs(self, pdf_paths: List[str], use_ocr: bool = True, 
                            language: str = 'spa', max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        Procesa múltiples PDFs en paralelo
        
        Args:
            pdf_paths: Lista de rutas a PDFs
            use_ocr: Usar OCR si es necesario
            language: Idioma para OCR
            max_workers: Número máximo de documentos a procesar simultáneamente
        
        Returns:
            Lista de resultados por documento
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="multi_pdf_") as executor:
            # Crear tareas para cada PDF
            future_to_pdf = {
                executor.submit(
                    self._process_single_document, 
                    pdf_path, use_ocr, language
                ): pdf_path 
                for pdf_path in pdf_paths
            }
            
            # Recoger resultados
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result(timeout=3600)  # Timeout de 1 hora por documento
                    results.append(result)
                    logger.info(f"Completado: {pdf_path}")
                except Exception as e:
                    logger.error(f"Error procesando {pdf_path}: {e}")
                    results.append({
                        'pdf_path': pdf_path,
                        'error': str(e),
                        'success': False
                    })
        
        return results
    
    def _process_single_document(self, pdf_path: str, use_ocr: bool, language: str) -> Dict[str, Any]:
        """Procesa un documento individual para procesamiento por lotes"""
        try:
            text, pages, used_ocr, ocr_pdf_path = self.extract_text_from_pdf(
                pdf_path, use_ocr, language, generate_ocr_pdf=True
            )
            
            # Generar ID único
            with open(pdf_path, 'rb') as f:
                file_bytes = f.read()
            pdf_id = self.generate_pdf_id(Path(pdf_path).name, file_bytes)
            
            # Guardar texto extraído
            text_path = self.save_extracted_text(text, pdf_id)
            
            return {
                'pdf_path': pdf_path,
                'pdf_id': pdf_id,
                'text_path': text_path,
                'ocr_pdf_path': ocr_pdf_path,
                'pages': pages,
                'used_ocr': used_ocr,
                'text_length': len(text),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error en _process_single_document para {pdf_path}: {e}")
            raise
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, use_ocr: bool = True, 
                                   language: str = 'spa') -> Tuple[str, int, bool, Optional[str]]:
        """Extrae texto directamente desde bytes del PDF con generación de PDF OCR"""
        try:
            # Guardar temporalmente para procesar
            temp_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex}.pdf")
            
            with open(temp_path, 'wb') as f:
                f.write(pdf_bytes)
            
            # Procesar archivo temporal (con generación de PDF OCR)
            text, pages, used_ocr, ocr_pdf_path = self.extract_text_from_pdf(
                temp_path, use_ocr, language, generate_ocr_pdf=True
            )
            
            # Limpiar archivo temporal
            try:
                os.remove(temp_path)
            except:
                pass
            
            return text, pages, used_ocr, ocr_pdf_path
            
        except Exception as e:
            logger.error(f"Error procesando bytes PDF: {e}")
            return "", 0, False, None
    
    def search_in_text(self, text: str, search_term: str, 
                      case_sensitive: bool = False, context_chars: int = 100) -> List[dict]:
        """Busca un término en el texto optimizado"""
        if not text or not search_term:
            return []
        
        # Compilar regex para búsqueda más rápida
        page_regex = re.compile(r'--- Página (\d+) ---')
        search_regex = re.compile(re.escape(search_term), 
                                 re.IGNORECASE if not case_sensitive else 0)
        
        results = []
        pages = list(page_regex.finditer(text))
        
        # Búsqueda por secciones (más eficiente para textos grandes)
        for i, page_match in enumerate(pages):
            page_num = int(page_match.group(1))
            start_pos = page_match.start()
            end_pos = pages[i + 1].start() if i + 1 < len(pages) else len(text)
            
            page_section = text[start_pos:end_pos]
            
            for match in search_regex.finditer(page_section):
                global_pos = start_pos + match.start()
                
                # Extraer contexto
                start = max(start_pos, global_pos - context_chars)
                end = min(end_pos, global_pos + len(search_term) + context_chars)
                
                results.append({
                    'page': page_num,
                    'position': global_pos,
                    'context': text[start:end],
                    'snippet': text[global_pos:global_pos + len(search_term)],
                    'score': self._calculate_relevance_score(page_section, search_term)
                })
        
        # Ordenar por relevancia
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _calculate_relevance_score(self, context: str, search_term: str) -> float:
        """Calcula puntuación de relevancia para resultados de búsqueda"""
        try:
            term_lower = search_term.lower()
            context_lower = context.lower()
            
            # Contar ocurrencias
            term_count = context_lower.count(term_lower)
            
            # Factor de proximidad al inicio
            first_pos = context_lower.find(term_lower)
            position_factor = 1.0
            if first_pos >= 0:
                position_factor = max(0.1, 1.0 - (first_pos / len(context)))
            
            # Factor de densidad
            density_factor = term_count * 2
            
            # Puntuación combinada
            score = (term_count * 10) + (position_factor * 5) + density_factor
            
            return score
        except:
            return 0.0
    
    def save_extracted_text(self, text: str, pdf_id: str) -> str:
        """Guarda el texto extraído comprimido si es muy grande"""
        try:
            filename = f"{pdf_id}.txt"
            filepath = os.path.join(settings.EXTRACTED_FOLDER, filename)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Comprimir solo si es realmente grande
            if len(text) > 5 * 1024 * 1024:  # 5MB
                import gzip
                filepath += '.gz'
                with gzip.open(filepath, 'wt', encoding='utf-8', compresslevel=3) as f:
                    f.write(text)
            else:
                with open(filepath, 'w', encoding='utf-8', buffering=8192) as f:  # Buffer de 8KB
                    f.write(text)
            
            logger.info(f"Texto guardado en: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error guardando texto: {e}")
            # Fallback a archivo temporal
            temp_path = os.path.join(self.temp_dir, f"{pdf_id}_temp.txt")
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return temp_path
    
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
    def _split_text_by_pages(self, text: str, total_pages: int) -> List[str]:
        """Divide el texto por páginas"""
        pages_text = []
        
        for page_num in range(1, total_pages + 1):
            page_text = self._extract_page_text(text, page_num)
            pages_text.append(page_text)
        
        return pages_text