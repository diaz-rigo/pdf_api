# C:\Angular\OCR\pdf_api\app\services\ocr_service.py

import pytesseract
from PIL import Image
import io
import concurrent.futures
import logging
from typing import List, Tuple, Optional
import numpy as np
from pdf2image import convert_from_bytes

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, tesseract_cmd: str):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Configuración optimizada de Tesseract
        self.tesseract_config = {
            'lang': 'spa+eng',  # Español + Inglés como fallback
            'oem': 3,  # OEM_LSTM_ONLY - más rápido y preciso
            'psm': 3,  # PSM_AUTO - automático
            'dpi': 300,  # DPI óptimo para balance velocidad/calidad
        }
        
        self.verify_tessdata()
    
    def verify_tessdata(self):
        """Verifica que los archivos de idioma estén disponibles"""
        try:
            langs = pytesseract.get_languages()
            logger.info(f"Idiomas Tesseract disponibles: {langs}")
            
            if 'spa' not in langs:
                logger.warning("Idioma 'spa' no encontrado, usando inglés como fallback")
        except Exception as e:
            logger.error(f"Error verificando idiomas Tesseract: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocesa la imagen para mejorar OCR"""
        # Convertir a escala de grises si es necesario
        if image.mode != 'L':
            image = image.convert('L')
        
        # Mejorar contraste (opcional, solo si necesario)
        # from PIL import ImageEnhance
        # enhancer = ImageEnhance.Contrast(image)
        # image = enhancer.enhance(1.5)
        
        return image
    
    def extract_text_from_image(self, image_bytes: bytes, language: str = "spa") -> str:
        """Extrae texto de una imagen individual con manejo de errores"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocesar imagen
            image = self._preprocess_image(image)
            
            # Configurar parámetros para esta extracción
            config = self.tesseract_config.copy()
            config['lang'] = language if language else 'spa+eng'
            
            # Extraer texto
            text = pytesseract.image_to_string(
                image,
                config=f'--oem {config["oem"]} --psm {config["psm"]}'
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error en OCR para imagen: {e}")
            return ""
    
    def process_page_batch(self, pages: List, start_page: int, language: str = "spa") -> List[Tuple[int, str]]:
        """Procesa un lote de páginas en paralelo"""
        results = []
        
        # Preparar tareas
        tasks = []
        for i, page in enumerate(pages):
            page_num = start_page + i
            image_bytes = page.tobytes("png")
            tasks.append((page_num, image_bytes))
        
        # Procesar en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_page = {
                executor.submit(self.extract_text_from_image, img_bytes, language): page_num
                for page_num, img_bytes in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    text = future.result(timeout=60)  # Timeout por página
                    results.append((page_num, text))
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout en página {page_num}")
                    results.append((page_num, ""))
                except Exception as e:
                    logger.error(f"Error procesando página {page_num}: {e}")
                    results.append((page_num, ""))
        
        return results
    
    def extract_text_from_pdf_images(self, pdf_bytes: bytes, language: str = "spa", 
                                     batch_size: int = 10) -> str:
        """
        Extrae texto de PDF usando OCR con procesamiento por lotes
        
        Args:
            pdf_bytes: Bytes del PDF
            language: Idioma para OCR
            batch_size: Número de páginas por lote (optimizado para memoria)
        """
        try:
            # Convertir PDF a imágenes con configuración optimizada
            # DPI más bajo para documentos de texto, más alto para escaneados complejos
            dpi = 200  # Balance entre calidad y velocidad
            
            # Convertir en lotes para reducir uso de memoria
            all_text = []
            page_offset = 0
            
            # Determinar número total de páginas sin cargar todas las imágenes
            from pdf2image.pdf2image import pdfinfo_from_bytes
            info = pdfinfo_from_bytes(pdf_bytes)
            total_pages = info["Pages"]
            
            logger.info(f"Procesando PDF de {total_pages} páginas en lotes de {batch_size}")
            
            # Procesar por lotes
            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                
                logger.info(f"Procesando páginas {start_page + 1} a {end_page}")
                
                # Convertir lote actual
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    first_page=start_page + 1,
                    last_page=end_page,
                    thread_count=2,  # Reducir para evitar sobrecarga
                    use_pdftocairo=True,  # Usar pdftocairo que es más rápido
                    fmt='png',
                    grayscale=True  # Convertir a escala de grises directamente
                )
                
                # Procesar lote
                batch_results = self.process_page_batch(images, start_page, language)
                
                # Ordenar y agregar resultados
                batch_results.sort(key=lambda x: x[0])
                for page_num, page_text in batch_results:
                    all_text.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
                
                # Liberar memoria
                del images
                
            return "\n".join(all_text)
            
        except Exception as e:
            logger.error(f"Error procesando PDF con OCR: {e}")
            return ""
    
    def extract_text_with_fallback(self, pdf_bytes: bytes, language: str = "spa") -> Tuple[str, bool]:
        """
        Extrae texto primero con PyMuPDF y usa OCR solo si es necesario
        """
        try:
            import fitz
            
            # Abrir PDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            used_ocr = False
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Intentar extraer texto nativo primero
                page_text = page.get_text()
                
                # Si hay poco texto o está vacío, usar OCR
                if len(page_text.strip()) < 50:  # Umbral configurable
                    logger.info(f"Usando OCR en página {page_num + 1}")
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    page_text = self.extract_text_from_image(img_bytes, language)
                    used_ocr = True
                
                text_parts.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
            
            doc.close()
            return "\n".join(text_parts), used_ocr
            
        except Exception as e:
            logger.error(f"Error en extracción con fallback: {e}")
            return "", False