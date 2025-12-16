# import pytesseract
# from PIL import Image
# import io
# import os
# from pdf2image import convert_from_bytes

# class OCRService:
#     def __init__(self, tesseract_cmd: str):
#         pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
#         # Verificar que el archivo de idioma existe
#         self.verify_tessdata()
    
#     def verify_tessdata(self):
#         """Verifica que los archivos de idioma estén disponibles"""
#         try:
#             # Listar idiomas disponibles
#             langs = pytesseract.get_languages()
#             print(f"Idiomas Tesseract disponibles: {langs}")
            
#             if 'spa' not in langs:
#                 print("Advertencia: spa no encontrado en idiomas disponibles")
#         except Exception as e:
#             print(f"Error verificando idiomas Tesseract: {e}")
    
#     def extract_text_from_image(self, image_bytes: bytes, language: str = "spa") -> str:
#         try:
#             image = Image.open(io.BytesIO(image_bytes))
            
#             # Primero intentar con el idioma especificado
#             try:
#                 text = pytesseract.image_to_string(image, lang=language)
#             except:
#                 # Si falla, intentar sin idioma específico
#                 print(f"Intento fallido con idioma '{language}', intentando sin idioma específico")
#                 text = pytesseract.image_to_string(image)
            
#             return text
#         except Exception as e:
#             print(f"Error en OCR: {e}")
#             return ""
    
#     def extract_text_from_pdf_images(self, pdf_bytes: bytes, language: str = "spa") -> str:
#         try:
#             images = convert_from_bytes(pdf_bytes, dpi=200)
#             text = ""
#             for i, image in enumerate(images):
#                 text += f"\n--- Página {i+1} ---\n"
#                 page_text = self.extract_text_from_image(image.tobytes(), language)
#                 text += page_text
#             return text
#         except Exception as e:
#             print(f"Error procesando PDF con OCR: {e}")
#             return ""