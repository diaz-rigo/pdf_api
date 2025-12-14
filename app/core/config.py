import os
from pathlib import Path


from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    EXTRACTED_FOLDER = os.getenv("EXTRACTED_FOLDER", "extracted_texts")
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:4200"
    ).split(",")

    @classmethod
    def create_folders(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.EXTRACTED_FOLDER, exist_ok=True)
    def __init__(self):
        # Establecer TESSDATA_PREFIX si no está configurado
        if 'TESSDATA_PREFIX' not in os.environ:
            tessdata_path = Path(self.TESSERACT_CMD).parent / 'tessdata'
            if tessdata_path.exists():
                os.environ['TESSDATA_PREFIX'] = str(tessdata_path)


settings = Settings()
settings.create_folders()
