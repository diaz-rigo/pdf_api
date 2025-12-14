from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import pdf_router
from app.core.config import settings
import uvicorn

app = FastAPI(
    title="PDF Processing API",
    description="API para extraer texto y buscar en PDFs",
    version="1.0.0"
)

# Configurar CORS para Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(pdf_router.router)

@app.get("/")
async def root():
    return {
        "message": "PDF Processing API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/pdf/upload",
            "search": "/api/pdf/{pdf_id}/search",
            "text": "/api/pdf/{pdf_id}/text",
            "info": "/api/pdf/{pdf_id}/info",
            "list": "/api/pdf/list",
            "quick_search": "/api/pdf/quick-search"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pdf-api"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )