# 📄 PDF Processing API (OCR & Search)

API profesional desarrollada con **FastAPI** para la **carga múltiple de PDFs**, extracción de texto (incluyendo **OCR para documentos escaneados**) y **búsqueda eficiente** dentro del contenido.

Pensada para integrarse fácilmente con **Angular**, y soportar documentos digitales y escaneados sin perder la cordura en producción.

---

## 🚀 Características

* 📤 Subida de **uno o múltiples PDFs**
* 🔍 Extracción de texto automática:

  * PDFs con texto embebido
  * PDFs escaneados mediante **OCR (Tesseract)**
* 🔎 Búsqueda de texto:

  * Búsqueda normal
  * Búsqueda con **expresiones regulares (regex)**
* 🌍 Detección automática de idioma
* ⚡ API REST rápida y escalable
* 🔐 CORS configurado para frontend (Angular)
* 📁 Persistencia de archivos y texto extraído
* 📚 Documentación automática con Swagger

---

## 🧱 Tecnologías

* **Python 3.11 (recomendado)**
* **FastAPI**
* **Uvicorn**
* **PyMuPDF (fitz)**
* **Tesseract OCR**
* **pdf2image**
* **Pillow**
* **pytesseract**
* **Pydantic v2**

---

## 📋 Requisitos previos

### 🐍 Python
> ⚠️ Python 3.13+ (incluido 3.14) **NO es compatible** actualmente con PyMuPDF en Windows.
> Usar Python 3.11 evita errores de compilación nativa.

Instalar **Python 3.9 o superior**:

👉 [https://www.python.org/downloads/](https://www.python.org/downloads/)

> ⚠️ Durante la instalación, marcar **Add Python to PATH**

Verificar instalación:

```bash
python --version
```

---

### 🔠 Tesseract OCR (OBLIGATORIO)

#### Windows

Descargar desde:

👉 [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Durante la instalación:

* ✔️ Marcar **Add to PATH**
* ✔️ Instalar idiomas:

  * English
  * Spanish

Ruta típica de instalación:

```txt
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Verificar instalación:

```bash
tesseract --version
```

---

## 📂 Estructura del proyecto

```txt
pdf_api/
│── app/            # Código principal de la API
│── uploads/        # PDFs subidos y texto procesado
│── venv/           # Entorno virtual
│── .env            # Variables de entorno
│── requirements.txt
│── run.py          # Punto de entrada de la API
│── README.md
```

---

## ⚙️ Configuración del entorno

### 1️⃣ Crear y activar entorno virtual

```bash
py -3.11 -m venv venv

venv\Scripts\activate
```

---
Esto garantiza que el entorno virtual use **Python 3.11**, incluso si el sistema tiene otra versión por defecto.

### 2️⃣ Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3️⃣ Configurar variables de entorno

Crear o editar el archivo `.env`:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
UPLOAD_DIR=uploads
CORS_ORIGINS=http://localhost:4200
```

---

## ▶️ Ejecutar la API

Desde la raíz del proyecto:

```bash
python run.py
```

La API estará disponible en:

```
http://127.0.0.1:8000
```

---

## 📘 Documentación (Swagger)

FastAPI genera documentación automática:

```
http://127.0.0.1:8000/docs
```

Desde aquí puedes:

* Subir PDFs
* Probar OCR
* Ejecutar búsquedas

---

## 🔌 Integración con Angular

* Usar `FormData` para subir archivos
* Endpoint típico:

```http
POST http://localhost:8000/upload
```

> Tailwind se utiliza únicamente para la UI. Todo el procesamiento pesado ocurre en la API.

---

## 🧪 Recomendaciones de prueba

* Probar con:

  * 📄 PDF digital
  * 📄 PDF escaneado
* Verificar que ambos devuelvan texto
* Probar búsquedas con y sin regex

---

## 🧠 Próximos pasos sugeridos

* 🔍 Indexación avanzada (SQLite / PostgreSQL)
* ⚡ Motor de búsqueda optimizado
* 🐳 Dockerización
* ☁️ Despliegue en servidor o cloud

---

## 🏁 Estado del proyecto

✅ Listo para desarrollo y pruebas locales
🚀 Preparado para escalar a producción

---