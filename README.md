# Chest-Xpert Backend API

API REST de inferencia para clasificación de radiografías de tórax. Construida con **FastAPI** y **ONNX Runtime**, clasifica simultáneamente 5 patologías torácicas: Cardiomegaly, Edema, Consolidation, Atelectasis y Pleural Effusion.

---

## Arquitectura

```
[Cliente Angular] ──( POST /predict )──► [FastAPI] ──► [ONNX Runtime] ──► JSON Response
```

El backend utiliza un modelo DenseNet121 exportado a formato ONNX para inferencia eficiente (~20ms por predicción en CPU), reemplazando la dependencia de TensorFlow (~2GB) por ONNX Runtime (~50MB).

---

## Estructura del Proyecto

```
chest-xpert-backend/
├── pyproject.toml              # Dependencias y metadata (PEP 621, UV)
├── uv.lock                     # Versiones fijadas (builds reproducibles)
├── .env                        # Configuración de entorno (no versionado)
├── .env.example                # Plantilla de configuración
├── Dockerfile                  # Imagen Docker multi-stage
├── .dockerignore               # Exclusiones para Docker build
├── .gitignore                  # Exclusiones de Git
├── app/
│   ├── __init__.py
│   ├── main.py                 # App factory + lifespan (ciclo de vida del modelo)
│   ├── config.py               # Settings (pydantic-settings + .env)
│   ├── dependencies.py         # Proveedores FastAPI Depends()
│   ├── errors.py               # Manejo de errores estructurado
│   ├── routers/
│   │   ├── health.py           # GET /health
│   │   └── predict.py          # POST /predict
│   ├── schemas/
│   │   └── prediction.py       # Modelos Pydantic de request/response
│   └── services/
│       ├── inference.py        # Gestión de sesión ONNX Runtime
│       ├── preprocessing.py    # Imagen → tensor (1,1,224,224)
│       └── filter.py           # Filtro de seguridad RGB-Diff
├── models/
│   ├── chest-xpert-model.onnx
│   └── chest-xpert-model.onnx.data
├── scripts/
│   └── download_test_images.py # Descarga imágenes de prueba NIH ChestX-ray14
└── tests/
    ├── conftest.py             # Fixtures compartidos
    ├── test_preprocessing.py   # Tests unitarios de preprocesamiento
    └── images/                 # Imágenes de prueba descargadas (por patología)
```

---

## Requisitos Previos

- **Python 3.14** o superior
- **UV** — gestor de paquetes ([guía de instalación](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker** (opcional, para despliegue containerizado)

---

## Inicio Rápido

```bash
# 1. Instalar dependencias (crea .venv automáticamente)
uv sync

# 2. Copiar y configurar variables de entorno
cp .env.example .env
# Editar .env con la ruta al modelo y configuración deseada

# 3. Iniciar el servidor
uv run uvicorn app.main:app
```

La API estará disponible en `http://localhost:8000`.

### Modo desarrollo (hot-reload)

```bash
uv run uvicorn app.main:app --reload
```

---

## Configuración

Todas las configuraciones se cargan desde el archivo `.env` y/o variables de entorno. Las variables de entorno tienen prioridad sobre los valores del `.env`.

```bash
cp .env.example .env
```

### Configuración de la Aplicación

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `CHESTXPERT_MODEL_PATH` | str | `models/chest-xpert-model.onnx` | Ruta al archivo del modelo ONNX |
| `CHESTXPERT_CORS_ORIGINS` | list[str] | `["http://localhost:4200"]` | Orígenes CORS permitidos |
| `CHESTXPERT_SERVER_PORT` | int | `8000` | Puerto del servidor (1024–65535) |
| `CHESTXPERT_RGB_DIFF_THRESHOLD` | float | `5.0` | Umbral del filtro RGB-Diff (0.0–255.0) |
| `CHESTXPERT_TARGET_CLASSES` | list[str] | 5 patologías | Nombres de las clases objetivo |

### Configuración de Kaggle (para script de descarga de imágenes)

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `KAGGLE_USERNAME` | str | — | Usuario de Kaggle |
| `KAGGLE_KEY` | str | — | API key de Kaggle |
| `KAGGLE_IMAGES_PER_CLASS` | int | `10` | Imágenes a descargar por patología |

---

## Modelo ONNX

El archivo del modelo debe ser accesible en la ruta especificada por `CHESTXPERT_MODEL_PATH`. Para desarrollo local con el repositorio hermano:

```env
CHESTXPERT_MODEL_PATH=../chest-xpert-ai/models/chest-xpert-model.onnx
```

O copiar los archivos del modelo al backend:

```bash
mkdir models
cp ../chest-xpert-ai/models/chest-xpert-model.onnx models/
cp ../chest-xpert-ai/models/chest-xpert-model.onnx.data models/
```

**Especificaciones del modelo:**
- Entrada: tensor `float32`, forma `(1, 1, 224, 224)`, valores de píxel `[0, 255]`
- Salida: 5 probabilidades para `[Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion]`
- No requiere normalización externa — el modelo la maneja internamente

---

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Verificación de estado del servicio |
| POST | `/predict` | Clasificar radiografía de tórax (multipart/form-data, campo `file`) |
| GET | `/docs` | Documentación interactiva Swagger UI |
| GET | `/redoc` | Documentación ReDoc |

### POST /predict

**Request:** multipart/form-data con campo `file` (JPEG o PNG, máximo 10MB)

**Respuesta exitosa (200):**
```json
{
  "success": true,
  "predictions": [
    {"pathology": "Cardiomegaly", "probability": 0.12},
    {"pathology": "Edema", "probability": 0.03},
    {"pathology": "Consolidation", "probability": 0.08},
    {"pathology": "Atelectasis", "probability": 0.45},
    {"pathology": "Pleural Effusion", "probability": 0.67}
  ]
}
```

**Rechazo por filtro (200):**
```json
{
  "success": false,
  "error": "SAMPLE REJECTED: Chromatic artifacts detected..."
}
```

**Respuestas de error:** 400 (imagen inválida), 413 (archivo muy grande), 422 (archivo faltante), 500 (error interno)

---

## Tests

```bash
# Instalar dependencias de desarrollo
uv sync --extra dev

# Ejecutar todos los tests
uv run pytest tests/ -v
```

---

## Imágenes de Prueba

Descarga radiografías reales del dataset NIH ChestX-ray14 para pruebas:

```bash
# 1. Configurar credenciales de Kaggle en .env
#    KAGGLE_USERNAME=tu_usuario
#    KAGGLE_KEY=tu_api_key

# 2. Instalar dependencias de desarrollo
uv sync --extra dev

# 3. Ejecutar el script de descarga
uv run python scripts/download_test_images.py
```

Esto descarga 10 imágenes por clase de patología en `tests/images/`.

---

## Docker

### Build

```bash
docker build -t chest-xpert-backend .
```

### Ejecución

```bash
docker run -d \
  --name chest-xpert-api \
  -p 8000:8000 \
  -v ./models:/app/models:ro \
  --env-file .env \
  chest-xpert-backend
```

### Docker Compose

```yaml
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
```

---

## Stack Tecnológico

| Componente | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.14 | Runtime |
| FastAPI | ≥0.136 | Framework web async |
| ONNX Runtime | ≥1.25 | Motor de inferencia (reemplaza TensorFlow) |
| UV | latest | Gestor de paquetes (reemplaza pip) |
| Pydantic Settings | ≥2.14 | Configuración tipada desde .env + env vars |
| NumPy | ≥2.3 | Manipulación de tensores |
| Pillow | ≥12.2 | Procesamiento de imágenes |
| Hypothesis | ≥6.130 | Property-based testing (dev) |
