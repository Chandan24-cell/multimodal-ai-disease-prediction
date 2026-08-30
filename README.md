# Multimodal AI-Based Intelligent Healthcare System

## ⚠️ MEDICAL DISCLAIMER
**This system is a research and clinical decision-support prototype ONLY.** 
It is NOT a diagnostic replacement for licensed medical professionals. All outputs, predictions, and generated reports are for educational and research purposes and must not be used as definitive medical advice or to make clinical decisions without human expert validation.

## Domain & Project Type
- **Domain:** Artificial Intelligence & Deep Learning
- **Project Type:** Advanced Multimodal Deep Learning System
- **Core Principle:** Deep learning models sit at the center of the disease-analysis and multimodal-fusion pipeline. The LLM is used strictly at the end for report generation, grounded via RAG, and does not make the diagnosis itself.

## System Architecture
1. **Image Modality:** Vision Transformer (ViT) for MRI/X-Ray analysis.
2. **Text Modality:** Clinical Transformer (BERT-based) for symptom analysis.
3. **History Modality:** Neural Network/Transformer over structured medical history.
4. **Multimodal Fusion:** Deep Neural Network combining the three modality embeddings.
5. **Prediction:** Multi-class disease classifier.
6. **Explainability:** Grad-CAM (images), Attention Maps (text), SHAP (tabular/fusion).
7. **RAG & LLM:** Retrieves medical context and generates an AI-assisted clinical report.

## Technology Stack
- **Backend:** Python, FastAPI, Pydantic
- **Frontend:** React (Vite), Charting/Visualization libraries
- **Database:** MongoDB
- **AI/ML:** PyTorch, HuggingFace Transformers, SHAP, Grad-CAM
- **RAG:** LangChain/LlamaIndex, FAISS/ChromaDB, SentenceTransformers
- **Deployment:** Docker, Docker Compose

## Local development

1. Create a local secrets file and replace every placeholder with a unique value:

   ```bash
   cp .env.example .env
   ```

   At minimum, set `MONGO_ROOT_PASSWORD` and `JWT_SECRET_KEY` to strong random
   values. Never commit `.env`.

2. Start the development stack:

   ```bash
   docker compose up --build
   ```

   The development configuration binds the source directories and enables
   Uvicorn reload. MongoDB is available only on `127.0.0.1:27017` and requires
   the credentials from `.env`.

## Production Docker Compose

Use the production override to run the backend from the built image without
source bind mounts or autoreload:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

The backend runs with four Uvicorn workers in this configuration. Terminate TLS
at a reverse proxy or hosting platform and use TLS for every external MongoDB
connection.

## Testing

The repository currently contains executable backend smoke-test scripts:

```bash
PYTHONPATH=backend python backend/test_vit.py
PYTHONPATH=backend python backend/test_history.py
LLM_PROVIDER=mock PYTHONPATH=backend python backend/test_llm.py
```

The RAG, clinical-text, pipeline, and explainability scripts require their
Hugging Face model assets to be available. Download and pin those assets before
running them in an offline or production environment.

## Render deployment

Create one Python web service from the repository root. Use:

```text
Build command: cd backend && pip install -r requirements.txt && cd ../frontend && npm ci && npm run build
Start command: uvicorn main:app --app-dir backend --host 0.0.0.0 --port $PORT
Health check: /api/health
```

Set `APP_ENV=production`, `MONGODB_URI`, `MONGODB_DB_NAME`,
`JWT_SECRET_KEY`, `PYTHON_VERSION=3.11.9`, and any selected LLM provider
credentials in the Render dashboard. The frontend calls the same-origin `/api`
path in production, so do not set `VITE_API_URL` for this single-service setup.
