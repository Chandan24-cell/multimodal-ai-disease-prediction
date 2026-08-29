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

## Getting Started
Please refer to the `docker-compose.yml` and `.env.example` for setup instructions.