# backend/api/rag.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict

from rag.retriever import medical_retriever
from api.auth import get_current_user

router = APIRouter(prefix="/rag", tags=["RAG"])

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3

@router.post("/query")
async def query_knowledge_base(
    request: RAGQueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Direct query endpoint for the medical knowledge base (RAG).
    Useful for debugging or allowing clinicians to search guidelines manually.
    """
    # We pass an empty list for predicted diseases to just search the raw query
    context = medical_retriever.retrieve_context(
        predicted_diseases=[], 
        symptom_text=request.query, 
        top_k=request.top_k
    )
    return {"results": context}