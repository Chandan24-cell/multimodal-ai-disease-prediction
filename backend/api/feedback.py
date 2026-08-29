# backend/api/feedback.py
from datetime import datetime

from fastapi import APIRouter, Depends, status

from api.auth import get_current_user
from database.mongodb import db
from database.schemas import FeedbackCreate, FeedbackDB, UserDB

router = APIRouter(prefix="/feedback", tags=["Continuous Learning"])


@router.post("/", response_model=FeedbackDB, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback: FeedbackCreate,
    current_user: UserDB = Depends(get_current_user),
):
    """
    Submit doctor-corrected labels for a specific prediction.
    This data is queued for the nightly continuous learning fine-tuning process.
    """
    feedback_dict = feedback.model_dump()
    feedback_dict["reviewed_by"] = current_user.username
    feedback_dict["reviewed_at"] = datetime.utcnow()
    feedback_dict["is_processed"] = False

    result = await db.db.feedback.insert_one(feedback_dict)
    feedback_dict["_id"] = str(result.inserted_id)

    return FeedbackDB(**feedback_dict)


@router.get("/pending", response_model=list[FeedbackDB])
async def get_pending_feedback(current_user: UserDB = Depends(get_current_user)):
    """
    Get all unprocessed feedback (used by the continuous learning script).
    """
    # In a real system, restrict this to admin/researcher roles.
    pending = await db.db.feedback.find({"is_processed": False}).to_list(length=1000)
    for item in pending:
        item["_id"] = str(item["_id"])
    return pending
