from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_user
from src.db.session import get_db
from src.db.models import User, OCRImage
from src.services.calibration import CalibrationService
from src.services.gemini import gemini_service
import uuid

router = APIRouter()

@router.get("/start")
async def start_calibration(
    language: str = "en",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    service = CalibrationService(db)
    pangrams = await service.get_pangrams(language)
    return {"pangrams": pangrams}

@router.post("/submit")
async def submit_calibration(
    image_id: str = Body(..., embed=True),
    ground_truth: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submits a calibration attempt.
    1. Fetches the OCR text from the original image record.
    2. Runs analysis via Gemini.
    3. Updates user patterns.
    """
    try:
        img_uuid = uuid.UUID(image_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Image ID")

    # Fetch OCR Image
    from sqlalchemy import select
    stmt = select(OCRImage).where(OCRImage.id == img_uuid, OCRImage.user_id == current_user.id)
    result = await db.execute(stmt)
    ocr_image = result.scalars().first()

    if not ocr_image or not ocr_image.ocr_text:
        raise HTTPException(status_code=404, detail="OCR Image not found or processed")

    # Analyze
    analysis = gemini_service.analyze_calibration(ground_truth, ocr_image.ocr_text)

    # Update Patterns
    service = CalibrationService(db)
    updated_pattern = await service.process_submission(current_user.id, analysis)

    return {
        "status": "success",
        "accuracy": updated_pattern.current_accuracy,
        "learned_patterns": updated_pattern.confusion_matrix
    }
