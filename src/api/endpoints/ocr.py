from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.api.deps import get_current_user
from src.db.session import get_db
from src.db.models import User, OCRImage, UserPattern
from src.services.file_manager import save_upload_file_temp, cleanup_temp_file
from src.services.gemini import gemini_service
from src.services.selector import ModelSelector
import json

router = APIRouter()

@router.post("/upload")
async def upload_ocr_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload an image for OCR processing.
    Uses the Store-then-Upload pattern to handle large files efficiently.
    Includes Personalization Engine (Week 4).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 1. Save temp file (Async)
    temp_path = await save_upload_file_temp(file)

    # Ensure cleanup runs after response
    background_tasks.add_task(cleanup_temp_file, temp_path)

    try:
        # 2. Fetch User Patterns (Personalization)
        stmt = select(UserPattern).where(UserPattern.user_id == current_user.id)
        result = await db.execute(stmt)
        user_pattern = result.scalars().first()

        # 3. Select Model
        model_name = ModelSelector.select("ocr_personalized", user_pattern)

        # 4. Format Patterns for Prompt Injection
        pattern_str = None
        if user_pattern and user_pattern.confusion_matrix:
            # Sort top confusions
            sorted_confusions = sorted(
                user_pattern.confusion_matrix.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            pattern_str = "\n".join([f"- '{k}': {v} errors" for k, v in sorted_confusions])

        # 5. Transcribe with Gemini
        ocr_text = gemini_service.transcribe_image(
            file_path=temp_path,
            model_name=model_name,
            user_patterns=pattern_str
        )

        # 6. Save Record to DB
        db_image = OCRImage(
            user_id=current_user.id,
            s3_key=f"temp/{file.filename}",
            ocr_text=ocr_text,
            model_used=model_name,
            processed=True,
            tokens_used=0
        )
        db.add(db_image)
        await db.commit()
        await db.refresh(db_image)

        return {
            "id": str(db_image.id),
            "text": ocr_text,
            "model": db_image.model_used,
            "personalized": bool(pattern_str)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Processing failed: {str(e)}")
