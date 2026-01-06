from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_user
from src.db.session import get_db
from src.db.models import User, OCRImage
from src.services.file_manager import save_upload_file_temp, cleanup_temp_file
from src.services.gemini import gemini_service

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
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 1. Save temp file (Async)
    temp_path = await save_upload_file_temp(file)

    # Ensure cleanup runs after response
    background_tasks.add_task(cleanup_temp_file, temp_path)

    try:
        # 2. Transcribe with Gemini (Retry logic inside service)
        # Note: Using default model (Gemini 2.5 Flash) for v1 MVP
        ocr_text = gemini_service.transcribe_image(temp_path)

        # 3. Save Record to DB
        # Note: In a real S3 setup, we would upload to S3 here.
        # For MVP we just log the temp path or a dummy key.
        db_image = OCRImage(
            user_id=current_user.id,
            s3_key=f"temp/{file.filename}",
            ocr_text=ocr_text,
            model_used="gemini-2.5-flash-001",
            processed=True,
            tokens_used=0 # Placeholder, would extract from response usage_metadata if available
        )
        db.add(db_image)
        await db.commit()
        await db.refresh(db_image)

        return {
            "id": str(db_image.id),
            "text": ocr_text,
            "model": db_image.model_used
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Processing failed: {str(e)}")
