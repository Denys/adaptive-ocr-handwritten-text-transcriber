import os
import tempfile
import aiofiles
from fastapi import UploadFile, HTTPException

async def save_upload_file_temp(upload_file: UploadFile) -> str:
    """
    Saves upload file to a temporary file in 1MB chunks using aiofiles.
    Returns the path to the temporary file.
    The caller is responsible for deleting the file after use.
    """
    temp_path = None
    try:
        # Create a temp file (not auto-deleted on close, so we can pass path to SDK)
        # We use delete=False because we need the file to persist until uploaded to Gemini
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name

        # Write asynchronously
        async with aiofiles.open(temp_path, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)

        return temp_path
    except Exception as e:
        # Clean up if write fails
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

def cleanup_temp_file(path: str):
    """Background task to remove temp file."""
    if os.path.exists(path):
        os.remove(path)
