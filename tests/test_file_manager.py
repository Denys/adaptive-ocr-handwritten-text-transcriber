import os
import pytest
from fastapi import UploadFile
from src.services.file_manager import save_upload_file_temp, cleanup_temp_file
import io

class MockUploadFile(UploadFile):
    def __init__(self, content: bytes, filename: str):
        self.file = io.BytesIO(content)
        self.filename = filename
        self.size = len(content)

    async def read(self, size: int = -1):
        return self.file.read(size)

@pytest.mark.asyncio
async def test_save_upload_file_temp():
    content = b"fake image content" * 1000  # Make it a bit larger
    mock_file = MockUploadFile(content, "test.jpg")

    path = await save_upload_file_temp(mock_file)

    try:
        assert os.path.exists(path)
        with open(path, "rb") as f:
            saved_content = f.read()
        assert saved_content == content
    finally:
        cleanup_temp_file(path)
        assert not os.path.exists(path)

@pytest.mark.asyncio
async def test_cleanup_temp_file():
    # Create a dummy file
    with open("dummy_temp_file.txt", "w") as f:
        f.write("test")

    path = os.path.abspath("dummy_temp_file.txt")
    assert os.path.exists(path)

    cleanup_temp_file(path)
    assert not os.path.exists(path)
