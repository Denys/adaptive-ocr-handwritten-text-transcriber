import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from src.main import app
from src.db.models import User
import uuid
import io

client = TestClient(app)

# Mock dependencies
@pytest.fixture
def mock_deps():
    with patch("src.api.endpoints.ocr.get_current_user", new_callable=AsyncMock) as mock_user, \
         patch("src.api.endpoints.ocr.get_db", new_callable=AsyncMock) as mock_db, \
         patch("src.api.endpoints.ocr.save_upload_file_temp", new_callable=AsyncMock) as mock_save, \
         patch("src.api.endpoints.ocr.cleanup_temp_file") as mock_cleanup, \
         patch("src.api.endpoints.ocr.gemini_service.transcribe_image") as mock_transcribe:

        yield {
            "user": mock_user,
            "db": mock_db,
            "save": mock_save,
            "cleanup": mock_cleanup,
            "transcribe": mock_transcribe
        }

def test_upload_ocr_image_success(mock_deps):
    # Setup mocks
    user_id = uuid.uuid4()
    mock_user = User(id=user_id, email="test@example.com")
    mock_deps["user"].return_value = mock_user

    mock_db_session = AsyncMock()
    mock_deps["db"].return_value = mock_db_session

    mock_deps["save"].return_value = "/tmp/test.jpg"
    mock_deps["transcribe"].return_value = "Detected Text"

    # Create dummy image file
    file_content = b"fake image"
    files = {"file": ("test.jpg", file_content, "image/jpeg")}

    # Override dependency in app (FastAPI style)
    # Actually, patch handles the import path, so this should work without app.dependency_overrides
    # IF the patch target is correct.
    # Note: `src.api.endpoints.ocr.get_current_user` is imported in the endpoint file.
    # However, FastAPI resolves dependencies at runtime.
    # It's safer to override via app.dependency_overrides for the Depends() calls.

    async def override_get_current_user():
        return mock_user

    async def override_get_db():
        return mock_db_session

    app.dependency_overrides["src.api.deps.get_current_user"] = override_get_current_user
    # Note: main.py -> ocr.router -> ocr endpoint imports get_current_user from src.api.deps
    # But the endpoint definition uses `Depends(get_current_user)`.
    # We need to override the function object that `Depends` is holding.
    # So we should override the function imported in `src.api.endpoints.ocr` OR the original function.
    from src.api.deps import get_current_user
    from src.db.session import get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db

    response = client.post(
        "/api/ocr/upload",
        files=files,
        headers={"X-User-ID": str(user_id)}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Detected Text"
    assert data["model"] == "gemini-2.5-flash-001"

    # Verify calls
    mock_deps["save"].assert_called_once()
    mock_deps["transcribe"].assert_called_once_with("/tmp/test.jpg")
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

    # Clean up overrides
    app.dependency_overrides = {}

def test_upload_ocr_image_invalid_type(mock_deps):
    files = {"file": ("test.txt", b"text content", "text/plain")}

    # We need to auth even for bad request
    mock_user = User(id=uuid.uuid4(), email="test@example.com")
    async def override_get_current_user():
        return mock_user

    from src.api.deps import get_current_user
    app.dependency_overrides[get_current_user] = override_get_current_user

    response = client.post(
        "/api/ocr/upload",
        files=files,
        headers={"X-User-ID": str(mock_user.id)}
    )

    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]

    app.dependency_overrides = {}
