import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from src.main import app
from src.db.models import User, OCRImage, UserPattern
import uuid

client = TestClient(app)

# E2E Test covering Phase 1 (Weeks 1-4)
# Flow: Upload -> Baseline OCR -> Calibration -> Personalized OCR

@pytest.fixture
def mock_full_deps():
    with patch("src.api.endpoints.ocr.get_current_user", new_callable=AsyncMock) as mock_user_ocr, \
         patch("src.api.endpoints.calibration.get_current_user", new_callable=AsyncMock) as mock_user_cal, \
         patch("src.api.endpoints.ocr.get_db", new_callable=AsyncMock) as mock_db_ocr, \
         patch("src.api.endpoints.calibration.get_db", new_callable=AsyncMock) as mock_db_cal, \
         patch("src.api.endpoints.ocr.save_upload_file_temp", new_callable=AsyncMock) as mock_save, \
         patch("src.api.endpoints.ocr.cleanup_temp_file") as mock_cleanup, \
         patch("src.services.gemini.gemini_service.transcribe_image") as mock_transcribe, \
         patch("src.services.gemini.gemini_service.analyze_calibration") as mock_analyze:

        yield {
            "user_ocr": mock_user_ocr,
            "user_cal": mock_user_cal,
            "db_ocr": mock_db_ocr,
            "db_cal": mock_db_cal,
            "save": mock_save,
            "transcribe": mock_transcribe,
            "analyze": mock_analyze
        }

def test_phase1_e2e_flow(mock_full_deps):
    # 1. Setup User and DB Mocks
    user_id = uuid.uuid4()
    mock_user = User(id=user_id, email="e2e@example.com")

    # Mock User Dependency (for both endpoints)
    # Using side_effect to return the mock user for any call
    mock_full_deps["user_ocr"].return_value = mock_user
    mock_full_deps["user_cal"].return_value = mock_user

    # Mock DB (Shared state)
    mock_db = AsyncMock()
    mock_db.add = MagicMock() # add is sync

    # Fix: Ensure execute returns a MagicMock (sync result), not AsyncMock
    mock_result = MagicMock()
    mock_db.execute.return_value = mock_result

    # Simulate db.refresh setting the ID
    def mock_refresh_side_effect(instance):
        if hasattr(instance, 'id') and instance.id is None:
            instance.id = uuid.uuid4()

    mock_db.refresh.side_effect = mock_refresh_side_effect

    mock_full_deps["db_ocr"].return_value = mock_db
    mock_full_deps["db_cal"].return_value = mock_db

    # 2. Step 1: Baseline OCR Upload
    mock_full_deps["save"].return_value = "/tmp/e2e.jpg"
    mock_full_deps["transcribe"].return_value = "Thquick brown fox" # Simulating error

    # Mock empty patterns initially
    mock_result.scalars.return_value.first.return_value = None

    files = {"file": ("e2e.jpg", b"image", "image/jpeg")}

    # Must override dependencies since app resolves them at runtime
    app.dependency_overrides = {} # Reset
    # We patch the import locations in the fixture, but FastAPI needs the overrides map if we want to bypass the actual dependency logic
    # However, since we patched the 'get_current_user' function itself in the fixture (using new_callable=AsyncMock),
    # and that function is what Depends() uses, the patch *should* work if the import path matches.
    # But Depends(get_current_user) uses the function object.

    # Let's enforce overrides for safety and clarity
    async def override_get_user():
        return mock_user
    async def override_get_db():
        return mock_db

    from src.api.deps import get_current_user
    from src.db.session import get_db
    app.dependency_overrides[get_current_user] = override_get_user
    app.dependency_overrides[get_db] = override_get_db

    # --- ACTION: Upload Image ---
    resp_ocr = client.post("/api/ocr/upload", files=files, headers={"X-User-ID": str(user_id)})
    if resp_ocr.status_code != 200:
        print(f"OCR Upload Failed: {resp_ocr.json()}")
    assert resp_ocr.status_code == 200
    data_ocr = resp_ocr.json()
    image_id = data_ocr["id"]
    assert data_ocr["text"] == "Thquick brown fox"
    assert data_ocr["personalized"] is False

    # 3. Step 2: Calibration
    # Prepare mock data for calibration lookup
    mock_ocr_image = OCRImage(id=uuid.UUID(image_id), user_id=user_id, ocr_text="Thquick brown fox")

    # DB Mock needs to return the image when queried
    # We need to configure the mock_result object we created earlier

    # Let's re-configure the mock for the calibration call specifically
    # We can't use reset_mock() on execute easily if we want to keep the return_value object identity
    # but we can reset the side_effect on the inner mock

    mock_result.scalars.return_value.first.side_effect = [
        mock_ocr_image, # 1. Lookup OCRImage
        None # 2. Lookup UserPattern (not found)
    ]

    mock_full_deps["analyze"].return_value = {
        "accuracy": 80.0,
        "confusion_matrix": {"Thquick->The quick": 1},
        "problem_chars": ["h", "e"]
    }

    # --- ACTION: Submit Calibration ---
    payload = {
        "image_id": image_id,
        "ground_truth": "The quick brown fox"
    }
    resp_cal = client.post("/api/calibration/submit", json=payload)
    assert resp_cal.status_code == 200
    data_cal = resp_cal.json()
    assert data_cal["accuracy"] == 80.0

    # 4. Step 3: Personalized OCR
    # Now user has patterns.
    mock_pattern = UserPattern(
        user_id=user_id,
        current_accuracy=80.0,
        confusion_matrix={"Thquick->The quick": 1}
    )

    # Mock DB to return this pattern now
    # Reset side_effect and set return_value
    mock_result.scalars.return_value.first.side_effect = None
    mock_result.scalars.return_value.first.return_value = mock_pattern

    # --- ACTION: Upload Image Again ---
    # We expect the service to be called with user_patterns
    mock_full_deps["transcribe"].reset_mock()
    mock_full_deps["transcribe"].return_value = "The quick brown fox" # Improved result

    resp_ocr_2 = client.post("/api/ocr/upload", files=files, headers={"X-User-ID": str(user_id)})

    assert resp_ocr_2.status_code == 200
    data_ocr_2 = resp_ocr_2.json()
    assert data_ocr_2["personalized"] is True

    # Verify transcribe was called with patterns
    args, kwargs = mock_full_deps["transcribe"].call_args
    assert "Thquick->The quick" in kwargs["user_patterns"] or "Thquick->The quick" in str(kwargs)

    # Clean up
    app.dependency_overrides = {}
