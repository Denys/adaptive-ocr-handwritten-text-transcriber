# Verification & Installation Guide

**Version**: 1.0 (Phase 1 Complete)
**Date**: February 2026

---

## 1. Installation

### Prerequisites
*   Docker & Docker Compose
*   Gemini API Key (with access to `gemini-2.5-flash-001` and `gemini-3-flash-preview`)
*   Python 3.11+ (optional, for local debugging)

### Setup Steps
1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd ocr-system
    ```

2.  **Environment Configuration**:
    Copy `.env.example` to `.env` and set your API key:
    ```bash
    cp .env.example .env
    # Edit .env: GEMINI_API_KEY=your_key_here
    ```

3.  **Launch with Docker**:
    ```bash
    docker-compose up --build -d
    ```
    *   API will be available at `http://localhost:8000`
    *   PostgreSQL database will be running on port 5432

---

## 2. Verification Steps (Manual Testing)

You can verify the system functionality using `curl` or Postman.

### Step 1: Baseline OCR (Week 2 Feature)
Upload a handwritten image to get a basic transcription.

```bash
curl -X POST "http://localhost:8000/api/ocr/upload" \
  -H "X-User-ID: test-user-123" \
  -F "file=@/path/to/handwritten_note.jpg"
```

**Expected Output**:
```json
{
  "id": "uuid-string",
  "text": "Transcribed text...",
  "model": "gemini-2.5-flash-001",
  "personalized": false
}
```

### Step 2: Calibration (Week 3 Feature)
If the OCR output had errors, submit the correct text (Ground Truth).

1.  **Get Pangrams** (Optional, to see what to write):
    ```bash
    curl "http://localhost:8000/api/calibration/start?language=en"
    ```

2.  **Submit Correction**:
    Take the `id` from Step 1 and the *correct* text.
    ```bash
    curl -X POST "http://localhost:8000/api/calibration/submit" \
      -H "X-User-ID: test-user-123" \
      -H "Content-Type: application/json" \
      -d '{
        "image_id": "uuid-from-step-1",
        "ground_truth": "The actual text written on the note."
      }'
    ```

**Expected Output**:
```json
{
  "status": "success",
  "accuracy": 85.5,
  "learned_patterns": {"ocr_error": 1, ...}
}
```

### Step 3: Personalized OCR (Week 4 Feature)
Upload another image (or the same one) for the same user. The system should now use the learned patterns.

```bash
curl -X POST "http://localhost:8000/api/ocr/upload" \
  -H "X-User-ID: test-user-123" \
  -F "file=@/path/to/handwritten_note.jpg"
```

**Expected Output**:
```json
{
  "id": "new-uuid",
  "text": "Better transcribed text...",
  "model": "gemini-2.5-flash-001",
  "personalized": true
}
```
*Note: If accuracy dropped below 85% in Step 2, the `model` field might switch to `gemini-3-flash-preview`.*

---

## 3. Edge Cases

### 1. Blurry / Low Quality Images
*   **Behavior**: Gemini 2.5 Flash acts as baseline. If confidence is low (which we infer from accuracy history), the system might struggle.
*   **Verification**: Upload a blurred image. Expect potential errors in transcription. Submit calibration to help the system learn (though blurring is hard to "learn" away, context helps).

### 2. Invalid Files
*   **Input**: Upload a `.txt` file instead of image.
*   **Behavior**: API returns `400 Bad Request`.
*   **Verification**:
    ```bash
    curl -X POST "http://localhost:8000/api/ocr/upload" \
      -H "X-User-ID: test-user-123" \
      -F "file=@readme.txt"
    ```

### 3. Empty User ID
*   **Input**: Omit `X-User-ID` header.
*   **Behavior**: API returns `422 Unprocessable Entity` (FastAPI validation).

### 4. Non-Existent Image ID in Calibration
*   **Input**: Submit calibration with random UUID.
*   **Behavior**: API returns `404 Not Found` because the image record doesn't exist.
