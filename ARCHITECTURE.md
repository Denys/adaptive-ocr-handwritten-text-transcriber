# OCR System Architecture Review & Design

**Version**: 1.0
**Date**: February 2026
**Status**: Architecture Freeze

---

## 1. FastAPI Async Patterns: "Store-then-Upload"

The Gemini API requires file uploads (via `files.upload`) for images, rather than direct byte streaming. To handle this efficiently in a high-concurrency FastAPI environment, we must use **asynchronous non-blocking I/O** for the temporary file storage.

### Design Pattern

1.  **Receive Request**: Client uploads file (`UploadFile`).
2.  **Chunked Write**: Stream file content in 1MB chunks to a temporary file on disk using `aiofiles`. This prevents blocking the event loop and keeps memory usage low.
3.  **Gemini Upload**: Use the `google-genai` SDK to upload the temp file path.
4.  **Process**: Call `models.generate_content`.
5.  **Cleanup**: Use FastAPI `BackgroundTasks` to delete the local temp file *after* the response is sent (or if an error occurs).

### Implementation Specification

```python
import os
import tempfile
import aiofiles
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from google import genai
from google.genai import types

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

async def save_upload_file_temp(upload_file: UploadFile) -> str:
    """
    Saves upload file to a temporary file in 1MB chunks using aiofiles.
    Returns the path to the temporary file.
    """
    try:
        # Create a temp file (not auto-deleted on close, so we can pass path to SDK)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name

        # Write asynchronously
        async with aiofiles.open(temp_path, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)

        return temp_path
    except Exception as e:
        # Clean up if write fails
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

def cleanup_temp_file(path: str):
    """Background task to remove temp file."""
    if os.path.exists(path):
        os.remove(path)

@app.post("/api/ocr/upload")
async def ocr_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # 1. Store locally (Async)
    temp_path = await save_upload_file_temp(file)

    # Ensure cleanup runs after response
    background_tasks.add_task(cleanup_temp_file, temp_path)

    try:
        # 2. Upload to Gemini
        gemini_file = client.files.upload(file=temp_path)

        # 3. Process
        response = client.models.generate_content(
            model="gemini-2.5-flash-001", # MVP Selection
            contents=[
                types.Part.from_uri(file_uri=gemini_file.uri, mime_type="image/jpeg"),
                types.Part.from_text("Transcribe this handwritten text.")
            ]
        )

        return {"text": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 2. Thought Signature Handling Strategy

**Challenge**: Gemini 3 models (especially with `thinking_level="high"`) produce "Thought Signatures" that represent the model's internal reasoning state. For multi-turn conversations (like our **Calibration System**), these signatures *must* be passed back in subsequent requests to maintain context and avoid performance degradation or errors.

**Context**: Our API is RESTful (stateless). We cannot hold a `ChatSession` object in memory indefinitely for every user.

### Strategy: Serialized State Storage

We will treat the Gemini conversation history (including Thought Signatures) as session state that must be persisted to the database between HTTP requests.

1.  **Storage**: Use a `JSONB` column in PostgreSQL to store the full serialized list of `Content` objects (which contain the `thought_signature`).
2.  **Serialization**: The `google-genai` SDK objects (like `Content`, `Part`) can be converted to dictionaries.
3.  **Flow**:
    *   **Request 1 (Baseline OCR)**:
        *   User uploads image.
        *   System calls Gemini.
        *   System saves `[UserContent, ModelResponse]` to DB (`ocr_sessions` table).
        *   System returns OCR text + `session_id` to client.
    *   **Request 2 (Calibration/Correction)**:
        *   Client sends `session_id` + correction text.
        *   System retrieves history from DB.
        *   System deserializes to `types.Content` objects.
        *   System creates new chat: `chat = client.chats.create(history=restored_history)`.
        *   System sends correction: `chat.send_message(correction)`.
        *   System appends new turn to DB history.

### Implementation Detail

```python
# Pseudo-code for serialization/deserialization

def serialize_history(history: list[types.Content]) -> list[dict]:
    return [content.to_dict() for content in history] # SDK supports to_dict()

def deserialize_history(data: list[dict]) -> list[types.Content]:
    return [types.Content(**item) for item in data] # Reconstruct from dict
```

**Database Schema Update**:
```sql
CREATE TABLE calibration_sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    history JSONB NOT NULL DEFAULT '[]', -- Stores list of Content dicts
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## 3. Model Selection Logic (The `ModelSelector` Class)

We will implement a dedicated class to encapsulate the logic for choosing the right Gemini model. This allows us to switch strategies (e.g., cost-saving vs. accuracy) without changing business logic.

### Logic Matrix

| Task Type | Condition | Model Selection | Reason |
| :--- | :--- | :--- | :--- |
| **v1 Baseline OCR** | Default | `gemini-2.5-flash-001` | **40% Cheaper** ($0.30/1M), equal OCR performance to 3 Flash. |
| **v1 Personalized** | User Accuracy > 85% | `gemini-2.5-flash-001` | Cost optimized. |
| **v1 Personalized** | User Accuracy < 85% | `gemini-3-flash-preview` | Needs **High Reasoning** to decipher difficult handwriting. |
| **v1.5 Summary** | Default | `gemini-2.5-flash-001` | Summarization is a standard NLP task. |
| **v1.5 Explanation** | Default | `gemini-3-flash-preview` | Requires **reasoning** + `google_search` tool integration. |
| **v2 Layout** | Default | `gemini-3-pro-preview` | Complex structural analysis requires **Pro** reasoning. |
| **v3 Diagram** | Default | `gemini-3-flash-preview` | Balanced speed/reasoning for agentic classification. |

### Code Structure

```python
class ModelSelector:
    MODELS = {
        "MVP_BASELINE": "gemini-2.5-flash-001",
        "REASONING_FLASH": "gemini-3-flash-preview",
        "REASONING_PRO": "gemini-3-pro-preview",
        "IMAGE_GEN": "gemini-3-pro-image-preview"
    }

    @classmethod
    def select(cls, task: str, user_metrics: dict = None) -> str:
        if task == "ocr_baseline":
            return cls.MODELS["MVP_BASELINE"]

        if task == "ocr_personalized":
            # Upgrade if accuracy is struggling
            if user_metrics and user_metrics.get("current_accuracy", 100) < 85.0:
                return cls.MODELS["REASONING_FLASH"]
            return cls.MODELS["MVP_BASELINE"]

        if task == "concept_explanation":
             return cls.MODELS["REASONING_FLASH"]

        if task == "layout_analysis":
            return cls.MODELS["REASONING_PRO"]

        return cls.MODELS["MVP_BASELINE"]
```

---

## 4. Upgrade Criteria

Defines strict thresholds for moving users/tasks to more expensive models.

1.  **Accuracy-Based Upgrade (Automatic)**
    *   **Trigger**: Last 3 calibration sessions show average accuracy < 85%.
    *   **Action**: Switch user's default OCR model from `2.5-flash` to `3-flash`.
    *   **Re-evaluation**: Check again after 10 successful OCRs (user feedback > 4/5 stars).

2.  **Feature-Based Upgrade (User Initiated)**
    *   **Trigger**: User requests "Layout Analysis" (v2) or "Explain Concepts" (v1.5).
    *   **Action**: Use `3-pro` or `3-flash` respectively for that specific request.
    *   **Cost Check**: Verify user has BYOK key or sufficient Premium quota.

3.  **Context-Length Upgrade (System)**
    *   **Trigger**: Input text/image tokens > 200k (Gemini 3 Pro tier jump).
    *   **Action**:
        *   *If Pro needed*: Log warning, proceed (user pays).
        *   *If Flash sufficient*: Fallback to `gemini-2.5-flash` (1M context window is cheaper).

---

## 5. Tool Combination Strategy

Gemini 3 currently has limitations combining `code_execution` and `google_search` in a single turn in some contexts.

**Strategy**: **Sequential Execution**

For tasks requiring both (e.g., "Analyze this error using python, then search for a fix"):
1.  **Step 1**: Call with `tools=[code_execution]` to analyze/calculate.
2.  **Step 2**: Parse result.
3.  **Step 3**: Call with `tools=[google_search]` using insights from Step 1.

For **Calibration**: We only need `code_execution` (Levenshtein distance calc).
For **Concept Explanation**: We only need `google_search`.

**Conclusion**: We will keep tool configurations mutually exclusive per API endpoint to ensure reliability.
