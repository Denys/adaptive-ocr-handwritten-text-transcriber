# MultiScript OCR Implementation Plan

**Version**: 1.0
**Date**: February 2026
**Target Phase**: v1 MVP

---

## 1. 14-Week Implementation Roadmap

This roadmap takes us from empty repo to a fully production-ready diagram-recognizing OCR system.

### Phase 1: v1 MVP (Plain Text OCR) - Weeks 1-4

*   **Week 1: Infrastructure & Skeleton**
    *   Initialize FastAPI project structure.
    *   Docker setup (Dockerfile, docker-compose).
    *   Set up PostgreSQL with async driver (`asyncpg`).
    *   Implement `Store-then-Upload` async pattern.
*   **Week 2: Basic OCR Endpoint**
    *   Implement `POST /api/ocr/upload`.
    *   Integrate Gemini 2.5 Flash.
    *   Implement user authentication (simple API key or JWT).
    *   **Milestone**: First successful handwritten text extraction.
*   **Week 3: Calibration System**
    *   Implement `calibration_sessions` table (JSONB for thought signatures).
    *   Create `POST /api/calibration/start` and `POST /api/calibration/correct`.
    *   Implement sequential `code_execution` tool logic for accuracy calculation.
*   **Week 4: Personalization Engine**
    *   Implement `ModelSelector` logic.
    *   Create user profile updates based on calibration.
    *   **Milestone**: v1 MVP Complete (Accuracy > 70% baseline).

### Phase 2: v1.5 (Enrichment) - Weeks 5-6

*   **Week 5: Summarization**
    *   Add `POST /api/summarize`.
    *   Implement language detection (using `langdetect` or Gemini).
*   **Week 6: Concept Explanation**
    *   Add `POST /api/explain`.
    *   Integrate `google_search` tool (handling billing checks for Jan 2026).
    *   **Milestone**: v1.5 Release.

### Phase 3: v2 (Layout & Structure) - Weeks 7-10

*   **Week 7: Layout Analysis Prompting**
    *   Switch to Gemini 3 Pro for layout tasks.
    *   Design prompts for JSON layout extraction.
*   **Week 8: HTML Generation**
    *   Implement JSON-to-HTML converter.
    *   Test with complex menu samples.
*   **Week 9: Confidence Highlighting**
    *   Implement logic to parse uncertain regions.
    *   Frontend CSS for confidence heatmaps.
*   **Week 10: Format Export**
    *   Add DOCX export.
    *   **Milestone**: v2 Release (Rich Text).

### Phase 4: v3 (Diagrams) - Weeks 11-13

*   **Week 11: Diagram Detection**
    *   Integrate OpenCV for region detection.
*   **Week 12: Classification & Code Gen**
    *   Implement Mermaid.js generation prompts.
*   **Week 13: Image Re-generation**
    *   Integrate `gemini-3-pro-image-preview` ("Nano Banana Pro").
    *   **Milestone**: v3 Release (Full Multi-Modal).

### Phase 5: Hardening - Week 14

*   **Week 14: Final Polish**
    *   Load testing (Locust).
    *   Security audit (PII redaction verification).
    *   Documentation finalization.

---

## 2. Code Conventions

### Python / FastAPI
*   **Type Hinting**: Mandatory for all function arguments and returns. Use `typing` module.
*   **Async/Await**: All I/O bound operations (DB, API calls, File I/O) must be async.
*   **Pydantic**: Use Pydantic V2 models for all request/response schemas.
*   **Docstrings**: Google style docstrings for all functions.
*   **Linter**: `ruff` for linting and formatting.

### Gemini Integration
*   **SDK**: Use `google-genai` (latest v1.0+).
*   **Error Handling**: Wrap all API calls in `try/except` blocks handling `ClientError` (4xx) and `ServerError` (5xx).
*   **Retries**: Use `tenacity` for transient errors (rate limits), but *never* retry on 400 (Bad Request).

### Database
*   **Migration**: Use `alembic` for all schema changes.
*   **Naming**: Snake_case for tables and columns.

---

## 3. Step-by-Step Implementation Guide (Immediate Next Steps)

These are the immediate tasks to execute if moving to coding phase now.

1.  **Project Initialization**
    *   Create `requirements.txt` with: `fastapi`, `uvicorn`, `google-genai`, `aiofiles`, `asyncpg`, `sqlalchemy`, `pydantic-settings`.
    *   Create `Dockerfile` (Python 3.11 slim).

2.  **Core Module (`src/core`)**
    *   `config.py`: Load `GEMINI_API_KEY`, `DATABASE_URL` from env.
    *   `database.py`: Async engine setup.

3.  **Gemini Client (`src/services/gemini.py`)**
    *   Initialize `genai.Client`.
    *   Implement `upload_file_async` helper.

4.  **OCR Endpoint (`src/api/endpoints/ocr.py`)**
    *   Implement the `ocr_upload` function using the Architecture Review pattern.

5.  **Model Selector (`src/services/selector.py`)**
    *   Implement `ModelSelector` class.

---

## 4. Testing Strategy

*   **Unit Tests**: `pytest` for individual functions (ModelSelector, logic).
*   **Integration Tests**: Test FastAPI endpoints with mocked Gemini responses.
*   **E2E Tests**: (Optional for now) Real calls to Gemini with test account.
