# Adaptive Multi-Language Handwritten OCR System

**Status**: v1 MVP Planning Phase
**Documentation**: [Architecture](ARCHITECTURE.md) | [Roadmap](CLAUDE.md) | [Cost Analysis](COST_ANALYSIS.md)

A production-ready, continuously learning OCR system designed for mixed-language handwriting (English, Italian, Russian). Built on **Google Gemini 3**, **FastAPI**, and **PostgreSQL**.

---

## 🚀 Quick Start

### Prerequisites

1.  **Google Cloud Project**: Enabled with Gemini API.
2.  **API Key**: Obtain a `GEMINI_API_KEY` from AI Studio or Vertex AI.
3.  **Docker**: Installed and running.
4.  **Python**: 3.11+ (for local development).

### Setup

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd ocr-system
    ```

2.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env and set GEMINI_API_KEY
    ```

3.  **Run with Docker**:
    ```bash
    docker-compose up --build
    ```

4.  **Access API**:
    *   Docs: `http://localhost:8000/docs`
    *   OCR Endpoint: `POST /api/ocr/upload`

---

## 🏗 System Architecture

*   **Backend**: FastAPI (Async)
*   **LLM**: Gemini 2.5 Flash (MVP) -> Gemini 3 Pro (v2)
*   **Database**: PostgreSQL (User patterns, Session history)
*   **Pattern**: "Store-then-Upload" for robust file handling
*   **State Management**: JSONB storage for Gemini "Thought Signatures"

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

---

## 📅 Roadmap (Summary)

*   **Phase 1 (Weeks 1-4)**: Infrastructure, v1 Endpoint, Calibration System.
*   **Phase 2 (Weeks 5-6)**: Summarization & Concept Explanation.
*   **Phase 3 (Weeks 7-10)**: Layout Analysis & HTML/DOCX Export.
*   **Phase 4 (Weeks 11-13)**: Diagram Classification & Generation.

See [CLAUDE.md](CLAUDE.md) for the full 14-week plan.

---

## 💰 Cost Model

Designed for affordability:
*   **v1 Cost**: ~$0.0006 per image (Gemini 2.5 Flash)
*   **Monthly Estimate (50 users)**: ~$34/month total
*   **Sustainable Model**: BYOK (Bring Your Own Key) recommended for scale.

See [COST_ANALYSIS.md](COST_ANALYSIS.md) for details.
