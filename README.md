# Adaptive Multi-Language Handwritten OCR System

**Status**: Phase 1 Complete (v1 MVP)
**Documentation**: [Architecture](ARCHITECTURE.md) | [Roadmap](CLAUDE.md) | [Cost Analysis](COST_ANALYSIS.md) | [Verification](VERIFICATION.md)

A production-ready, continuously learning OCR system designed for mixed-language handwriting (English, Italian, Russian). Built on **Google Gemini 3**, **FastAPI**, and **PostgreSQL**.

---

## 🚀 Quick Start

For detailed installation and manual testing instructions, see **[VERIFICATION.md](VERIFICATION.md)**.

### Prerequisites

1.  **Google Cloud Project**: Enabled with Gemini API.
2.  **API Key**: Obtain a `GEMINI_API_KEY` from AI Studio or Vertex AI.
3.  **Docker**: Installed and running.

### One-Minute Setup

```bash
# 1. Clone
git clone <repo-url>
cd ocr-system

# 2. Configure
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here

# 3. Launch
docker-compose up --build -d
```

Access the API documentation at `http://localhost:8000/docs`.

---

## ✨ Features (Phase 1 Implemented)

### 1. Adaptive OCR
*   **Model Switching**: Automatically selects between **Gemini 2.5 Flash** (Baseline) and **Gemini 3 Flash** (Reasoning) based on accuracy requirements.
*   **Async Processing**: Uses "Store-then-Upload" pattern for efficient file handling.

### 2. Calibration System
*   **Active Learning**: Users can submit corrections (Ground Truth) for OCR outputs.
*   **Pattern Recognition**: The system analyzes errors to build a per-user "Confusion Matrix" (e.g., "User often writes 'e' looking like 'c'").

### 3. Personalization Engine
*   **Prompt Injection**: Learned patterns are injected into the context of future OCR requests for that specific user.
*   **Accuracy Tracking**: Tracks baseline vs. current accuracy to measure improvement over time.

---

## 🏗 System Architecture

*   **Backend**: FastAPI (Async)
*   **LLM**: Gemini 2.5 Flash / Gemini 3
*   **Database**: PostgreSQL (User patterns, Session history)
*   **State Management**: JSONB storage for Gemini "Thought Signatures"

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

---

## 📅 Roadmap (Summary)

*   ✅ **Phase 1 (Weeks 1-4)**: Infrastructure, v1 Endpoint, Calibration, Personalization.
*   ⬜ **Phase 2 (Weeks 5-6)**: Summarization & Concept Explanation.
*   ⬜ **Phase 3 (Weeks 7-10)**: Layout Analysis & HTML/DOCX Export.
*   ⬜ **Phase 4 (Weeks 11-13)**: Diagram Classification & Generation.

See [CLAUDE.md](CLAUDE.md) for the full 14-week plan.

---

## 💰 Cost Model

Designed for affordability:
*   **v1 Cost**: ~$0.0006 per image (Gemini 2.5 Flash)
*   **Monthly Estimate (50 users)**: ~$34/month total
*   **Sustainable Model**: BYOK (Bring Your Own Key) recommended for scale.

See [COST_ANALYSIS.md](COST_ANALYSIS.md) for details.
