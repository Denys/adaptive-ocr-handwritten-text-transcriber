# Cost Analysis & Model Selection Strategy
**Date**: February 2026
**Currency**: USD

---

## 1. Gemini API Pricing (January 2026)

| Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) | Context Window | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | 1M | v1 Baseline OCR, Summarization |
| **Gemini 3 Flash** | $0.50 | $3.00 | 1M | v1.5 Concept Explanation, v3 Diagrams |
| **Gemini 3 Pro** | $2.00 | $12.00 | 2M | v2 Layout Analysis (Complex) |
| **Google Search** | N/A | $14.00 (per 1K queries) | N/A | Grounding for Concept Explanation |

> **Note**: Audio input for Gemini 3 is $1.00/1M tokens. Image generation ("Nano Banana Pro") is ~$0.13 per image.

---

## 2. v1 MVP Cost Projection (50 Users)

**Assumptions**:
*   **Users**: 50
*   **Usage**: 20 images/week per user (= 80/month)
*   **Total Images**: 4,000 images/month
*   **Avg Tokens**: 1,000 input (image), 125 output (500 chars)

### Calculation (Using Gemini 2.5 Flash)

*   **Input Cost**: 4,000 images * 1,000 tokens * ($0.30 / 1M) = **$1.20**
*   **Output Cost**: 4,000 images * 125 tokens * ($2.50 / 1M) = **$1.25**
*   **Total API Cost**: **$2.45 / month**

### Infrastructure Cost (GCP Estimate)

*   **Cloud Run (CPU/Mem)**: ~$5.00 (scales to zero)
*   **Cloud Storage**: ~$0.20 (10GB)
*   **Cloud SQL (PostgreSQL)**: ~$20.00 (db-f1-micro)
*   **Load Balancer/CDN**: ~$6.00
*   **Total Infra Cost**: **~$31.20 / month**

### Total Monthly Burn
*   **$33.65 / month** (~$0.67 per user)

---

## 3. Freemium Model Economics

To ensure sustainability, we employ a **BYOK (Bring Your Own Key)** model for power users and a **Premium Subscription** for convenience.

### Tiers

1.  **Free Tier**
    *   **Limit**: 10 images / day
    *   **Model**: Gemini 2.5 Flash only
    *   **Features**: Plain text OCR
    *   **Cost to Us**: Max ~$0.02/user/month (negligible)

2.  **BYOK (Recommended)**
    *   **Limit**: Unlimited
    *   **Model**: User pays Google directly
    *   **Features**: All (v1, v1.5, v2, v3)
    *   **Cost to Us**: $0 (Infrastructure covered by Free/Premium mix)

3.  **Premium ($9.99/month)**
    *   **Limit**: 500 images / month
    *   **Model**: Auto-switching (Flash/Pro based on task)
    *   **Features**: Priority processing, 90-day storage
    *   **Margin Analysis**:
        *   Revenue: $9.99
        *   Est. Cost (500 images mixed): ~$1.50
        *   **Profit**: ~$8.50 / user

---

## 4. Model Selection Logic (Cost Optimization)

We implement `ModelSelector` logic to route traffic to the cheapest adequate model.

*   **Baseline**: Always try **Gemini 2.5 Flash** first.
*   **Escalation**: Only upgrade to **Gemini 3 Flash** if:
    *   User calibration accuracy < 85%.
    *   Task requires specialized reasoning (Medical terms).
*   **Pro Tier**: Only upgrade to **Gemini 3 Pro** if:
    *   Task is "Layout Analysis" (v2).
    *   User is Premium or BYOK.
