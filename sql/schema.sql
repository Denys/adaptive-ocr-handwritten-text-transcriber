-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    language_preferences TEXT[] DEFAULT ARRAY['en', 'it', 'ru'],
    gemini_api_key_encrypted BYTEA,
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    tier VARCHAR(20) DEFAULT 'free' -- free, byok, premium
);

-- User patterns (JSONB for flexibility)
CREATE TABLE user_patterns (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    calibration_count INTEGER DEFAULT 0,
    baseline_accuracy FLOAT,
    current_accuracy FLOAT,
    confusion_matrix JSONB,  -- {"a→o": 12, "e→c": 8}
    problem_chars TEXT[],
    accuracy_history JSONB,  -- [{"date": "2026-01-01", "accuracy": 72.5}]
    updated_at TIMESTAMP DEFAULT NOW()
);

-- OCR history
CREATE TABLE ocr_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    s3_key TEXT NOT NULL, -- or local path for MVP
    uploaded_at TIMESTAMP DEFAULT NOW(),
    ocr_text TEXT,
    language_detected TEXT,
    tokens_used INTEGER,
    processed BOOLEAN DEFAULT FALSE,
    model_used VARCHAR(50)
);

-- Calibration Sessions (Thought Signature Storage)
CREATE TABLE calibration_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    history JSONB NOT NULL DEFAULT '[]', -- Stores list of Content dicts (with thought signatures)
    status VARCHAR(20) DEFAULT 'active', -- active, completed, abandoned
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_user_images ON ocr_images(user_id, uploaded_at DESC);
CREATE INDEX idx_user_patterns ON user_patterns(user_id);
CREATE INDEX idx_calibration_user ON calibration_sessions(user_id);
