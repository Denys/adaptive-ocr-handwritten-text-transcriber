from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from src.db.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    language_preferences = Column(ARRAY(String), default=["en", "it", "ru"])
    gemini_api_key_encrypted = Column(LargeBinary, nullable=True)
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime(timezone=True), nullable=True)
    tier = Column(String(20), default="free")

    ocr_images = relationship("OCRImage", back_populates="user", cascade="all, delete-orphan")
    patterns = relationship("UserPattern", back_populates="user", uselist=False, cascade="all, delete-orphan")
    calibration_sessions = relationship("CalibrationSession", back_populates="user", cascade="all, delete-orphan")

class UserPattern(Base):
    __tablename__ = "user_patterns"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    calibration_count = Column(Integer, default=0)
    baseline_accuracy = Column(Float, nullable=True)
    current_accuracy = Column(Float, nullable=True)
    confusion_matrix = Column(JSONB, nullable=True)
    problem_chars = Column(ARRAY(Text), nullable=True)
    accuracy_history = Column(JSONB, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="patterns")

class OCRImage(Base):
    __tablename__ = "ocr_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    s3_key = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    ocr_text = Column(Text, nullable=True)
    language_detected = Column(String, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    processed = Column(Boolean, default=False)
    model_used = Column(String(50), nullable=True)

    user = relationship("User", back_populates="ocr_images")

class CalibrationSession(Base):
    __tablename__ = "calibration_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    history = Column(JSONB, nullable=False, server_default='[]')
    status = Column(String(20), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="calibration_sessions")
