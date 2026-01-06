import json
from src.db.models import User, UserPattern
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

class CalibrationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_pangrams(self, language: str = "en") -> list[str]:
        # Load from JSON file (cached in memory ideally, but file I/O for MVP is fine)
        try:
            with open("data/calibration_pangrams.json", "r") as f:
                data = json.load(f)

            # Map 'en' to 'english', etc.
            lang_map = {"en": "english", "it": "italian", "ru": "russian"}
            key = lang_map.get(language, "english")
            return data.get(key, [])
        except FileNotFoundError:
            return ["The quick brown fox jumps over the lazy dog."]

    async def process_submission(self, user_id: str, analysis_result: dict):
        """
        Updates user patterns based on calibration analysis.
        """
        # Fetch existing patterns
        stmt = select(UserPattern).where(UserPattern.user_id == user_id)
        result = await self.db.execute(stmt)
        pattern = result.scalars().first()

        if not pattern:
            pattern = UserPattern(user_id=user_id, calibration_count=0)
            self.db.add(pattern)

        # Merge confusion matrix
        new_matrix = analysis_result.get("confusion_matrix", {})
        existing_matrix = pattern.confusion_matrix or {}

        # Simple merge: sum frequencies
        for key, value in new_matrix.items():
            existing_matrix[key] = existing_matrix.get(key, 0) + value

        pattern.confusion_matrix = existing_matrix
        pattern.current_accuracy = analysis_result.get("accuracy", 0.0)
        pattern.problem_chars = analysis_result.get("problem_chars", [])

        # Update counts
        pattern.calibration_count += 1
        if pattern.calibration_count == 1:
            pattern.baseline_accuracy = pattern.current_accuracy

        await self.db.commit()
        await self.db.refresh(pattern)
        return pattern
