from src.db.models import User, UserPattern

class ModelSelector:
    MODELS = {
        "MVP_BASELINE": "gemini-2.5-flash-001",
        "REASONING_FLASH": "gemini-3-flash-preview",
        "REASONING_PRO": "gemini-3-pro-preview",
        "IMAGE_GEN": "gemini-3-pro-image-preview"
    }

    @classmethod
    def select(cls, task: str, user_pattern: UserPattern = None) -> str:
        if task == "ocr_baseline":
            return cls.MODELS["MVP_BASELINE"]

        if task == "ocr_personalized":
            # Upgrade if accuracy is struggling
            if user_pattern and user_pattern.current_accuracy is not None:
                if user_pattern.current_accuracy < 85.0:
                    return cls.MODELS["REASONING_FLASH"]
            return cls.MODELS["MVP_BASELINE"]

        if task == "concept_explanation":
             return cls.MODELS["REASONING_FLASH"]

        if task == "layout_analysis":
            return cls.MODELS["REASONING_PRO"]

        return cls.MODELS["MVP_BASELINE"]
