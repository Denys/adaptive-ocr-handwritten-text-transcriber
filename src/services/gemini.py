from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from src.core.config import settings

class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def transcribe_image(self, file_path: str, model_name: str = "gemini-2.5-flash-001") -> str:
        """
        Uploads image and transcribes text using the specified Gemini model.
        Uses the Store-then-Upload pattern.
        """
        try:
            # 1. Upload file (The SDK handles the upload logic)
            # Note: client.files.upload is synchronous in the v1 SDK (v0.3.0+)
            # For high concurrency, this might block, but in a threadpool (FastAPI default for sync) it's okay.
            # Ideally we run this in a thread executor if it was purely blocking.
            # But let's assume standard usage for now.
            gemini_file = self.client.files.upload(file=file_path)

            # 2. Generate Content
            # We use simple prompts for v1
            response = self.client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_uri(file_uri=gemini_file.uri, mime_type="image/jpeg"),
                    types.Part.from_text(text="Create digital transcription of the text. Preserve line breaks.")
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )

            return response.text
        except Exception as e:
            # Pass through for tenacity to handle (if transient) or API to catch
            raise e

gemini_service = GeminiService()
