from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from src.core.config import settings
import json
import yaml

# Load prompts
with open("prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)

class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def transcribe_image(self, file_path: str, model_name: str = "gemini-2.5-flash-001", user_patterns: str = None) -> str:
        """
        Uploads image and transcribes text using the specified Gemini model.
        Uses the Store-then-Upload pattern.
        If user_patterns is provided, uses the personalized prompt.
        """
        try:
            gemini_file = self.client.files.upload(file=file_path)

            if user_patterns:
                template = PROMPTS["ocr_personalized"]["template"]
                prompt = template.format(user_patterns=user_patterns)
                # Use model from args (selector logic) or fallback to prompt default
                model = model_name or PROMPTS["ocr_personalized"]["model"]
            else:
                template = PROMPTS["ocr_baseline"]["template"]
                prompt = template
                model = model_name or PROMPTS["ocr_baseline"]["model"]

            response = self.client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_uri(file_uri=gemini_file.uri, mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt)
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )

            return response.text
        except Exception as e:
            raise e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def analyze_calibration(self, ground_truth: str, ocr_output: str) -> dict:
        """
        Analyzes corrections to build a confusion matrix.
        Note: We are not passing user corrections explicitly for MVP, just comparing GT vs OCR.
        The prompt template expects 'corrections', but we can infer it from diffing GT vs OCR.
        """
        try:
            template = PROMPTS["calibration_analysis"]["template"]
            # We treat 'corrections' as the difference, but the prompt handles analysis.
            prompt = template.format(
                ground_truth=ground_truth,
                ocr_output=ocr_output,
                corrections="See differences between Ground Truth and OCR Output"
            )

            # Use code_execution to calculate metrics accurately
            config = types.GenerateContentConfig(
                tools=[types.Tool(code_execution=types.CodeExecution())],
                response_mime_type="application/json",
                temperature=0.0
            )

            response = self.client.models.generate_content(
                model=PROMPTS["calibration_analysis"]["model"],
                contents=[types.Part.from_text(text=prompt)],
                config=config
            )

            # The model should return a JSON object with the results
            try:
                # If code execution is used, the response might contain executable code blocks.
                # However, with response_mime_type="application/json", it attempts to force JSON output.
                # Gemini 3 might wrap it. Let's try to parse response.text directly.
                return json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback: simple text parsing or return raw text wrapped
                return {"error": "Failed to parse JSON", "raw": response.text}

        except Exception as e:
            raise e

gemini_service = GeminiService()
