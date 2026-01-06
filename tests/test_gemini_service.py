import pytest
from unittest.mock import MagicMock, patch
from src.services.gemini import GeminiService

class MockGeminiFile:
    def __init__(self, uri):
        self.uri = uri

class MockResponse:
    def __init__(self, text):
        self.text = text

@pytest.fixture
def mock_genai_client():
    with patch("src.services.gemini.genai.Client") as mock:
        yield mock

def test_transcribe_image_success(mock_genai_client):
    # Setup mock
    client_instance = mock_genai_client.return_value
    client_instance.files.upload.return_value = MockGeminiFile("gs://fake/uri")
    client_instance.models.generate_content.return_value = MockResponse("Transcribed Text")

    service = GeminiService()
    result = service.transcribe_image("/tmp/fake.jpg")

    assert result == "Transcribed Text"
    client_instance.files.upload.assert_called_once_with(file="/tmp/fake.jpg")
    client_instance.models.generate_content.assert_called_once()

def test_transcribe_image_retry_on_failure(mock_genai_client):
    # Setup mock to fail twice then succeed
    client_instance = mock_genai_client.return_value
    client_instance.files.upload.return_value = MockGeminiFile("gs://fake/uri")

    # Simulate transient error then success
    client_instance.models.generate_content.side_effect = [
        Exception("Transient Error"),
        Exception("Transient Error"),
        MockResponse("Success after retry")
    ]

    service = GeminiService()
    result = service.transcribe_image("/tmp/fake.jpg")

    assert result == "Success after retry"
    assert client_instance.models.generate_content.call_count == 3
