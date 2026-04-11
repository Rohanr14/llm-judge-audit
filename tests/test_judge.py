import pytest
from unittest.mock import MagicMock, patch

from llm_judge_audit.judge import (
    AnthropicJudge,
    BaseJudge,
    GeminiJudge,
    JudgeAPIError,
    OpenAIJudge,
    get_judge,
)


def test_get_judge_factory():
    judge1 = get_judge("gpt-4o", api_key="test-key")
    assert isinstance(judge1, OpenAIJudge)
    assert judge1.model_name == "gpt-4o"
    assert judge1.api_key == "test-key"

    judge2 = get_judge("o1-preview")
    assert isinstance(judge2, OpenAIJudge)

    judge3 = get_judge("claude-3-5-sonnet-20240620", api_key="test-key")
    assert isinstance(judge3, AnthropicJudge)

    with patch("llm_judge_audit.judge.importlib.import_module") as mock_import_module:
        mock_genai = MagicMock()
        mock_import_module.return_value = mock_genai
        judge4 = get_judge("gemini-1.5-pro", api_key="test-key")
        assert isinstance(judge4, GeminiJudge)

    judge5 = get_judge("llama-3.1-70b-instruct", api_key="test-key")
    assert isinstance(judge5, OpenAIJudge)


@patch("openai.OpenAI")
def test_openai_judge_evaluate(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"preference": "A"}'
    mock_client.chat.completions.create.return_value = mock_response

    judge = OpenAIJudge("gpt-4o", api_key="test-key")
    result = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")

    assert result == "A"
    mock_client.chat.completions.create.assert_called_once()

    mock_client.chat.completions.create.side_effect = Exception("API Error")
    with pytest.raises(JudgeAPIError):
        judge.evaluate_pairwise("Which is better?", "Response A", "Response B")


@patch("anthropic.Anthropic")
def test_anthropic_judge_evaluate(mock_anthropic):
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    mock_response = MagicMock()
    mock_response_content = MagicMock()
    mock_response_content.text = '{"preference": "B"}'
    mock_response.content = [mock_response_content]

    mock_client.messages.create.return_value = mock_response

    judge = AnthropicJudge("claude-3-5-sonnet", api_key="test-key")
    result = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")

    assert result == "B"
    mock_client.messages.create.assert_called_once()

    mock_client.messages.create.side_effect = Exception("API Error")
    with pytest.raises(JudgeAPIError):
        judge.evaluate_pairwise("Which is better?", "Response A", "Response B")


@patch("llm_judge_audit.judge.importlib.import_module")
def test_gemini_judge_evaluate(mock_import_module):
    mock_genai = MagicMock()
    mock_import_module.return_value = mock_genai
    mock_model_instance = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model_instance

    mock_response = MagicMock()
    mock_response.text = '{"preference": "Tie"}'
    mock_model_instance.generate_content.return_value = mock_response

    judge = GeminiJudge("gemini-1.5-pro", api_key="test-key")
    result = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")

    assert result == "Tie"
    mock_genai.configure.assert_called_once_with(api_key="test-key")
    mock_model_instance.generate_content.assert_called_once()

    mock_model_instance.generate_content.side_effect = Exception("API Error")
    with pytest.raises(JudgeAPIError):
        judge.evaluate_pairwise("Which is better?", "Response A", "Response B")


@patch("llm_judge_audit.judge.importlib.import_module", side_effect=ModuleNotFoundError)
def test_gemini_judge_missing_dependency_raises(_mock_import_module):
    with pytest.raises(JudgeAPIError):
        GeminiJudge("gemini-1.5-pro", api_key="test-key")
