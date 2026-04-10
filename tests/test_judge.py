import pytest
from unittest.mock import MagicMock, patch

from llm_judge_audit.judge import (
    BaseJudge,
    OpenAIJudge,
    AnthropicJudge,
    GeminiJudge,
    get_judge
)

def test_get_judge_factory():
    # Test OpenAI models
    judge1 = get_judge("gpt-4o", api_key="test-key")
    assert isinstance(judge1, OpenAIJudge)
    assert judge1.model_name == "gpt-4o"
    assert judge1.api_key == "test-key"

    judge2 = get_judge("o1-preview")
    assert isinstance(judge2, OpenAIJudge)

    # Test Anthropic models
    judge3 = get_judge("claude-3-5-sonnet-20240620", api_key="test-key")
    assert isinstance(judge3, AnthropicJudge)

    # Test Gemini models
    judge4 = get_judge("gemini-1.5-pro", api_key="test-key")
    assert isinstance(judge4, GeminiJudge)

    # Test unknown model fallback
    judge5 = get_judge("llama-3.1-70b-instruct", api_key="test-key")
    assert isinstance(judge5, OpenAIJudge)

@patch("openai.OpenAI")
def test_openai_judge_evaluate(mock_openai):
    # Mock the OpenAI client response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"preference": "A"}'
    mock_client.chat.completions.create.return_value = mock_response

    judge = OpenAIJudge("gpt-4o", api_key="test-key")
    result = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")
    
    assert result == "A"
    mock_client.chat.completions.create.assert_called_once()
    
    # Test Tie on error
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    result_error = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")
    assert result_error == "Tie"


@patch("anthropic.Anthropic")
def test_anthropic_judge_evaluate(mock_anthropic):
    # Mock the Anthropic client response
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

    # Test Tie on error
    mock_client.messages.create.side_effect = Exception("API Error")
    result_error = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")
    assert result_error == "Tie"


@patch("llm_judge_audit.judge.importlib.import_module")
def test_gemini_judge_evaluate(mock_import_module):
    # Mock the Gemini client response
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

    # Test error handling
    mock_model_instance.generate_content.side_effect = Exception("API Error")
    result_error = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")
    assert result_error == "Tie"


@patch("llm_judge_audit.judge.importlib.import_module", side_effect=ModuleNotFoundError)
def test_gemini_judge_missing_dependency_returns_tie(_mock_import_module):
    judge = GeminiJudge("gemini-1.5-pro", api_key="test-key")
    result = judge.evaluate_pairwise("Which is better?", "Response A", "Response B")
    assert result == "Tie"
