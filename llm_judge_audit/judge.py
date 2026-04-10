import json
import importlib
from abc import ABC, abstractmethod
from typing import Literal

from llm_judge_audit.config import config
from llm_judge_audit.logger import logger

class BaseJudge(ABC):
    """Abstract base class for all LLM judges."""
    
    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        """Evaluates two responses and returns the preferred one."""
        pass

    def evaluate_pairwise_with_confidence(
        self, prompt: str, response_a: str, response_b: str
    ) -> tuple[Literal["A", "B", "Tie"], float | None]:
        """
        Optional extension used by confidence calibration tests.
        Returns a (preference, confidence) tuple where confidence is in [0.0, 1.0].
        """
        return self.evaluate_pairwise(prompt, response_a, response_b), None

class OpenAIJudge(BaseJudge):
    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name, api_key or config.OPENAI_API_KEY)
        import openai
        self.client = openai.OpenAI(api_key=self.api_key)

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        logger.debug(f"OpenAI evaluate_pairwise called with {self.model_name}")
        system_prompt = (
            "You are an expert evaluator. You will be provided with a prompt and two responses (Response A and Response B). "
            "Your task is to evaluate which response better addresses the prompt. "
            "You must output a JSON object with a single key 'preference' and a value of 'A', 'B', or 'Tie'."
        )
        user_prompt = f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            result_json = response.choices[0].message.content
            if result_json is None:
                logger.warning("Received None from OpenAI API")
                return "Tie"
            result = json.loads(result_json)
            pref = result.get("preference", "Tie")
            if pref not in ("A", "B", "Tie"):
                return "Tie"
            return pref
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "Tie"

class AnthropicJudge(BaseJudge):
    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name, api_key or config.ANTHROPIC_API_KEY)
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        logger.debug(f"Anthropic evaluate_pairwise called with {self.model_name}")
        system_prompt = (
            "You are an expert evaluator. You will be provided with a prompt and two responses (Response A and Response B). "
            "Your task is to evaluate which response better addresses the prompt. "
            "Output your choice in JSON format with a single key 'preference' containing exactly 'A', 'B', or 'Tie'."
        )
        user_prompt = f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"

        try:
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.0,
            )
            # Find JSON block or parse directly
            content = response.content[0].text
            # Very basic extraction if not perfectly JSON formatted
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                result = json.loads(json_str)
                pref = result.get("preference", "Tie")
                if pref in ("A", "B", "Tie"):
                    return pref
            
            # Fallback heuristic
            if "A" in content and "B" not in content:
                return "A"
            if "B" in content and "A" not in content:
                return "B"
            return "Tie"
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return "Tie"

class GeminiJudge(BaseJudge):
    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name, api_key or config.GEMINI_API_KEY)
        self.model = None
        try:
            genai = importlib.import_module("google.generativeai")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ModuleNotFoundError:
            logger.warning(
                "google-generativeai is not installed; GeminiJudge will return 'Tie'."
            )

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        logger.debug(f"Gemini evaluate_pairwise called with {self.model_name}")
        if self.model is None:
            return "Tie"
        full_prompt = (
            "You are an expert evaluator. You will be provided with a prompt and two responses (Response A and Response B). "
            "Your task is to evaluate which response better addresses the prompt. "
            "Output your choice as a valid JSON object with a single key 'preference' containing exactly 'A', 'B', or 'Tie'.\n\n"
            f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"
        )

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config={"temperature": 0.0}
            )
            content = response.text
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                result = json.loads(json_str)
                pref = result.get("preference", "Tie")
                if pref in ("A", "B", "Tie"):
                    return pref
            return "Tie"
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return "Tie"

def get_judge(model_name: str, api_key: str | None = None) -> BaseJudge:
    """Factory function to get the appropriate judge instance based on model name."""
    if model_name.startswith(("gpt-", "o1-", "o3-")):
        return OpenAIJudge(model_name, api_key)
    elif model_name.startswith("claude-"):
        return AnthropicJudge(model_name, api_key)
    elif model_name.startswith("gemini-"):
        return GeminiJudge(model_name, api_key)
    else:
        # Fallback to OpenAI API format for open-weights models served via standard endpoints (e.g., vLLM)
        # Assuming the base URL is set in the environment or client instantiation for OpenAI
        logger.warning(f"Unknown model prefix for '{model_name}'. Falling back to OpenAI compatible API.")
        return OpenAIJudge(model_name, api_key)
