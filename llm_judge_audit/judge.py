import json
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import yaml
from tenacity import RetryError, retry, retry_if_exception, stop_after_attempt, wait_exponential

from llm_judge_audit.config import config
from llm_judge_audit.logger import logger


class JudgeAPIError(RuntimeError):
    """Raised when a judge API cannot return a valid preference after retries."""


class BaseJudge(ABC):
    """Abstract base class for all LLM judges."""

    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key

    @property
    def prompt_family(self) -> str:
        """Returns the prompt family key used in prompts.yaml."""
        return "default"

    @abstractmethod
    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        """Evaluates two responses and returns the preferred one."""

    def evaluate_pairwise_with_confidence(
        self, prompt: str, response_a: str, response_b: str
    ) -> tuple[Literal["A", "B", "Tie"], float | None]:
        """
        Optional extension used by confidence calibration tests.
        Returns a (preference, confidence) tuple where confidence is in [0.0, 1.0].
        """
        return self.evaluate_pairwise(prompt, response_a, response_b), None


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    transient_markers = (
        "rate limit",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection",
        "service unavailable",
        "too many requests",
        "internal server error",
    )
    return any(marker in message for marker in transient_markers)


@retry(
    retry=retry_if_exception(_is_transient_error),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _retry_transient(callable_fn):
    return callable_fn()


def _load_prompts(path: str | None = None) -> dict[str, dict[str, str]]:
    prompt_path = Path(path or config.PROMPTS_PATH)
    with prompt_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload.get("pairwise_evaluation", {})


_PROMPTS = _load_prompts()


def _get_system_prompt(prompt_family: str) -> str:
    family_cfg = _PROMPTS.get(prompt_family) or _PROMPTS.get("default") or {}
    system_prompt = family_cfg.get("system_prompt")
    if not system_prompt:
        raise JudgeAPIError(f"Missing system_prompt for prompt family '{prompt_family}'.")
    return system_prompt


def _parse_preference_from_json(content: str) -> Literal["A", "B", "Tie"]:
    if "{" in content and "}" in content:
        content = content[content.find("{") : content.rfind("}") + 1]

    result = json.loads(content)
    pref = result.get("preference")
    if pref not in ("A", "B", "Tie"):
        raise JudgeAPIError(f"Invalid preference payload: {result}")
    return pref


class OpenAIJudge(BaseJudge):
    @property
    def prompt_family(self) -> str:
        return "openai"

    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name, api_key or config.OPENAI_API_KEY)
        import openai

        self.client = openai.OpenAI(api_key=self.api_key)

    def _call_api(self, prompt: str, response_a: str, response_b: str) -> str:
        user_prompt = f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _get_system_prompt(self.prompt_family)},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result_json = response.choices[0].message.content
        if result_json is None:
            raise JudgeAPIError("Received empty response from OpenAI API.")
        return result_json

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        logger.debug("OpenAI evaluate_pairwise called with %s", self.model_name)
        try:
            result_json = _retry_transient(lambda: self._call_api(prompt, response_a, response_b))
            return _parse_preference_from_json(result_json)
        except RetryError as exc:
            logger.error("OpenAI transient retries exhausted: %s", exc)
            raise JudgeAPIError(f"OpenAI judge failed after retries for model {self.model_name}.") from exc
        except Exception as exc:
            logger.error("Error calling OpenAI API: %s", exc)
            if isinstance(exc, JudgeAPIError):
                raise
            raise JudgeAPIError(f"OpenAI judge request failed for model {self.model_name}.") from exc


class AnthropicJudge(BaseJudge):
    @property
    def prompt_family(self) -> str:
        return "anthropic"

    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name, api_key or config.ANTHROPIC_API_KEY)
        import anthropic

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _call_api(self, prompt: str, response_a: str, response_b: str) -> str:
        user_prompt = f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"
        response = self.client.messages.create(
            model=self.model_name,
            system=_get_system_prompt(self.prompt_family),
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        return response.content[0].text

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        logger.debug("Anthropic evaluate_pairwise called with %s", self.model_name)
        try:
            content = _retry_transient(lambda: self._call_api(prompt, response_a, response_b))
            return _parse_preference_from_json(content)
        except RetryError as exc:
            logger.error("Anthropic transient retries exhausted: %s", exc)
            raise JudgeAPIError(f"Anthropic judge failed after retries for model {self.model_name}.") from exc
        except Exception as exc:
            logger.error("Error calling Anthropic API: %s", exc)
            if isinstance(exc, JudgeAPIError):
                raise
            raise JudgeAPIError(f"Anthropic judge request failed for model {self.model_name}.") from exc


class GeminiJudge(BaseJudge):
    @property
    def prompt_family(self) -> str:
        return "gemini"

    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name, api_key or config.GEMINI_API_KEY)
        self.model = None
        try:
            genai = importlib.import_module("google.generativeai")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ModuleNotFoundError as exc:
            raise JudgeAPIError("google-generativeai is not installed; GeminiJudge cannot be used.") from exc

    def _call_api(self, prompt: str, response_a: str, response_b: str) -> str:
        full_prompt = (
            f"{_get_system_prompt(self.prompt_family)}\n\n"
            f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"
        )
        response = self.model.generate_content(full_prompt, generation_config={"temperature": 0.0})
        return response.text

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Literal["A", "B", "Tie"]:
        logger.debug("Gemini evaluate_pairwise called with %s", self.model_name)
        if self.model is None:
            raise JudgeAPIError("Gemini model is not initialized.")
        try:
            content = _retry_transient(lambda: self._call_api(prompt, response_a, response_b))
            return _parse_preference_from_json(content)
        except RetryError as exc:
            logger.error("Gemini transient retries exhausted: %s", exc)
            raise JudgeAPIError(f"Gemini judge failed after retries for model {self.model_name}.") from exc
        except Exception as exc:
            logger.error("Error calling Gemini API: %s", exc)
            if isinstance(exc, JudgeAPIError):
                raise
            raise JudgeAPIError(f"Gemini judge request failed for model {self.model_name}.") from exc


def get_judge(model_name: str, api_key: str | None = None) -> BaseJudge:
    """Factory function to get the appropriate judge instance based on model name."""
    if model_name.startswith(("gpt-", "o1-", "o3-")):
        return OpenAIJudge(model_name, api_key)
    if model_name.startswith("claude-"):
        return AnthropicJudge(model_name, api_key)
    if model_name.startswith("gemini-"):
        return GeminiJudge(model_name, api_key)

    logger.warning("Unknown model prefix for '%s'. Falling back to OpenAI compatible API.", model_name)
    return OpenAIJudge(model_name, api_key)
