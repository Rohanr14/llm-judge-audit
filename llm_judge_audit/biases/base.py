from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, TypeVar

from pydantic import BaseModel

from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.runtime import SETTINGS

T = TypeVar("T")


class BiasTestResult(BaseModel):
    bias_name: str
    score: float
    details: Dict[str, Any]


class BaseBiasTest(ABC):
    """Abstract base class for all bias tests."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        """Runs the bias test on the provided dataset using the given judge."""
        pass

    def _parallel_map(
        self, fn: Callable[[T], Any], items: Iterable[T], max_workers: int | None = None
    ) -> list[Any]:
        seq = list(items)
        if not seq:
            return []

        # Honour the global concurrency cap by default. Callers can still pass
        # ``max_workers`` explicitly to override (e.g. in tests that stub out
        # the judge and aren't bound by API rate limits).
        workers = max_workers if max_workers is not None else SETTINGS.max_concurrency
        workers = max(1, min(workers, len(seq)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(fn, seq))
