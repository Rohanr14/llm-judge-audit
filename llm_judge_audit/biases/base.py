from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from typing import Any, Callable, Dict, Iterable, List, TypeVar

from pydantic import BaseModel

from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge

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

    def _parallel_map(self, fn: Callable[[T], Any], items: Iterable[T], max_workers: int | None = None) -> list[Any]:
        seq = list(items)
        if not seq:
            return []

        workers = max_workers or min(32, (cpu_count() or 1) * 5)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(fn, seq))
