from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem

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
