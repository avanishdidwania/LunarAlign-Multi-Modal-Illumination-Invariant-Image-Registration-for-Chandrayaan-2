from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from lunar_reg.detection.base import DetectionResult, Keypoint

@dataclass
class MatchPair:
    source_idx: int
    reference_idx: int
    source_pt: tuple[float, float]    # (x, y) in source image
    reference_pt: tuple[float, float] # (x, y) in reference image
    confidence: float                 # Match confidence [0, 1]

@dataclass
class MatchingResult:
    matches: List[MatchPair]
    source_keypoints: Optional[List[Keypoint]] = None
    reference_keypoints: Optional[List[Keypoint]] = None

class FeatureMatcherBase(ABC):
    """Abstract interface for feature matching algorithms."""

    @abstractmethod
    def match(self, source_result: DetectionResult,
              reference_result: DetectionResult) -> MatchingResult:
        """Establish correspondences between source and reference features."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass
