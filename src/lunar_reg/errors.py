from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")

@dataclass
class PipelineError:
    stage: str                    # Which pipeline stage failed (e.g., "loading", "detection")
    error_code: str               # Machine-readable code (e.g., "UNSUPPORTED_FORMAT")
    message: str                  # Human-readable description
    details: Optional[Dict[str, Any]] = None  # Additional context (counts, thresholds, etc.)
    recoverable: bool = False     # Can the pipeline continue with degraded output?

@dataclass
class StageResult(Generic[T]):
    success: bool
    value: Optional[T] = None
    error: Optional[PipelineError] = None
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, value: T, warnings: Optional[List[str]] = None) -> "StageResult[T]":
        return cls(success=True, value=value, warnings=warnings or [])

    @classmethod
    def fail(cls, error: PipelineError, warnings: Optional[List[str]] = None) -> "StageResult[T]":
        return cls(success=False, error=error, warnings=warnings or [])
