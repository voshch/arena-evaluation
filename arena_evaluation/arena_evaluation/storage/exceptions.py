class ArenaEvaluationError(Exception):
    """Base exception for arena_evaluation pipeline."""


class CircularDependencyError(ArenaEvaluationError):
    """Raised when metric calculators have a circular dependency."""


class SchemaViolationError(ArenaEvaluationError):
    """Raised when data schema validation fails."""


class ManifestGenerationError(ArenaEvaluationError):
    """Raised when generating or validating a manifest fails."""


class ContractViolationError(ArenaEvaluationError):
    """Raised when the stored benchmark breaks the publish-time contract; nothing partial is published."""


class StaleBenchmarkError(ContractViolationError):
    """Raised when stored benchmark outputs predate the native layout and must be reprocessed with --force-process."""


class MetricInputError(ArenaEvaluationError):
    """Raised when a metric calculator's required input is missing; the registry re-raises it instead of recording a null."""


class UnmigratedMetadataError(ContractViolationError):
    """Raised when episode metadata predates modes and params and must be migrated first."""
