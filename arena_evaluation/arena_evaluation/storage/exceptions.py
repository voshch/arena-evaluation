class ArenaEvaluationError(Exception):
    """Base exception for arena_evaluation pipeline."""

    pass


class CircularDependencyError(ArenaEvaluationError):
    """Raised when metric calculators have a circular dependency."""

    pass


class SchemaViolationError(ArenaEvaluationError):
    """Raised when data schema validation fails."""

    pass


class ManifestGenerationError(ArenaEvaluationError):
    """Raised when generating or validating a manifest fails."""

    pass


class ContractViolationError(ArenaEvaluationError):
    """Raised when the stored benchmark breaks the publish-time contract; nothing partial is published."""

    pass
