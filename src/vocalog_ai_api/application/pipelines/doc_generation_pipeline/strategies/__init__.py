from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.base import DocumentStrategy
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.srs import SRSStrategy
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.prd import PRDStrategy
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.sdd import SDDStrategy

_REGISTRY: dict[str, DocumentStrategy] = {
    "srs": SRSStrategy(),
    "prd": PRDStrategy(),
    "sdd": SDDStrategy(),
}


def get_strategy(document_type: str) -> DocumentStrategy:
    """
    Return the strategy instance for the given document type key.
    Raises ValueError for unrecognised types so callers fail fast.
    To add a new document type: subclass DocumentStrategy, instantiate it,
    and add it to _REGISTRY — no other code needs to change.
    """
    strategy = _REGISTRY.get(document_type.lower())
    if strategy is None:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown document type '{document_type}'. Supported types: {supported}"
        )
    return strategy


def list_supported_types() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = [
    "DocumentStrategy",
    "SRSStrategy",
    "PRDStrategy",
    "SDDStrategy",
    "get_strategy",
    "list_supported_types",
]
