"""Metrics consume explicit human labels, never lexical proxy scores."""
from typing import Any

def labels(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(type(x) is not int or x not in (0, 1) for x in value):
        raise ValueError('Разметка должна быть массивом 0/1 или null')
    return value

def fraction(value: Any) -> float | None:
    checked = labels(value)
    return sum(checked) / len(checked) if checked else None

def context_precision(value: Any) -> float | None:
    """Average precision over ranked, manually judged retrieved chunks."""
    checked = labels(value)
    if not checked:
        return None
    relevant = sum(checked)
    if not relevant:
        return 0.0
    running = 0
    total = 0.0
    for rank, flag in enumerate(checked, 1):
        running += flag
        total += flag * running / rank
    return total / relevant

def score(annotation: dict) -> dict[str, float | None]:
    return {'Faithfulness': fraction(annotation.get('claim_supported')),
            'Context Precision': context_precision(annotation.get('context_relevant')),
            'Context Recall': fraction(annotation.get('reference_fact_found')),
            'Citation Accuracy': fraction(annotation.get('citation_correct'))}
