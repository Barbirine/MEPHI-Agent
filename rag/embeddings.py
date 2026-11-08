from collections import Counter
from hashlib import blake2b
import re
import unicodedata
from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from rag.gigachat_client import GigaChatClient, GigaChatError

class Embedder(Protocol):
    model: str
    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...

class GigaChatEmbedder:
    def __init__(self, client: GigaChatClient, model: str = 'Embeddings') -> None:
        self.client = client
        self.api_model = model
        # Provider-prefixed identity prevents reuse of old OpenAI vectors.
        self.model = f'gigachat:{model}'

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError('Нужны непустые тексты')
        vectors = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            response = self.client.post('embeddings', {'model': self.api_model, 'input': batch})
            try:
                items = sorted(response['data'], key=lambda item: item['index'])
                if [item['index'] for item in items] != list(range(len(batch))):
                    raise ValueError('Неверные индексы')
                vectors.extend(item['embedding'] for item in items)
            except (KeyError, TypeError, ValueError) as exc:
                raise GigaChatError('GigaChat вернул некорректный список эмбеддингов.') from exc
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise GigaChatError('Несовместимые размеры эмбеддингов.') from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] == 0:
            raise GigaChatError('Некорректная размерность эмбеддингов.')
        if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
            raise GigaChatError('Эмбеддинги содержат некорректные значения.')
        return matrix


class LocalEmbedder:
    """Deterministic lexical vectors for CPU search, without model downloads."""

    model = 'local:lexical-hash-v1'
    dimensions = 4096
    stop_words = frozenset(
        'а в во и из к на о об от по с со у для или что как где когда какой '
        'какая какое какие такое такой означает объясни назови the a an is '
        'of to in and or what how which where when are define'.split()
    )

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError('Нужны непустые тексты')

        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in zip(vectors, texts):
            normalized = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
            words = re.findall(r'[^\W_]+', normalized)
            counts = Counter(word for word in words if word not in self.stop_words)
            if not counts:
                counts = Counter(words or re.findall(r'\S', normalized))

            for word, count in counts.items():
                weight = count ** 0.5
                self._add(row, 'word:' + word, weight)
                # Character trigrams let inflected forms share search features.
                if len(word) >= 3:
                    grams = set(word[i:i + 3] for i in range(len(word) - 2))
                    for gram in sorted(grams):
                        self._add(row, 'gram:' + gram, weight / len(grams) ** 0.5)
        return vectors

    def _add(self, row: NDArray[np.float32], feature: str, weight: float) -> None:
        digest = blake2b(feature.encode('utf-8'), digest_size=8).digest()
        index = int.from_bytes(digest, 'little') % self.dimensions
        row[index] += weight
