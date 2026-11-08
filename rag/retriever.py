import logging
from rag.embeddings import Embedder
from rag.models import Hit
from rag.vector_store import VectorStore

class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int, threshold: float) -> None:
        self.store, self.embedder = store, embedder
        self.top_k, self.threshold = top_k, threshold

    def retrieve(self, question: str) -> list[Hit]:
        if not question.strip():
            raise ValueError('Введите вопрос')
        if len(question) > 4000:
            raise ValueError('Вопрос должен быть не длиннее 4000 символов')
        metadata = self.store.metadata()
        if metadata['model'] != self.embedder.model:
            raise ValueError('Модель эмбеддингов изменилась. Пересоздайте индекс в боковой панели.')
        hits = self.store.search(self.embedder.embed([question])[0], self.embedder.model,
                                 self.top_k, self.threshold)
        logging.getLogger(__name__).info('Найдено %s фрагментов', len(hits))
        return hits
