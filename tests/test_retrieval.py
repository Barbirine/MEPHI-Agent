import numpy as np
import pytest
from rag.models import Chunk
from rag.vector_store import VectorStore
from rag.retriever import Retriever

def test_cosine_threshold_metadata_and_atomicity(tmp_path):
    store = VectorStore(tmp_path / 'store.sqlite')
    chunks = [Chunk('a','a.pdf',1,'alpha'),Chunk('b','b.pdf',2,'beta')]
    store.replace(chunks,np.array([[10,0],[0,2]],dtype=np.float32),'test')
    hits = store.search(np.array([1,0]),'test',2,0.5)
    assert [h.chunk for h in hits] == chunks[:1]
    assert hits[0].score == pytest.approx(1)
    with pytest.raises(ValueError,match='Модель'):
        store.search(np.array([1,0]),'other',2,0)
    with pytest.raises(ValueError):
        store.search(np.array([1,0,1]),'test',2,0)
    with pytest.raises(ValueError):
        store.replace(chunks,np.zeros((2,2)),'test')
    # SQL primary-key failure after DELETE must roll back the whole rebuild.
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.replace([chunks[0],chunks[0]],np.eye(2),'test')
    assert store.metadata()['count'] == 2
    assert len(store.search(np.array([1,0]),'test',2,-1)) == 2

def test_retriever_query_and_missing_index(tmp_path):
    class Embedder:
        model = 'test'
        calls = 0
        def embed(self, texts):
            self.calls += 1
            return np.array([[1,0]],dtype=np.float32)
    embedder = Embedder()
    store = VectorStore(tmp_path/'index.sqlite')
    retriever = Retriever(store,embedder,2,0.5)
    with pytest.raises(ValueError):
        retriever.retrieve(' ')
    with pytest.raises(FileNotFoundError):
        retriever.retrieve('question')
    assert embedder.calls == 0
    store.replace([Chunk('a','a.pdf',1,'text')],np.array([[1,0]]),'test')
    assert retriever.retrieve('question')[0].chunk.page == 1

def test_model_mismatch_does_not_spend_embedding_request(tmp_path):
    class Embedder:
        model = 'gigachat:Embeddings'
        def embed(self, texts):
            raise AssertionError('API must not be called for an incompatible index')
    store = VectorStore(tmp_path/'old.sqlite')
    store.replace([Chunk('a','a.pdf',1,'text')],np.array([[1,0]]),'text-embedding-3-small')
    with pytest.raises(ValueError,match='Пересоздайте индекс'):
        Retriever(store,Embedder(),2,0.25).retrieve('вопрос')


def test_local_search_matches_russian_word_forms(tmp_path):
    from rag.embeddings import LocalEmbedder

    embedder = LocalEmbedder()
    chunks = [
        Chunk('a', 'physics.pdf', 1, 'Определение теплоёмкости вещества.'),
        Chunk('b', 'algebra.pdf', 2, 'Сложение и вычитание матриц.'),
    ]
    store = VectorStore(tmp_path / 'local.sqlite')
    store.replace(chunks, embedder.embed([c.text for c in chunks]), embedder.model)

    results = Retriever(store, embedder, 5, 0.25).retrieve('Что такое теплоёмкость?')
    assert [hit.chunk.document for hit in results] == ['physics.pdf']
    assert Retriever(store, embedder, 5, 0.25).retrieve('Где живут пингвины?') == []
