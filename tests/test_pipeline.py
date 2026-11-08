from dataclasses import replace
import numpy as np
import pytest
from config import Settings
from rag.models import Chunk, Hit
from rag.pipeline import Pipeline, ingest
from rag.llm import REFUSAL
from rag.vector_store import VectorStore
from scripts.make_demo import create_demo

class Search:
    def __init__(self, hits):
        self.hits = hits
    def retrieve(self, question):
        return self.hits

class Generate:
    def __init__(self, text):
        self.text = text
        self.calls = 0
    def generate(self, question, hits):
        self.calls += 1
        return self.text

def test_pipeline_sources_and_refusal():
    hit = Hit(Chunk('a','demo.pdf',3,'Source'),0.9)
    generator = Generate('Факт [1]')
    answer = Pipeline(Search([hit]),generator).ask('Question')
    assert answer.citation_ids == [1]
    assert answer.hits[0].chunk.page == 3
    no_hits = Pipeline(Search([]),generator).ask('Question')
    assert no_hits.text == REFUSAL
    assert generator.calls == 1
    for text in ['Факт [2]','Факт [0]','Факт без ссылки']:
        with pytest.raises(ValueError):
            Pipeline(Search([hit]),Generate(text)).ask('Question')

def test_ingest_full_rebuild_with_test_embeddings(tmp_path, monkeypatch):
    import rag.pipeline as module
    create_demo(tmp_path/'pdfs')
    store = VectorStore(tmp_path/'db.sqlite')
    class Embedder:
        model = 'unit-test-only'
        def embed(self, texts):
            return np.array([[len(text),1] for text in texts],dtype=np.float32)
    monkeypatch.setattr(module, 'LocalEmbedder', Embedder)
    settings = replace(Settings.load(),pdf_dir=tmp_path/'pdfs',index_path=store.path)
    assert ingest(settings) == 5
    assert store.metadata()['count'] == 5
    (tmp_path/'pdfs/synthetic_b.pdf').unlink()
    assert ingest(settings) == 4
    assert {h.chunk.document for h in store.search(np.array([1,0]),'unit-test-only',10,-1)} == {'synthetic_a.pdf'}
