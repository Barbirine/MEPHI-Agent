import json
import pytest
from evaluation.metrics import score, context_precision
from evaluation.evaluate import load_dataset, summarize
from config import ROOT

def test_known_metric_values_and_missing_labels():
    assert context_precision([1,0,1]) == pytest.approx((1+2/3)/2)
    assert context_precision([0,0]) == 0
    result = score({'claim_supported':[1,0], 'context_relevant':[1,0,1],
                    'reference_fact_found':[1,1,0], 'citation_correct':[1,0]})
    assert result['Faithfulness'] == 0.5
    assert result['Context Recall'] == pytest.approx(2/3)
    assert result['Citation Accuracy'] == 0.5
    assert all(v is None for v in score({}).values())
    with pytest.raises(ValueError):
        context_precision([0.5])
    with pytest.raises(ValueError):
        context_precision([True])

def test_dataset_formats_identical():
    assert load_dataset(ROOT/'data/test_questions.csv') == load_dataset(ROOT/'data/test_questions.json')
    assert len({x['question_type'] for x in load_dataset(ROOT/'data/test_questions.json')}) == 6

def test_aggregate_coverage_and_errors(tmp_path):
    path = tmp_path/'review.json'
    path.write_text(json.dumps([{'answer':'a','contexts':[{}],'citations':[{}],
        'annotation':{'claims':['a','b'],'reference_facts':['a'],
                      'claim_supported':[1,0],'context_relevant':[1],
                      'reference_fact_found':[1],'citation_correct':[0]}},
        {'answer':'b','contexts':[],'citations':[],'annotation':{}},
        {'error':'network'}]))
    result = summarize(path)
    assert result['errors'] == 1
    assert result['metrics']['Faithfulness'] == {'mean':0.5,'annotated_questions':1}
    assert result['metrics']['Citation Accuracy']['mean'] == 0

def test_run_exports_unscored_answers(tmp_path, monkeypatch):
    import evaluation.evaluate as module
    from rag.models import Answer, Chunk, Hit
    class FakePipeline:
        def ask(self, question):
            return Answer('Факт [1]. Ещё факт [1].',
                          [Hit(Chunk('a','synthetic_a.pdf',1,'text'),0.8)], [1])
    monkeypatch.setattr(module,'build_pipeline', lambda _: FakePipeline())
    path = tmp_path/'review.json'
    module.run(ROOT/'data/test_questions.json',path)
    records = json.loads(path.read_text())
    assert len(records) == 6
    assert len(records[0]['citations']) == 2
    assert records[0]['citations'][1]['answer_offset'] > records[0]['citations'][0]['answer_offset']
    assert all(x['mean'] is None for x in summarize(path)['metrics'].values())
