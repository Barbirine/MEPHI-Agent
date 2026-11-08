"""Run paid RAG separately from scoring human annotations."""
import argparse
import csv
from dataclasses import asdict
import json
import re
from pathlib import Path
from config import Settings
from evaluation.metrics import score, labels
from rag.pipeline import build_pipeline

FIELDS = {'question', 'expected_answer', 'expected_document', 'expected_page',
          'expected_context', 'question_type'}

def load_dataset(path: Path) -> list[dict]:
    if path.suffix.lower() == '.json':
        rows = json.loads(path.read_text(encoding='utf-8'))
    else:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ('expected_document', 'expected_page', 'expected_context'):
                row[key] = json.loads(row[key])
    if not isinstance(rows, list) or not rows:
        raise ValueError('Датасет должен быть непустым массивом')
    for row in rows:
        if not FIELDS <= row.keys():
            raise ValueError('В датасете отсутствуют обязательные поля')
        for key in ('expected_document', 'expected_page', 'expected_context'):
            if not isinstance(row[key], list):
                raise ValueError(f'{key} должен быть массивом')
        if len({len(row[k]) for k in ('expected_document', 'expected_page', 'expected_context')}) != 1:
            raise ValueError('Массивы эталонных источников должны иметь одинаковую длину')
    return rows

def run(dataset: Path, output: Path) -> None:
    rows = load_dataset(dataset)
    pipeline = build_pipeline(Settings.load())
    results = []
    for row in rows:
        record = {'reference': row}
        try:
            answer = pipeline.ask(row['question'])
            record.update(answer=answer.text, contexts=[asdict(h) for h in answer.hits],
                          citations=[{'source_id': int(m.group(1)),
                                      'document': answer.hits[int(m.group(1))-1].chunk.document,
                                      'page': answer.hits[int(m.group(1))-1].chunk.page,
                                      'answer_offset': m.start()}
                                     for m in re.finditer(r'\[(\d+)\]', answer.text)],
                          annotation={'claims': None, 'reference_facts': None,
                                      'claim_supported': None, 'context_relevant': None,
                                      'reference_fact_found': None, 'citation_correct': None,
                                      'reviewer': '', 'notes': ''})
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        results.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

def summarize(path: Path) -> dict:
    records = json.loads(path.read_text(encoding='utf-8'))
    scores = []
    errors = 0
    for record in records:
        if 'error' in record:
            errors += 1
            continue
        annotation = record.get('annotation', {})
        for key, count in [('context_relevant', len(record['contexts'])),
                           ('citation_correct', len(record['citations']))]:
            value = labels(annotation.get(key))
            if value is not None and len(value) != count:
                raise ValueError(f'{key}: число меток не совпадает с числом объектов')
        for label_key, fact_key in [('claim_supported', 'claims'),
                                    ('reference_fact_found', 'reference_facts')]:
            value = labels(annotation.get(label_key))
            if value is not None:
                facts = annotation.get(fact_key)
                if not isinstance(facts, list) or len(facts) != len(value) or any(
                        not isinstance(fact, str) or not fact.strip() for fact in facts):
                    raise ValueError(f'{fact_key}: перечислите факты, соответствующие меткам')
        scores.append(score(annotation))
    aggregate = {}
    for name in ('Faithfulness', 'Context Precision', 'Context Recall', 'Citation Accuracy'):
        valid = [s[name] for s in scores if s[name] is not None]
        aggregate[name] = {'mean': sum(valid)/len(valid) if valid else None,
                           'annotated_questions': len(valid)}
    return {'questions': len(records), 'errors': errors, 'metrics': aggregate,
            'method': 'manual judgments; per-question macro average; undefined excluded'}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--dataset', type=Path, default=Path('data/test_questions.json'))
    runner.add_argument('--output', type=Path, default=Path('reports/review.json'))
    scorer = sub.add_parser('score')
    scorer.add_argument('review', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.dataset, args.output)
    else:
        print(json.dumps(summarize(args.review), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
