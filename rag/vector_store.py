"""SQLite stores text and float32 vectors; NumPy performs exact cosine search."""
import json
import sqlite3
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from rag.models import Chunk, Hit

class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def replace(self, chunks: list[Chunk], vectors: NDArray[np.float32], model: str) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not chunks:
            raise ValueError('Размеры фрагментов и векторов не совпадают или индекс пуст')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError('Некорректные эмбеддинги')
        normalized = vectors / norms
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            # One transaction: failed rebuild preserves the previous index.
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, document TEXT, page INTEGER, text TEXT, vector BLOB)')
            db.execute('DELETE FROM chunks')
            db.execute('DELETE FROM metadata')
            db.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)',
                           [(c.id, c.document, c.page, c.text, v.tobytes()) for c, v in zip(chunks, normalized)])
            meta = {'model': model, 'dimension': vectors.shape[1], 'count': len(chunks)}
            db.executemany('INSERT INTO metadata VALUES (?,?)',
                           [(key, json.dumps(value)) for key, value in meta.items()])

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise FileNotFoundError('Индекс не найден. Запустите python -m scripts.ingest')
        return sqlite3.connect(f'{self.path.resolve().as_uri()}?mode=ro', uri=True)

    def metadata(self) -> dict:
        with self._connect() as db:
            return {key: json.loads(value) for key, value in db.execute('SELECT key,value FROM metadata')}

    def search(self, query: NDArray[np.float32], model: str, top_k: int,
               min_similarity: float) -> list[Hit]:
        if top_k < 1:
            raise ValueError('top_k должен быть положительным')
        with self._connect() as db:
            meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
            if meta['model'] != model:
                raise ValueError('Модель эмбеддингов изменилась. Пересоздайте индекс.')
            query = np.asarray(query, dtype=np.float32)
            if query.shape != (meta['dimension'],) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
                raise ValueError('Некорректная размерность или значения вектора запроса')
            rows = db.execute('SELECT id,document,page,text,vector FROM chunks ORDER BY id').fetchall()
        matrix = np.stack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        scores = matrix @ (query / np.linalg.norm(query))
        order = np.argsort(-scores, kind='stable')[:top_k]
        return [Hit(Chunk(*rows[i][:4]), float(scores[i])) for i in order
                if scores[i] >= min_similarity]
