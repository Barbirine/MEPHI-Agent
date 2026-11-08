import logging
from config import Settings
from rag.pipeline import ingest

def main() -> int:
    try:
        print(f'Проиндексировано фрагментов: {ingest(Settings.load())}')
        return 0
    except Exception as exc:
        logging.getLogger(__name__).error('Индексация не завершена: %s', exc)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
