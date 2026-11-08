"""Validate uploaded PDF bytes locally, then save without overwriting originals."""
import hashlib
import logging
from pathlib import Path
import re
import tempfile
from rag.pdf_parser import parse_pdf

log = logging.getLogger(__name__)

def save_uploaded_pdf(filename: str, content: bytes, directory: Path,
                      max_size_bytes: int = 20 * 1024 * 1024) -> Path:
    name = filename.replace('\\', '/').rsplit('/', 1)[-1]
    if not name or not name.lower().endswith('.pdf') or re.search(r'[\x00-\x1f]', name):
        raise ValueError('Выберите файл с расширением .pdf и корректным именем')
    if not content:
        raise ValueError('Загружен пустой файл')
    if len(content) > max_size_bytes:
        raise ValueError(f'PDF превышает лимит {max_size_bytes // (1024 * 1024)} МБ')
    if b'%PDF-' not in content[:1024]:
        raise ValueError('Содержимое файла не похоже на PDF')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    # Same name with different content creates a new file, preserving the original.
    if target.exists():
        if target.read_bytes() == content:
            return target
        target = directory / f'{Path(name).stem}_{hashlib.sha256(content).hexdigest()}.pdf'
        if target.exists():
            if target.read_bytes() == content:
                return target
            raise ValueError('Конфликт имени файла. Переименуйте загружаемый PDF.')
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        parse_pdf(temporary, target.name)
        temporary.replace(target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    log.info('Сохранён PDF %s (%s байт)', target.name, len(content))
    return target
