"""Configuration from .env; paths are relative to the project directory."""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Settings:
    credentials: str = field(repr=False)
    chat_model: str
    pdf_dir: Path
    index_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float
    scope: str
    base_url: str
    auth_url: str
    ca_bundle_file: Path | None
    timeout: float
    max_upload_mb: int

    @classmethod
    def load(cls) -> 'Settings':
        load_dotenv(ROOT / '.env')
        logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                            format='%(levelname)s %(name)s: %(message)s')
        # HTTP logs may contain request details; never log credentials or tokens.
        logging.getLogger('httpx').setLevel(logging.WARNING)
        ca_path = os.getenv('GIGACHAT_CA_BUNDLE_FILE', 'certs/russian_trusted_root_ca.pem').strip()
        value = cls(
            credentials=os.getenv('GIGACHAT_CREDENTIALS', '').strip(),
            chat_model=os.getenv('GIGACHAT_MODEL', 'GigaChat'),
            pdf_dir=ROOT / os.getenv('PDF_DIR', 'data/pdfs'),
            index_path=ROOT / os.getenv('INDEX_PATH', 'data/index.sqlite'),
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '150')),
            top_k=int(os.getenv('TOP_K', '5')),
            min_similarity=float(os.getenv('MIN_SIMILARITY', '0.25')),
            scope=os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS'),
            base_url=os.getenv('GIGACHAT_BASE_URL', 'https://api.giga.chat/v1').rstrip('/'),
            auth_url=os.getenv('GIGACHAT_AUTH_URL', 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'),
            ca_bundle_file=ROOT / ca_path if ca_path else None,
            timeout=float(os.getenv('GIGACHAT_TIMEOUT', '60')),
            max_upload_mb=int(os.getenv('MAX_UPLOAD_MB', '20')))
        if not 0 <= value.chunk_overlap < value.chunk_size <= 2000:
            raise ValueError('Нужно 0 <= CHUNK_OVERLAP < CHUNK_SIZE <= 2000')
        if value.top_k < 1 or not -1 <= value.min_similarity <= 1:
            raise ValueError('Неверные TOP_K или MIN_SIMILARITY')
        if value.timeout <= 0 or not 1 <= value.max_upload_mb <= 200:
            raise ValueError('Таймаут должен быть положительным; MAX_UPLOAD_MB от 1 до 200')
        if value.scope not in {'GIGACHAT_API_PERS', 'GIGACHAT_API_B2B', 'GIGACHAT_API_CORP'}:
            raise ValueError('Проверьте GIGACHAT_SCOPE в личном кабинете')
        for url in (value.base_url, value.auth_url):
            parsed = urlparse(url)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('Адреса GigaChat должны использовать HTTPS без логина и пароля в URL')
        return value
