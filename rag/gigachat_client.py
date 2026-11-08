"""Small GigaChat REST adapter: OAuth, token refresh, TLS and safe errors.

No network calls on import or initialization. No automatic retry of billable
requests after ambiguous network errors (they may already have been processed).
"""
from contextlib import nullcontext
import logging
from pathlib import Path
import ssl
import time
from typing import Any
from uuid import uuid4
import certifi
import httpx

log = logging.getLogger(__name__)

class GigaChatError(RuntimeError):
    """An actionable error without secrets or raw API response bodies."""

class GigaChatClient:
    def __init__(self, credentials: str, scope: str = 'GIGACHAT_API_PERS',
                 base_url: str = 'https://api.giga.chat/v1',
                 auth_url: str = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
                 ca_bundle_file: Path | None = None, timeout: float = 60,
                 http_client: httpx.Client | None = None) -> None:
        if not credentials.strip():
            raise ValueError('Заполните GIGACHAT_CREDENTIALS в .env: нужен Authorization key.')
        self._credentials = credentials.strip().removeprefix('Basic ').strip()
        self.scope = scope
        self.base_url = base_url.rstrip('/')
        self.auth_url = auth_url
        self.ca_bundle_file = ca_bundle_file
        self.timeout = timeout
        self._http_client = http_client
        self._token = ''
        self._expires_at = 0.0

    def _ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(cafile=certifi.where())
        if self.ca_bundle_file:
            try:
                context.load_verify_locations(cafile=str(self.ca_bundle_file))
            except (OSError, ssl.SSLError) as exc:
                raise GigaChatError('Не удалось прочитать GIGACHAT_CA_BUNDLE_FILE. '
                                    'Проверьте путь к сертификату PEM.') from exc
        return context

    def _send(self, url: str, **kwargs: Any) -> httpx.Response:
        manager = (nullcontext(self._http_client) if self._http_client is not None else
                   httpx.Client(verify=self._ssl_context(), timeout=self.timeout,
                                follow_redirects=False))
        try:
            with manager as client:
                return client.post(url, **kwargs)
        except httpx.TimeoutException as exc:
            raise GigaChatError('GigaChat не ответил вовремя. Запрос мог быть обработан; '
                                'автоматического повтора нет.') from exc
        except httpx.HTTPError as exc:
            raise GigaChatError('Не удалось соединиться с GigaChat. Проверьте сеть, VPN '
                                'и сертификат GIGACHAT_CA_BUNDLE_FILE. '
                                'Проверка TLS включена.') from exc

    @staticmethod
    def _check(response: httpx.Response, operation: str) -> None:
        if 200 <= response.status_code < 300:
            return
        status = response.status_code
        hint = {
            400: 'Проверьте параметры запроса и scope.',
            401: 'Проверьте Authorization key в GIGACHAT_CREDENTIALS и scope.',
            402: 'Проверьте баланс и тариф в личном кабинете.',
            403: 'Проверьте доступ к API, модели и тарифу в личном кабинете.',
            404: 'Проверьте GIGACHAT_BASE_URL и название модели.',
            413: 'Слишком большой запрос. Уменьшите размер фрагментов.',
            422: 'Проверьте модель, размер контекста и параметры запроса.',
            429: 'Достигнут лимит запросов или токенов. Проверьте квоту.'
        }.get(status, 'Сервис вернул ошибку. Повторите действие позже.')
        log.warning('GigaChat %s: HTTP %s', operation, status)
        raise GigaChatError(f'GigaChat: {operation}, HTTP {status}. {hint}')

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise GigaChatError('GigaChat вернул некорректный JSON.') from exc
        if not isinstance(payload, dict):
            raise GigaChatError('GigaChat вернул неожиданный формат ответа.')
        return payload

    def _get_token(self) -> str:
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        response = self._send(self.auth_url,
            headers={'Authorization': f'Basic {self._credentials}',
                     'RqUID': str(uuid4()), 'Accept': 'application/json'},
            data={'scope': self.scope})
        self._check(response, 'авторизация')
        payload = self._json(response)
        token = payload.get('access_token')
        expires = payload.get('expires_at')
        if not isinstance(token, str) or not token or not isinstance(expires, (int, float)):
            raise GigaChatError('В ответе авторизации нет токена или срока действия.')
        # GigaChat normally returns milliseconds since Unix epoch.
        expiry = expires / 1000 if expires > 100_000_000_000 else float(expires)
        if not time.time() < expiry:
            raise GigaChatError('Получен просроченный токен. Проверьте часы компьютера.')
        self._token, self._expires_at = token, expiry
        log.info('Токен GigaChat получен; содержимое не логируется')
        return token

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if endpoint not in {'embeddings', 'chat/completions'}:
            raise ValueError('Неизвестный метод GigaChat')
        response = self._send(f'{self.base_url}/{endpoint}',
                              headers={'Authorization': f'Bearer {self._get_token()}'},
                              json=payload)
        if response.status_code == 401:
            # A rejected token was not accepted for generation; refresh once only.
            self._token, self._expires_at = '', 0
            response = self._send(f'{self.base_url}/{endpoint}',
                                  headers={'Authorization': f'Bearer {self._get_token()}'},
                                  json=payload)
        self._check(response, endpoint)
        return self._json(response)
