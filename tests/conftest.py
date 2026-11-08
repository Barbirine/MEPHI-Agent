"""The test suite must never contact a live model API."""
import httpx
import pytest

@pytest.fixture(autouse=True)
def block_live_http(monkeypatch):
    def blocked(self, request):
        raise AssertionError('Live HTTP disabled in tests; use httpx.MockTransport')
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', blocked)
