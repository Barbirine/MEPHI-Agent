"""GigaChat REST contract tests; every HTTP request uses MockTransport."""
import json
import time
from urllib.parse import parse_qs
from uuid import UUID
import httpx
import numpy as np
import pytest
from rag.gigachat_client import GigaChatClient, GigaChatError
from rag.embeddings import GigaChatEmbedder
from rag.llm import GigaChatGenerator, REFUSAL
from rag.models import Chunk, Hit

HIT = Hit(Chunk('a','a.pdf',2,'Source text'),0.9)

def token_response():
    return httpx.Response(200,json={'access_token':'fake-token',
                                   'expires_at':int((time.time()+1800)*1000)})

def test_oauth_embeddings_chat_and_cached_token():
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path.endswith('/oauth'):
            assert request.headers['Authorization'] == 'Basic unit-test-only'
            assert UUID(request.headers['RqUID']).version == 4
            assert parse_qs(request.content.decode()) == {'scope':['GIGACHAT_API_PERS']}
            return token_response()
        assert request.headers['Authorization'] == 'Bearer fake-token'
        body = json.loads(request.content)
        if request.url.path.endswith('/embeddings'):
            assert body == {'model':'Embeddings','input':['a','b']}
            return httpx.Response(200,json={'data':[
                {'index':1,'embedding':[2.,1.]},{'index':0,'embedding':[1.,1.]}]})
        assert request.url.path == '/v1/chat/completions'
        assert body['model'] == 'GigaChat'
        assert body['max_tokens'] == 1800
        assert json.loads(body['messages'][1]['content'])['sources'][0]['page'] == 2
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop',
                                       'message':{'content':'Ответ [1]'}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = GigaChatClient('unit-test-only',http_client=http)
        embedder = GigaChatEmbedder(client)
        assert embedder.model == 'gigachat:Embeddings'
        assert embedder.embed(['a','b']).tolist() == [[1,1],[2,1]]
        assert GigaChatGenerator(client,'GigaChat').generate('вопрос',[HIT]) == 'Ответ [1]'
    assert len(requests) == 3  # One OAuth token shared by embeddings and generation.

def test_batching_and_bad_embedding_input():
    counts = []
    def handler(request):
        if request.url.path.endswith('/oauth'):
            return token_response()
        count = len(json.loads(request.content)['input'])
        counts.append(count)
        return httpx.Response(200,json={'data':[
            {'index':i,'embedding':[1.,2.]} for i in range(count)]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        embedder = GigaChatEmbedder(GigaChatClient('test',http_client=http))
        assert embedder.embed(['a']*33).shape == (33,2)
        with pytest.raises(ValueError):
            embedder.embed([''])
    assert counts == [16,16,1]

def test_refresh_on_401_is_bounded():
    calls = []
    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('/oauth'):
            return token_response()
        return httpx.Response(401,json={'message':'not logged'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GigaChatError,match='401'):
            GigaChatClient('test',http_client=http).post('embeddings',{})
    assert calls == ['/api/v2/oauth','/v1/embeddings']*2

def test_expired_token_is_renewed():
    calls = []
    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('/oauth'):
            return token_response()
        return httpx.Response(200,json={'data':[]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = GigaChatClient('test',http_client=http)
        client.post('embeddings',{})
        client._expires_at = time.time()-1
        client.post('embeddings',{})
    assert calls.count('/api/v2/oauth') == 2

@pytest.mark.parametrize('status',[400,401,402,403,404,422,429,500])
def test_safe_oauth_errors(status, caplog):
    secret = 'do-not-leak-this-key'
    def handler(request):
        return httpx.Response(status,json={'message':secret})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GigaChatError) as error:
            GigaChatClient(secret,http_client=http).post('embeddings',{})
    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert str(status) in str(error.value)

def test_no_automatic_retry_after_network_error():
    calls = []
    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('/oauth'):
            return token_response()
        raise httpx.ReadTimeout('timeout')
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GigaChatError,match='автоматического повтора нет'):
            GigaChatClient('test',http_client=http).post('embeddings',{})
    assert len(calls) == 2

@pytest.mark.parametrize('payload',[{'data':[]},{'data':[{'index':0,'embedding':[0,0]}]},
                                  {'data':[{'index':0,'embedding':[float('inf'),1]}]}])
def test_invalid_embedding_response(payload):
    # Use a fake adapter here so even non-JSON float infinity can be validated.
    class Adapter:
        def post(self, endpoint, body):
            return payload
    with pytest.raises(GigaChatError):
        GigaChatEmbedder(Adapter()).embed(['a'])

def test_generation_length_and_no_context():
    class Adapter:
        calls = 0
        def post(self, endpoint, body):
            self.calls += 1
            return {'choices':[{'finish_reason':'length','message':{'content':'partial'}}]}
    adapter = Adapter()
    generator = GigaChatGenerator(adapter,'GigaChat')
    assert generator.generate('q',[]) == REFUSAL
    assert adapter.calls == 0
    with pytest.raises(GigaChatError,match='обрезан'):
        generator.generate('q',[HIT])

def test_tls_bundle_and_missing_key():
    from config import Settings
    client = GigaChatClient('test',ca_bundle_file=Settings.load().ca_bundle_file)
    context = client._ssl_context()
    import ssl
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    with pytest.raises(ValueError,match='GIGACHAT_CREDENTIALS'):
        GigaChatClient('')
