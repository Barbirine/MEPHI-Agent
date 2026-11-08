from streamlit.testing.v1 import AppTest
from config import ROOT

def test_app_initial_render(monkeypatch):
    monkeypatch.delenv('GIGACHAT_CREDENTIALS',raising=False)
    app = AppTest.from_file(str(ROOT/'app.py')).run()
    assert not app.exception
    assert app.title[0].value == '📚 MIFI-Agent'
    assert app.text_area[0].label == 'Ваш вопрос'

def test_app_missing_key_reports_error(monkeypatch):
    monkeypatch.setenv('GIGACHAT_CREDENTIALS','')
    app = AppTest.from_file(str(ROOT/'app.py')).run()
    app.text_area[0].set_value('Что такое lum?')
    next(button for button in app.button if button.label == 'Найти ответ').click().run()
    assert not app.exception
    assert app.error
    assert 'GIGACHAT_CREDENTIALS' in app.error[0].value

def test_upload_index_question_flow_with_mock_http(tmp_path, monkeypatch):
    """Complete UI path using a real PDF and mocked GigaChat HTTP only."""
    import time
    import fitz
    import httpx
    import streamlit as st
    import rag.pipeline as module
    from rag.gigachat_client import GigaChatClient
    with fitz.open() as pdf:
        pdf.new_page().insert_text((50,50),'A lum is a fictional unit of stored light.')
        content = pdf.tobytes()
    class Upload:
        name = 'lecture.pdf'
        def getvalue(self):
            return content
    monkeypatch.setattr(st,'file_uploader',lambda *args, **kwargs: [Upload()])
    monkeypatch.setenv('GIGACHAT_CREDENTIALS', '')
    monkeypatch.setenv('PDF_DIR',str(tmp_path/'pdfs'))
    monkeypatch.setenv('INDEX_PATH',str(tmp_path/'index.sqlite'))
    requests = []
    def handler(request):
        requests.append(request.url.path)
        if request.url.path.endswith('/oauth'):
            return httpx.Response(200,json={'access_token':'test-token',
                                           'expires_at':int((time.time()+1800)*1000)})
        assert request.url.path.endswith('/chat/completions')
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop',
            'message':{'content':'Lum — вымышленная единица накопленного света [1].'}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        monkeypatch.setattr(module,'GigaChatClient',
                            lambda **kwargs: GigaChatClient(**kwargs,http_client=http))
        app = AppTest.from_file(str(ROOT/'app.py')).run()
        next(b for b in app.button if b.label == 'Сохранить PDF').click().run()
        assert (tmp_path/'pdfs/lecture.pdf').is_file()
        assert requests == []  # Saving PDFs never contacts GigaChat.
        next(b for b in app.button if b.label == 'Индексировать PDF').click().run()
        assert not app.exception
        assert not app.error
        assert (tmp_path/'index.sqlite').is_file()
        assert requests == []  # Indexing works without a key or network.
        monkeypatch.setenv('GIGACHAT_CREDENTIALS', 'unit-test-only')
        app.text_area[0].set_value('Что такое lum?')
        next(b for b in app.button if b.label == 'Найти ответ').click().run()
        assert not app.exception
        assert not app.error
        assert any('едининица' in m.value or 'единица' in m.value for m in app.markdown)
        assert 'lecture.pdf' in app.expander[0].label
        assert 'PDF-страница 1' in app.expander[0].label
    assert requests == ['/api/v2/oauth', '/v1/chat/completions']
