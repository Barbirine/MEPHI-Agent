import fitz
import pytest
from rag.uploads import save_uploaded_pdf
from rag.pdf_parser import parse_pdf

def pdf_bytes(text='Example text'):
    with fitz.open() as pdf:
        page = pdf.new_page()
        if text:
            page.insert_text((50,50),text)
        return pdf.tobytes()

def test_upload_is_local_validated_and_safe(tmp_path):
    content = pdf_bytes()
    path = save_uploaded_pdf('../../lecture.pdf',content,tmp_path)
    assert path == tmp_path/'lecture.pdf'
    assert parse_pdf(path)[0].text == 'Example text'
    assert save_uploaded_pdf('lecture.pdf',content,tmp_path) == path
    new = save_uploaded_pdf('lecture.pdf',pdf_bytes('Different'),tmp_path)
    assert new != path
    assert path.read_bytes() == content
    assert not list(tmp_path.glob('*.tmp'))

def test_bad_upload_never_persists(tmp_path):
    for name, data in [('bad.pdf',b'%PDF-broken'),('empty.pdf',b''),
                       ('not.pdf',b'not a pdf'),('file.txt',pdf_bytes()),
                       ('scan.pdf',pdf_bytes(''))]:
        with pytest.raises(ValueError):
            save_uploaded_pdf(name,data,tmp_path)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match='лимит'):
        save_uploaded_pdf('large.pdf',pdf_bytes(),tmp_path,max_size_bytes=1)
