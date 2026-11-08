import fitz
import pytest
from rag.pdf_parser import parse_pdf

def test_page_numbers_include_blank(tmp_path):
    path = tmp_path / 'test.pdf'
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.new_page().insert_text((50,50), 'Second page')
        pdf.save(path)
    pages = parse_pdf(path)
    assert [p.page for p in pages] == [1,2]
    assert pages[0].text == ''
    assert pages[1].text == 'Second page'

def test_scan_and_corrupt(tmp_path):
    empty = tmp_path / 'empty.pdf'
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(empty)
    with pytest.raises(ValueError, match='OCR'):
        parse_pdf(empty)
    bad = tmp_path / 'bad.pdf'
    bad.write_bytes(b'broken pdf')
    with pytest.raises(ValueError):
        parse_pdf(bad)
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / 'missing.pdf')
