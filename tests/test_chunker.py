import pytest
from rag.chunker import chunk_pages
from rag.models import Page

def test_overlap_page_boundaries_and_complete_coverage():
    pages = [Page('a.pdf', 2, 'abcdefghijklmnopqrst'), Page('a.pdf', 4, 'uvwxyz')]
    chunks = chunk_pages(pages, 8, 2)
    assert [c.text for c in chunks] == ['abcdefgh','ghijklmn','mnopqrst','uvwxyz']
    assert [c.page for c in chunks] == [2,2,2,4]
    assert len({c.id for c in chunks}) == 4
    assert chunks == chunk_pages(pages,8,2)

def test_empty_and_invalid():
    assert chunk_pages([Page('a',1,'')]) == []
    with pytest.raises(ValueError):
        chunk_pages([],8,8)
