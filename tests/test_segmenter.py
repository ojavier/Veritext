from ingest.cleaner import clean_segments, clean_text
from ingest.segmenter import segment_text


def test_paragraphs_and_offsets():
    text = "Primer párrafo.\n\n\n  Segundo párrafo.\nsigue aquí.\n\nTercero."
    segs = segment_text(text)
    assert len(segs) == 3
    assert [s.id for s in segs] == [0, 1, 2]
    for s in segs:
        assert text[s.start : s.end] == s.text


def test_empty_and_whitespace():
    assert segment_text("") == []
    assert segment_text(" \n\n \n") == []


def test_long_paragraph_split_by_sentences():
    text = " ".join(f"Oración número {i} del texto." for i in range(100))
    segs = segment_text(text, max_length=200)
    assert len(segs) > 1
    for s in segs:
        assert len(s.text) <= 200
        assert text[s.start : s.end] == s.text
        assert s.text.endswith(".")


def test_long_unbroken_word_hard_split():
    text = "a" * 250
    segs = segment_text(text, max_length=100)
    assert [len(s.text) for s in segs] == [100, 100, 50]


def test_cleaner():
    assert clean_text("Hola​  mundo  \x00ok") == "Hola mundo ok"
    assert clean_text("línea uno\nlínea dos") == "línea uno línea dos"
    assert clean_text("pala-\nbra") == "palabra"
    assert clean_text("café") == "café"


def test_clean_segments_keeps_offsets_and_drops_empty():
    text = "Hola​ mundo.\n\n​\n\nAdiós."
    segs = clean_segments(segment_text(text))
    assert [s.text for s in segs] == ["Hola mundo.", "Adiós."]
    assert segs[0].start == 0
