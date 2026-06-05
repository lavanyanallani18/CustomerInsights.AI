from src.ingestion.parser import detect_section, extract_financial_numbers, html_to_text


def test_html_to_text_removes_markup_and_keeps_content():
    text = html_to_text("<html><body><h1>Revenue</h1><p>Net sales were $12 million.</p></body></html>")

    assert "<h1>" not in text
    assert "Revenue" in text
    assert "Net sales were $12 million." in text


def test_detect_section_recognizes_financial_sections():
    assert detect_section("Item 7. Management's Discussion and Analysis") == "mda"
    assert detect_section("Liquidity and Capital Resources") == "liquidity"
    assert detect_section("Unrelated introductory material") == "general"


def test_extract_financial_numbers_returns_context_and_offsets():
    text = "For the year, revenue increased to $1.2 billion from $900 million."

    figures = extract_financial_numbers(text)

    assert [figure["text"] for figure in figures] == ["$1.2 billion", "$900 million"]
    assert all(text[figure["start"] :].startswith(figure["text"]) for figure in figures)
    assert all("revenue" in figure["context"] for figure in figures)
