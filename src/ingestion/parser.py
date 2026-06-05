"""Filing parser: converts HTML SEC filings into structured chunks with metadata."""
import re
import logging
import json
from pathlib import Path

try:
    import html2text
except ImportError:  # pragma: no cover - exercised only in lean runtimes.
    html2text = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - requirements include bs4, fallback remains regex-only.
    BeautifulSoup = None

logger = logging.getLogger(__name__)

# Key financial sections in 10-K/10-Q
SECTION_PATTERNS = {
    "business":           r"item\s*1[.\s]+business",
    "risk_factors":       r"item\s*1a[.\s]+risk\s*factors",
    "mda":                r"item\s*7[.\s]+management.{0,30}discussion",
    "quantitative_risk":  r"item\s*7a[.\s]+quantitative",
    "financial_statements": r"item\s*8[.\s]+financial\s*statements",
    "results_of_operations": r"results\s*of\s*operations",
    "liquidity":          r"liquidity\s*and\s*capital",
    "revenue":            r"revenue|net\s*sales",
    "income":             r"income\s*from\s*operations|operating\s*income",
    "balance_sheet":      r"balance\s*sheet|financial\s*position",
    "cash_flow":          r"cash\s*flow",
}


def html_to_text(html_content: str) -> str:
    """Convert HTML to clean plain text."""
    if html2text is not None:
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.body_width = 0
        text = converter.handle(html_content)
    elif BeautifulSoup is not None:
        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    else:
        text = re.sub(r"<[^>]+>", " ", html_content)
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def detect_section(text_chunk: str) -> str:
    """Detect which section of a filing a chunk belongs to."""
    text_lower = text_chunk.lower()
    for section_name, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text_lower):
            return section_name
    return "general"


def extract_financial_numbers(text: str) -> list[dict]:
    """Extract financial figures with context from text."""
    numbers = []
    # Match patterns like "$1.2 billion", "2,345 million", "$12.5M"
    patterns = [
        r'\$\s*([\d,]+\.?\d*)\s*(billion|million|thousand|B|M|K)\b',
        r'([\d,]+\.?\d*)\s*(billion|million|thousand)\s+(?:in\s+)?(?:revenue|sales|income|loss|expense)',
        r'\(\s*([\d,]+\.?\d*)\s*\)',  # negative numbers in parentheses
    ]
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            numbers.append({
                "text": match.group(0),
                "start": match.start(),
                "context": text[max(0, match.start()-100):match.end()+100],
            })
    return numbers


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Try to break at paragraph boundary
        if end < text_len:
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def parse_filing(
    file_path: str,
    ticker: str,
    filing_type: str,
    filing_date: str,
    accession: str,
) -> list[dict]:
    """
    Parse a single SEC filing into chunks with rich metadata.
    Returns list of document dicts ready for embedding.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    content = path.read_text(encoding="utf-8", errors="ignore")

    # Convert HTML to text
    if content.strip().startswith("<") or "<html" in content[:500].lower():
        text = html_to_text(content)
    else:
        text = content

    # Extract year from date
    year = filing_date[:4] if filing_date else "unknown"
    chunks = chunk_text(text, chunk_size=1200, overlap=250)
    documents = []

    for i, chunk in enumerate(chunks):
        section = detect_section(chunk)
        doc = {
            "id": f"{ticker}_{filing_type}_{filing_date}_{i}",
            "content": chunk,
            "metadata": {
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "year": year,
                # Filing date does not reliably identify fiscal quarter.
                "quarter": "",
                "accession": accession,
                "section": section,
                "chunk_index": i,
                "source_file": str(path.name),
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={filing_type}",
            },
        }
        documents.append(doc)

    logger.info(f"Parsed {ticker} {filing_type} {filing_date}: {len(chunks)} chunks")
    return documents


def parse_all_filings(manifest_path: str = "./data/filings/manifest.json") -> list[dict]:
    """Parse all filings listed in the manifest."""
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return []

    manifest = json.loads(manifest_file.read_text())
    all_documents = []

    if isinstance(manifest.get("filings"), list):
        filing_items = []
        for filing in manifest["filings"]:
            filing_items.append(
                (
                    filing.get("ticker", ""),
                    {
                        "file": filing.get("local_path"),
                        "type": filing.get("form"),
                        "date": filing.get("filed") or filing.get("report_date"),
                        "accession": filing.get("accession", ""),
                        "source_url": filing.get("source_url", ""),
                    },
                )
            )
    else:
        filing_items = [
            (ticker, filing)
            for ticker, info in manifest.items()
            if isinstance(info, dict)
            for filing in info.get("filings", [])
        ]

    for ticker, filing in filing_items:
        if not filing.get("file"):
            continue
        docs = parse_filing(
            file_path=filing["file"],
            ticker=ticker,
            filing_type=filing["type"],
            filing_date=filing["date"],
            accession=filing.get("accession", ""),
        )
        for doc in docs:
            if filing.get("source_url"):
                doc["metadata"]["url"] = filing["source_url"]
        all_documents.extend(docs)

    logger.info(f"Total documents parsed: {len(all_documents)}")
    return all_documents
