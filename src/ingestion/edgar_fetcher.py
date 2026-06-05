"""SEC EDGAR ingestion with traceable, offline-friendly caches.

The public helpers retain the original module's simple interface, while
``EdgarClient`` provides the stricter behavior used by ``scripts/prepare_data``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DATA_BASE = "https://data.sec.gov"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
ALLOWED_SEC_HOSTS = {"data.sec.gov", "www.sec.gov"}
DEFAULT_FORMS = ("10-K", "10-Q")
DEFAULT_TOTAL_FILINGS = 15
DEFAULT_REQUEST_INTERVAL = 0.12
DEFAULT_USER_AGENT = "CustomerInsights-AI research@customerinsights.ai"

COMPANIES = {
    "NVDA": {"cik": "0001045810", "name": "NVIDIA Corporation"},
    "AMD": {"cik": "0000002488", "name": "Advanced Micro Devices, Inc."},
    "INTC": {"cik": "0000050863", "name": "Intel Corporation"},
}

_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_CIK_RE = re.compile(r"^\d{1,10}$")
_SAFE_DOC_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_USER_AGENT_CONTACT_RE = re.compile(r"(@|https?://)", re.IGNORECASE)


class EdgarError(RuntimeError):
    """Base exception for SEC ingestion failures."""


class EdgarValidationError(EdgarError, ValueError):
    """Raised when unsafe or invalid SEC inputs are supplied."""


class OfflineDataUnavailable(EdgarError, FileNotFoundError):
    """Raised when offline mode is requested but a cache is absent."""


@dataclass(frozen=True)
class Filing:
    ticker: str
    company_name: str
    cik: str
    accession: str
    form: str
    filed: str
    report_date: str
    primary_document: str
    source_url: str
    local_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_user_agent(user_agent: str) -> str:
    value = user_agent.strip()
    if len(value) < 8 or len(value) > 256 or "\n" in value or "\r" in value:
        raise EdgarValidationError("SEC User-Agent must be 8-256 characters on one line")
    if not _USER_AGENT_CONTACT_RE.search(value):
        raise EdgarValidationError("SEC User-Agent must identify the requester and include contact information")
    return value


def _normalize_cik(cik: str | int) -> str:
    value = str(cik).strip().lstrip("0") or "0"
    if not _CIK_RE.fullmatch(value):
        raise EdgarValidationError(f"Invalid CIK: {cik!r}")
    return value.zfill(10)


def _validate_accession(accession: str) -> str:
    value = accession.strip()
    if not _ACCESSION_RE.fullmatch(value):
        raise EdgarValidationError(f"Invalid accession number: {accession!r}")
    return value


def _safe_primary_document(document: str) -> str:
    value = document.strip()
    if not value or not _SAFE_DOC_RE.fullmatch(value) or value in {".", ".."}:
        raise EdgarValidationError(f"Unsafe primary document name: {document!r}")
    return value


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_bytes_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


class EdgarClient:
    """A small SEC-compliant client with retries, pacing, and JSON caching."""

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        timeout: float = 30.0,
        request_interval: float = DEFAULT_REQUEST_INTERVAL,
        retries: int = 4,
        session: requests.Session | None = None,
    ) -> None:
        if timeout <= 0 or request_interval < 0 or retries < 0:
            raise EdgarValidationError("timeout must be positive; request_interval and retries cannot be negative")
        self.user_agent = _validate_user_agent(
            user_agent or os.getenv("SEC_USER_AGENT", DEFAULT_USER_AGENT)
        )
        self.timeout = timeout
        self.request_interval = request_interval
        self.session = session or requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            }
        )
        self._last_request = 0.0
        self._request_lock = threading.Lock()

    def _get(self, url: str) -> requests.Response:
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SEC_HOSTS:
            raise EdgarValidationError(f"Refusing non-SEC URL: {url}")
        with self._request_lock:
            wait = self.request_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            response = self.session.get(url, timeout=self.timeout)
            self._last_request = time.monotonic()
        response.raise_for_status()
        return response

    def get_json(self, url: str, cache_path: Path | None = None, *, offline: bool = False) -> dict[str, Any]:
        if cache_path and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(cached, dict):
                    raise ValueError("root is not an object")
                return cached
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                if offline:
                    raise OfflineDataUnavailable(f"Invalid offline cache {cache_path}: {exc}") from exc
                logger.warning("Ignoring invalid cache %s: %s", cache_path, exc)
        if offline:
            raise OfflineDataUnavailable(f"Offline cache not found: {cache_path or url}")
        response = self._get(url)
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise EdgarError(f"SEC returned invalid JSON for {url}") from exc
        if not isinstance(payload, dict):
            raise EdgarError(f"SEC returned a non-object JSON response for {url}")
        if cache_path:
            _atomic_json_write(cache_path, payload)
        return payload

    def submissions(self, cik: str | int, cache_dir: Path, *, offline: bool = False) -> dict[str, Any]:
        normalized = _normalize_cik(cik)
        return self.get_json(
            f"{DATA_BASE}/submissions/CIK{normalized}.json",
            cache_dir / "submissions" / f"CIK{normalized}.json",
            offline=offline,
        )

    def companyfacts(self, cik: str | int, cache_dir: Path, *, offline: bool = False) -> dict[str, Any]:
        normalized = _normalize_cik(cik)
        return self.get_json(
            f"{DATA_BASE}/api/xbrl/companyfacts/CIK{normalized}.json",
            cache_dir / "companyfacts" / f"CIK{normalized}.json",
            offline=offline,
        )

    def download_filing(self, filing: Filing, destination: Path, *, offline: bool = False) -> Path:
        if destination.is_file():
            return destination
        if offline:
            raise OfflineDataUnavailable(f"Offline filing not found: {destination}")
        _validate_accession(filing.accession)
        document = _safe_primary_document(filing.primary_document)
        response = self._get(filing.source_url)
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and "text" not in content_type and not response.content.lstrip().startswith(b"<"):
            raise EdgarError(f"Unexpected filing content type {content_type!r} for {filing.source_url}")
        _atomic_bytes_write(destination, response.content)
        return destination


def _column(recent: dict[str, Any], name: str, index: int, default: str = "") -> str:
    values = recent.get(name, [])
    if not isinstance(values, list) or index >= len(values):
        return default
    return str(values[index] or default)


def select_recent_filings(
    submissions: dict[str, Any],
    ticker: str,
    *,
    limit: int = 5,
    forms: Iterable[str] = DEFAULT_FORMS,
) -> list[Filing]:
    """Select recent non-amended filings and reject malformed SEC metadata."""
    ticker = ticker.upper()
    if ticker not in COMPANIES:
        raise EdgarValidationError(f"Unsupported ticker: {ticker}")
    if limit <= 0:
        raise EdgarValidationError("limit must be positive")
    allowed_forms = frozenset(forms)
    if not allowed_forms or not allowed_forms.issubset(DEFAULT_FORMS):
        raise EdgarValidationError(f"Forms must be a non-empty subset of {DEFAULT_FORMS}")
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict) or not isinstance(recent.get("form"), list):
        raise EdgarError(f"Malformed submissions response for {ticker}")

    company = COMPANIES[ticker]
    cik = _normalize_cik(company["cik"])
    selected: list[Filing] = []
    seen: set[str] = set()
    for index, form in enumerate(recent["form"]):
        if form not in allowed_forms:
            continue
        accession = _validate_accession(_column(recent, "accessionNumber", index))
        if accession in seen:
            continue
        primary_document = _safe_primary_document(_column(recent, "primaryDocument", index))
        accession_compact = accession.replace("-", "")
        source_url = (
            f"{ARCHIVES_BASE}/{int(cik)}/{accession_compact}/{quote(primary_document, safe='._-')}"
        )
        selected.append(
            Filing(
                ticker=ticker,
                company_name=company["name"],
                cik=cik,
                accession=accession,
                form=str(form),
                filed=_column(recent, "filingDate", index),
                report_date=_column(recent, "reportDate", index),
                primary_document=primary_document,
                source_url=source_url,
            )
        )
        seen.add(accession)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise EdgarError(f"Expected {limit} recent filings for {ticker}, found {len(selected)}")
    return selected


def prepare_sec_data(
    data_dir: str | Path = "data/sec",
    *,
    user_agent: str | None = None,
    offline: bool = False,
    download_filings: bool = True,
    client: EdgarClient | None = None,
) -> dict[str, Any]:
    """Prepare exactly 15 filings plus companyfacts for NVDA, AMD, and INTC."""
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sec = client or EdgarClient(user_agent)
    filings: list[dict[str, Any]] = []
    companies: dict[str, dict[str, Any]] = {}

    per_company = DEFAULT_TOTAL_FILINGS // len(COMPANIES)
    for ticker, company in COMPANIES.items():
        submissions = sec.submissions(company["cik"], root / "raw", offline=offline)
        sec.companyfacts(company["cik"], root / "raw", offline=offline)
        selected = select_recent_filings(submissions, ticker, limit=per_company)
        company_filings: list[dict[str, Any]] = []
        for filing in selected:
            record = filing.to_dict()
            if download_filings:
                suffix = Path(filing.primary_document).suffix.lower() or ".html"
                destination = root / "filings" / ticker / f"{filing.accession}{suffix}"
                local_path = sec.download_filing(filing, destination, offline=offline)
                record["local_path"] = str(local_path)
            company_filings.append(record)
            filings.append(record)
        companies[ticker] = {**company, "filings": company_filings}

    if len(filings) != DEFAULT_TOTAL_FILINGS:
        raise EdgarError(f"Expected exactly {DEFAULT_TOTAL_FILINGS} filings, got {len(filings)}")
    manifest = {
        "schema_version": 1,
        "filing_count": len(filings),
        "forms": list(DEFAULT_FORMS),
        "selection": "five most-recent non-amended 10-K/10-Q filings per company",
        "companies": companies,
        "filings": filings,
    }
    _atomic_json_write(root / "manifest.json", manifest)
    return manifest


# Compatibility helpers used by the original project code.
def get_company_filings(cik: str, filing_type: str, max_results: int = 4) -> list[dict[str, str]]:
    normalized = _normalize_cik(cik)
    ticker = next((key for key, value in COMPANIES.items() if value["cik"] == normalized), None)
    if ticker is None:
        raise EdgarValidationError(f"Unsupported CIK: {cik}")
    submissions = EdgarClient().get_json(f"{DATA_BASE}/submissions/CIK{normalized}.json")
    return [
        {
            "accession": filing.accession.replace("-", ""),
            "accession_fmt": filing.accession,
            "date": filing.filed,
            "primary_doc": filing.primary_document,
            "source_url": filing.source_url,
        }
        for filing in select_recent_filings(submissions, ticker, limit=max_results, forms=(filing_type,))
    ]


def download_filing_text(cik: str, accession: str, primary_doc: str, save_path: Path) -> str | None:
    normalized = _normalize_cik(cik)
    accession_fmt = _validate_accession(accession) if "-" in accession else (
        f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    )
    document = _safe_primary_document(primary_doc)
    url = f"{ARCHIVES_BASE}/{int(normalized)}/{accession_fmt.replace('-', '')}/{quote(document, safe='._-')}"
    response = EdgarClient()._get(url)
    _atomic_bytes_write(Path(save_path), response.content)
    return response.text


def fetch_all_filings(data_dir: str = "./data/filings") -> dict[str, Any]:
    """Compatibility entry point; now prepares exactly the required 15 filings."""
    return prepare_sec_data(data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_sec_data()
