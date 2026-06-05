"""Build, validate, and load the prepared SEC analytics dataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.ingestion.edgar_fetcher import COMPANIES, DEFAULT_TOTAL_FILINGS, EdgarClient, prepare_sec_data

from .metrics import calculate_analytics
from .normalization import MetricRecord, normalize_companyfacts, surface_anomalies

PREPARED_FILENAME = "prepared_analytics.json"


class DatasetValidationError(ValueError):
    """Raised when a prepared analytics dataset violates its contract."""


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def validate_prepared_dataset(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DatasetValidationError("Unsupported or missing prepared dataset schema_version")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("filing_count") != DEFAULT_TOTAL_FILINGS:
        raise DatasetValidationError(f"Prepared manifest must contain exactly {DEFAULT_TOTAL_FILINGS} filings")
    filings = manifest.get("filings")
    if not isinstance(filings, list) or len(filings) != DEFAULT_TOTAL_FILINGS:
        raise DatasetValidationError("Prepared manifest filing list has an invalid length")
    counts = {ticker: 0 for ticker in COMPANIES}
    accessions: set[str] = set()
    for filing in filings:
        if not isinstance(filing, dict) or filing.get("ticker") not in counts:
            raise DatasetValidationError("Prepared manifest contains an unsupported company")
        if filing.get("form") not in {"10-K", "10-Q"}:
            raise DatasetValidationError("Prepared manifest contains an unsupported form")
        accession = filing.get("accession")
        if not isinstance(accession, str) or accession in accessions:
            raise DatasetValidationError("Prepared manifest contains a missing or duplicate accession")
        accessions.add(accession)
        counts[filing["ticker"]] += 1
    if set(counts.values()) != {5}:
        raise DatasetValidationError("Prepared manifest must contain five filings per company")
    for field in ("records", "anomalies", "analytics"):
        if not isinstance(payload.get(field), list):
            raise DatasetValidationError(f"Prepared dataset field {field!r} must be a list")
    required_record_fields = {
        "record_id",
        "ticker",
        "metric",
        "value",
        "unit",
        "period_end",
        "accession",
        "form",
        "filed",
        "source_url",
    }
    for record in payload["records"]:
        if not isinstance(record, dict) or not required_record_fields.issubset(record):
            raise DatasetValidationError("Prepared dataset contains a malformed metric record")
        if record["accession"] not in accessions:
            raise DatasetValidationError("Metric record references an accession outside the manifest")


def load_prepared_dataset(data_dir: str | Path = "data/sec") -> dict[str, Any]:
    path = Path(data_dir).expanduser().resolve() / PREPARED_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"Prepared dataset not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot read prepared dataset {path}: {exc}") from exc
    validate_prepared_dataset(payload)
    return payload


def build_prepared_dataset(
    data_dir: str | Path = "data/sec",
    *,
    user_agent: str | None = None,
    offline: bool = False,
    download_filings: bool = True,
    client: EdgarClient | None = None,
) -> dict[str, Any]:
    root = Path(data_dir).expanduser().resolve()
    manifest = prepare_sec_data(
        root,
        user_agent=user_agent,
        offline=offline,
        download_filings=download_filings,
        client=client,
    )
    selected_accessions = {filing["accession"] for filing in manifest["filings"]}
    records: list[MetricRecord] = []
    for ticker, company in COMPANIES.items():
        companyfacts_path = root / "raw" / "companyfacts" / f"CIK{company['cik']}.json"
        try:
            companyfacts = json.loads(companyfacts_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetValidationError(f"Cannot read companyfacts cache for {ticker}: {exc}") from exc
        records.extend(
            normalize_companyfacts(
                companyfacts,
                ticker=ticker,
                selected_accessions=selected_accessions,
            )
        )
    anomalies = surface_anomalies(records)
    analytics = calculate_analytics(records)
    payload = {
        "schema_version": 1,
        "manifest": manifest,
        "records": [record.to_dict() for record in records],
        "anomalies": anomalies,
        "analytics": [metric.to_dict() for metric in analytics],
        "counts": {
            "filings": len(manifest["filings"]),
            "records": len(records),
            "anomalies": len(anomalies),
            "analytics": len(analytics),
        },
    }
    validate_prepared_dataset(payload)
    _atomic_write(root / PREPARED_FILENAME, payload)
    return payload
