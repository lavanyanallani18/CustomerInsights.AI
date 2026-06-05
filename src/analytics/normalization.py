"""Normalize SEC companyfacts into traceable, auditable metric records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SUPPORTED_FORMS = frozenset({"10-K", "10-Q"})
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")

# Earlier concepts take precedence when multiple concepts represent the same
# canonical metric for one period.
CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "debt_total": ("LongTermDebtAndFinanceLeaseObligations", "LongTermDebt"),
    "debt_current": (
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtCurrent",
    ),
    "debt_noncurrent": (
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebtNoncurrent",
    ),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
    ),
}
CONCEPT_TO_METRIC = {
    concept: metric for metric, concepts in CONCEPT_MAP.items() for concept in concepts
}
CONCEPT_PRIORITY = {
    concept: priority
    for _metric, concepts in CONCEPT_MAP.items()
    for priority, concept in enumerate(concepts)
}


class NormalizationError(ValueError):
    """Raised when companyfacts cannot be safely normalized."""


def _iso_date(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value in (None, "") and optional:
        return None
    if not isinstance(value, str):
        raise NormalizationError(f"{field} must be an ISO date string")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise NormalizationError(f"Invalid {field}: {value!r}") from exc


def _decimal_string(value: Any) -> str:
    if isinstance(value, bool):
        raise NormalizationError("Boolean is not a valid metric value")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise NormalizationError(f"Invalid metric value: {value!r}") from exc
    if not number.is_finite():
        raise NormalizationError("Metric value must be finite")
    return format(number, "f")


@dataclass(frozen=True)
class MetricRecord:
    """One normalized SEC fact with complete filing provenance."""

    record_id: str
    ticker: str
    cik: str
    metric: str
    concept: str
    taxonomy: str
    value: str
    unit: str
    period_start: str | None
    period_end: str
    fiscal_year: int | None
    fiscal_period: str | None
    accession: str
    form: str
    filed: str
    frame: str | None
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_id(fields: Iterable[Any]) -> str:
    material = json.dumps(list(fields), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def normalize_companyfacts(
    payload: dict[str, Any],
    *,
    ticker: str | None = None,
    selected_accessions: set[str] | None = None,
) -> list[MetricRecord]:
    """Return supported US-GAAP facts linked to selected filing accessions.

    Comparative periods reported in a selected filing are retained because
    companyfacts associates each comparison with that filing's accession.
    """
    if not isinstance(payload, dict):
        raise NormalizationError("companyfacts root must be an object")
    cik_raw = str(payload.get("cik", "")).strip()
    if not cik_raw.isdigit():
        raise NormalizationError("companyfacts cik must be numeric")
    cik = cik_raw.zfill(10)
    ticker_values = payload.get("tickers")
    if ticker is None:
        if not isinstance(ticker_values, list) or not ticker_values:
            raise NormalizationError("companyfacts must include a ticker or caller-supplied ticker")
        ticker = str(ticker_values[0]).upper()
    else:
        ticker = ticker.upper()
    facts = payload.get("facts", {}).get("us-gaap", {})
    if not isinstance(facts, dict):
        raise NormalizationError("companyfacts facts.us-gaap must be an object")

    records: list[MetricRecord] = []
    for concept, fact_data in facts.items():
        metric = CONCEPT_TO_METRIC.get(concept)
        if metric is None or not isinstance(fact_data, dict):
            continue
        units = fact_data.get("units", {})
        if not isinstance(units, dict):
            continue
        for unit, unit_facts in units.items():
            if not isinstance(unit, str) or not isinstance(unit_facts, list):
                continue
            for raw in unit_facts:
                if not isinstance(raw, dict) or raw.get("form") not in SUPPORTED_FORMS:
                    continue
                accession = str(raw.get("accn", ""))
                if not _ACCESSION_RE.fullmatch(accession):
                    continue
                if selected_accessions is not None and accession not in selected_accessions:
                    continue
                try:
                    period_start = _iso_date(raw.get("start"), "start", optional=True)
                    period_end = _iso_date(raw.get("end"), "end")
                    filed = _iso_date(raw.get("filed"), "filed")
                    value = _decimal_string(raw.get("val"))
                    fiscal_year = int(raw["fy"]) if raw.get("fy") is not None else None
                except (NormalizationError, TypeError, ValueError):
                    continue
                source_url = f"{SEC_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}"
                fields = (
                    ticker,
                    metric,
                    concept,
                    value,
                    unit,
                    period_start,
                    period_end,
                    accession,
                    raw["form"],
                    filed,
                    raw.get("frame"),
                )
                records.append(
                    MetricRecord(
                        record_id=_record_id(fields),
                        ticker=ticker,
                        cik=cik,
                        metric=metric,
                        concept=concept,
                        taxonomy="us-gaap",
                        value=value,
                        unit=unit,
                        period_start=period_start,
                        period_end=period_end or "",
                        fiscal_year=fiscal_year,
                        fiscal_period=str(raw["fp"]) if raw.get("fp") else None,
                        accession=accession,
                        form=str(raw["form"]),
                        filed=filed or "",
                        frame=str(raw["frame"]) if raw.get("frame") else None,
                        source_url=source_url,
                    )
                )
    return sorted(
        records,
        key=lambda item: (
            item.ticker,
            item.metric,
            item.period_end,
            item.period_start or "",
            item.filed,
            item.accession,
            CONCEPT_PRIORITY.get(item.concept, 99),
        ),
    )


def surface_anomalies(records: Iterable[MetricRecord]) -> list[dict[str, Any]]:
    """Surface exact duplicates, conflicting values, and later restatements."""
    groups: dict[tuple[Any, ...], list[MetricRecord]] = {}
    for record in records:
        key = (
            record.ticker,
            record.metric,
            record.unit,
            record.period_start,
            record.period_end,
            record.form,
            record.fiscal_period,
        )
        groups.setdefault(key, []).append(record)

    anomalies: list[dict[str, Any]] = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        distinct_values = {item.value for item in group}
        accessions = {item.accession for item in group}
        concepts = {item.concept for item in group}
        if len(accessions) > 1:
            kind = "restatement"
        elif len(distinct_values) == 1:
            kind = "duplicate"
        else:
            kind = "conflict"
        anomalies.append(
            {
                "kind": kind,
                "key": {
                    "ticker": key[0],
                    "metric": key[1],
                    "unit": key[2],
                    "period_start": key[3],
                    "period_end": key[4],
                    "form": key[5],
                    "fiscal_period": key[6],
                },
                "values": sorted(distinct_values, key=Decimal),
                "accessions": sorted(accessions),
                "concepts": sorted(concepts),
                "record_ids": sorted(item.record_id for item in group),
                "latest_filed": max(item.filed for item in group),
            }
        )
    return sorted(anomalies, key=lambda item: (item["key"]["ticker"], item["key"]["metric"], item["key"]["period_end"], item["kind"]))


def preferred_records(records: Iterable[MetricRecord]) -> list[MetricRecord]:
    """Choose one reproducible fact per metric/period, preserving anomalies separately."""
    groups: dict[tuple[Any, ...], list[MetricRecord]] = {}
    for record in records:
        key = (
            record.ticker,
            record.metric,
            record.unit,
            record.period_start,
            record.period_end,
            record.form,
            record.fiscal_period,
        )
        groups.setdefault(key, []).append(record)
    selected = []
    for group in groups.values():
        selected.append(
            max(
                group,
                key=lambda item: (
                    item.filed,
                    item.accession,
                    -CONCEPT_PRIORITY.get(item.concept, 99),
                ),
            )
        )
    return sorted(selected, key=lambda item: (item.ticker, item.period_end, item.metric))
