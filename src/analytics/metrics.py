"""Deterministic financial analytics with formulas and traceable inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Iterable

from .normalization import MetricRecord, preferred_records

FLOW_METRICS = frozenset(
    {
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "operating_cash_flow",
        "capital_expenditures",
    }
)


@dataclass(frozen=True)
class DerivedMetric:
    calculation_id: str
    ticker: str
    metric: str
    value: str
    unit: str
    period_start: str | None
    period_end: str
    formula: str
    inputs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["inputs"] = list(self.inputs)
        return result


def _calculation_id(fields: tuple[Any, ...]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _result(
    ticker: str,
    metric: str,
    value: Decimal,
    unit: str,
    period_start: str | None,
    period_end: str,
    formula: str,
    inputs: Iterable[MetricRecord],
) -> DerivedMetric:
    source_inputs = tuple(item.to_dict() for item in inputs)
    fields = (ticker, metric, str(value), unit, period_start, period_end, formula, source_inputs)
    return DerivedMetric(
        calculation_id=_calculation_id(fields),
        ticker=ticker,
        metric=metric,
        value=format(value, "f"),
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        formula=formula,
        inputs=source_inputs,
    )


def _period_key(record: MetricRecord) -> tuple[str, str | None, str, str]:
    return record.ticker, record.period_start, record.period_end, record.unit


def calculate_analytics(records: Iterable[MetricRecord]) -> list[DerivedMetric]:
    """Compute YoY growth, margins, debt-to-equity, and free cash flow."""
    facts = preferred_records(records)
    by_metric: dict[str, list[MetricRecord]] = {}
    for record in facts:
        by_metric.setdefault(record.metric, []).append(record)
    results: list[DerivedMetric] = []

    # Same-duration, approximately one-year-apart comparisons only.
    for metric in FLOW_METRICS:
        metric_records = by_metric.get(metric, [])
        for current in metric_records:
            if current.period_start is None:
                continue
            current_duration = (date.fromisoformat(current.period_end) - date.fromisoformat(current.period_start)).days
            candidates = []
            for prior in metric_records:
                if prior.ticker != current.ticker or prior.unit != current.unit or prior.period_start is None:
                    continue
                end_gap = (date.fromisoformat(current.period_end) - date.fromisoformat(prior.period_end)).days
                prior_duration = (date.fromisoformat(prior.period_end) - date.fromisoformat(prior.period_start)).days
                if 330 <= end_gap <= 400 and abs(current_duration - prior_duration) <= 7:
                    candidates.append(prior)
            if not candidates:
                continue
            prior = min(candidates, key=lambda item: abs((date.fromisoformat(current.period_end) - date.fromisoformat(item.period_end)).days - 365))
            prior_value = Decimal(prior.value)
            if prior_value == 0:
                continue
            value = (Decimal(current.value) - prior_value) / abs(prior_value)
            results.append(
                _result(
                    current.ticker,
                    f"{metric}_yoy_growth",
                    value,
                    "ratio",
                    current.period_start,
                    current.period_end,
                    "(current - prior_year) / abs(prior_year)",
                    (current, prior),
                )
            )

    period_index: dict[tuple[str, str | None, str, str], dict[str, MetricRecord]] = {}
    for record in facts:
        period_index.setdefault(_period_key(record), {})[record.metric] = record
    for (ticker, period_start, period_end, unit), period in period_index.items():
        revenue = period.get("revenue")
        if revenue and Decimal(revenue.value) != 0:
            for numerator_name, output_name in (
                ("gross_profit", "gross_margin"),
                ("operating_income", "operating_margin"),
                ("net_income", "net_margin"),
            ):
                numerator = period.get(numerator_name)
                if numerator:
                    results.append(
                        _result(
                            ticker,
                            output_name,
                            Decimal(numerator.value) / Decimal(revenue.value),
                            "ratio",
                            period_start,
                            period_end,
                            f"{numerator_name} / revenue",
                            (numerator, revenue),
                        )
                    )
        cash_flow = period.get("operating_cash_flow")
        capex = period.get("capital_expenditures")
        if cash_flow and capex:
            results.append(
                _result(
                    ticker,
                    "free_cash_flow",
                    Decimal(cash_flow.value) - Decimal(capex.value),
                    unit,
                    period_start,
                    period_end,
                    "operating_cash_flow - capital_expenditures",
                    (cash_flow, capex),
                )
            )

    instant_index: dict[tuple[str, str, str], dict[str, MetricRecord]] = {}
    for record in facts:
        if record.period_start is None:
            instant_index.setdefault((record.ticker, record.period_end, record.unit), {})[record.metric] = record
    for (ticker, period_end, _unit), period in instant_index.items():
        equity = period.get("equity")
        if not equity or Decimal(equity.value) == 0:
            continue
        debt_inputs: tuple[MetricRecord, ...]
        if period.get("debt_total"):
            debt_inputs = (period["debt_total"],)
            debt = Decimal(debt_inputs[0].value)
            formula = "total_debt / equity"
        elif period.get("debt_current") and period.get("debt_noncurrent"):
            debt_inputs = (period["debt_current"], period["debt_noncurrent"])
            debt = sum((Decimal(item.value) for item in debt_inputs), Decimal(0))
            formula = "(current_debt + noncurrent_debt) / equity"
        else:
            continue
        results.append(
            _result(
                ticker,
                "debt_to_equity",
                debt / Decimal(equity.value),
                "ratio",
                None,
                period_end,
                formula,
                (*debt_inputs, equity),
            )
        )
    return sorted(results, key=lambda item: (item.ticker, item.period_end, item.metric, item.calculation_id))
