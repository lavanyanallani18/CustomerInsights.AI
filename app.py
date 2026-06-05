"""Comparative SEC financial dashboard for NVDA, AMD, and INTC.

The presentation layer deliberately treats analytics and Q&A as optional
adapters. That keeps the app useful while data is being prepared and prevents
missing dependencies from turning into a blank Streamlit page.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
TARGET_TICKERS = ("NVDA", "AMD", "INTC")
COMPANY_NAMES = {
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
}
COLORS = {"NVDA": "#76B900", "AMD": "#ED1C24", "INTC": "#00A3F6"}
METRIC_ALIASES = {
    "sales": "Revenue",
    "net sales": "Revenue",
    "total revenue": "Revenue",
    "revenues": "Revenue",
    "revenue": "Revenue",
    "gross profit": "Gross profit",
    "operating income": "Operating income",
    "income from operations": "Operating income",
    "net income": "Net income",
    "net earnings": "Net income",
    "cash and cash equivalents": "Cash and cash equivalents",
    "total assets": "Total assets",
    "total liabilities": "Total liabilities",
    "research and development": "R&D expense",
    "r&d": "R&D expense",
    "free cash flow": "Free cash flow",
}
JSON_CANDIDATES = (
    "data/sec/prepared_analytics.json",
    "data/prepared/metrics.json",
    "data/prepared_metrics.json",
    "data/metrics.json",
    "prepared/metrics.json",
    "outputs/metrics.json",
    "artifacts/metrics.json",
    "evaluation/results.json",
)
SECRET_PATTERN = re.compile(
    r"(api[_-]?key|secret|token|password|authorization|bearer|sk-[a-z0-9_-]+)",
    re.IGNORECASE,
)


st.set_page_config(
    page_title="Silicon Signals | SEC comparison",
    page_icon="▰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #17212b;
          --muted: #62717d;
          --paper: #f7f5f0;
          --line: rgba(23, 33, 43, .12);
          --accent: #ff5a36;
        }
        .stApp {
          background:
            radial-gradient(circle at 87% 2%, rgba(255,90,54,.10), transparent 23rem),
            linear-gradient(180deg, #fcfbf8 0%, var(--paper) 100%);
        }
        [data-testid="stSidebar"] { background: #17212b; }
        [data-testid="stSidebar"] * { color: #f7f5f0; }
        [data-testid="stSidebar"] .stCaption { color: #b7c0c7; }
        .block-container { max-width: 1450px; padding-top: 2rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
        h1 { font-size: clamp(2.4rem, 5vw, 5rem) !important; line-height: .95 !important; }
        [data-testid="stMetric"] {
          background: rgba(255,255,255,.72);
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: .9rem 1rem;
          box-shadow: 0 8px 30px rgba(23,33,43,.04);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        .eyebrow {
          color: var(--accent); font-weight: 800; letter-spacing: .14em;
          text-transform: uppercase; font-size: .75rem; margin-bottom: .5rem;
        }
        .lede { color: var(--muted); font-size: 1.1rem; max-width: 62rem; }
        .panel {
          background: rgba(255,255,255,.72); border: 1px solid var(--line);
          border-radius: 16px; padding: 1rem 1.1rem; margin: .4rem 0 1rem;
        }
        .status-dot {
          display: inline-block; width: .55rem; height: .55rem;
          border-radius: 100%; margin-right: .45rem; background: #76b900;
        }
        .status-dot.warn { background: #f4a261; }
        .status-dot.off { background: #9aa5ad; }
        .source-card {
          border-left: 3px solid #ff5a36; padding: .25rem 0 .25rem .9rem;
          margin: .75rem 0; color: var(--ink);
        }
        .small-muted { color: var(--muted); font-size: .84rem; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
        .stTabs [data-baseweb="tab-list"] { gap: .25rem; }
        .stTabs [data-baseweb="tab"] { padding: .65rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_message(message: Any) -> str:
    """Return a useful UI error without leaking credentials or long internals."""
    text = str(message).replace(str(ROOT), ".")
    if SECRET_PATTERN.search(text):
        return "A protected configuration value or provider credential needs attention."
    return text[:240]


def canonical_metric(value: Any) -> str:
    text = str(value or "Unknown metric").strip()
    if text.lower() in METRIC_ALIASES:
        return METRIC_ALIASES[text.lower()]
    return text.replace("_", " ").replace("-", " ").strip().title()


def numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip().replace(",", "").replace("$", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    match = re.fullmatch(r"(-?[\d.]+)\s*([kmbt])?", text, re.IGNORECASE)
    if not match:
        return None
    result = float(match.group(1))
    result *= {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(
        (match.group(2) or "").lower(), 1
    )
    return -result if negative else result


def first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    lower = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key.lower() in lower and lower[key.lower()] not in (None, ""):
            return lower[key.lower()]
    return default


def looks_like_metric(record: Mapping[str, Any]) -> bool:
    keys = {str(key).lower() for key in record}
    has_ticker = bool(keys & {"ticker", "symbol", "company"})
    has_value = bool(keys & {"value", "amount", "metric_value", "reported_value"})
    has_metric = bool(keys & {"metric", "metric_name", "name", "concept", "label"})
    return has_ticker and has_value and has_metric


def collect_metric_records(node: Any, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Recursively turn common prepared JSON layouts into metric records."""
    context = dict(context or {})
    records: list[dict[str, Any]] = []
    if isinstance(node, list):
        for item in node:
            records.extend(collect_metric_records(item, context))
        return records
    if not isinstance(node, Mapping):
        return records
    if looks_like_metric(node):
        records.append({**context, **node})
        return records

    structural = {"metrics", "records", "data", "companies", "results", "facts", "financials"}
    for key, value in node.items():
        next_context = dict(context)
        key_text = str(key)
        if key_text.upper() in TARGET_TICKERS:
            next_context["ticker"] = key_text.upper()
        elif re.fullmatch(r"(19|20)\d{2}", key_text):
            next_context["fiscal_year"] = key_text
        elif key_text.lower() not in structural and isinstance(value, (int, float, str)):
            possible = numeric(value)
            if possible is not None and context.get("ticker"):
                records.append({**context, "metric": key_text, "value": value})
                continue
        records.extend(collect_metric_records(value, next_context))
    return records


def normalize_metrics(payload: Any) -> pd.DataFrame:
    records = collect_metric_records(payload)
    normalized: list[dict[str, Any]] = []
    for record in records:
        ticker = str(first(record, "ticker", "symbol", "company", default="")).upper()
        if ticker not in TARGET_TICKERS:
            continue
        value = numeric(first(record, "value", "amount", "metric_value", "reported_value"))
        if value is None:
            continue
        period = first(
            record,
            "period_end",
            "end_date",
            "date",
            "as_of",
            "fiscal_year",
            "year",
            "period",
            "fiscal_period",
            "filing_date",
            default="Unknown",
        )
        inputs = first(record, "inputs", default=[])
        first_input = inputs[0] if isinstance(inputs, list) and inputs and isinstance(inputs[0], Mapping) else {}
        source_url = str(first(record, "source_url", "url", "source", default=""))
        if not source_url and first_input:
            source_url = str(first(first_input, "source_url", "url", "source", default=""))
        accession = str(first(record, "accession", "accession_number", default=""))
        if not accession and first_input:
            accession = str(first(first_input, "accession", "accession_number", default=""))
        form = str(first(record, "form", "filing_type", default=""))
        if not form and first_input:
            form = str(first(first_input, "form", "filing_type", default=""))
        filed = str(first(record, "filed", "filing_date", default=""))
        if not filed and first_input:
            filed = str(first(first_input, "filed", "filing_date", default=""))
        normalized.append(
            {
                "ticker": ticker,
                "company": COMPANY_NAMES[ticker],
                "metric": canonical_metric(
                    first(record, "metric", "metric_name", "concept", "label", "name")
                ),
                "period": str(period),
                "value": value,
                "unit": str(first(record, "unit", "units", "currency", default="USD")),
                "form": form,
                "filed": filed,
                "accession": accession,
                "source_url": source_url,
                "source_label": str(
                    first(record, "source_label", "source_file", "origin", default="")
                ),
                "input": str(first(record, "input", "formula", "calculation", default="Reported")),
                "quality": str(first(record, "quality", "status", "confidence", default="")),
            }
        )
    if not normalized:
        return pd.DataFrame(
            columns=[
                "ticker", "company", "metric", "period", "value", "unit", "form",
                "filed", "accession", "source_url", "source_label", "input", "quality",
            ]
        )
    frame = pd.DataFrame(normalized).drop_duplicates()
    return frame.sort_values(["metric", "period", "ticker"], kind="stable")


def collect_named_lists(payload: Any, names: set[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            found.extend(collect_named_lists(item, names))
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in names and isinstance(value, list):
                found.extend(item for item in value if isinstance(item, Mapping))
            else:
                found.extend(collect_named_lists(value, names))
    return found


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover_json() -> list[Path]:
    paths = [ROOT / name for name in JSON_CANDIDATES if (ROOT / name).is_file()]
    for base in ("prepared", "outputs", "artifacts", "evaluation"):
        directory = ROOT / base
        if directory.exists():
            paths.extend(
                path
                for path in directory.rglob("*.json")
                if path.name != "manifest.json" and "filing" not in path.parts
            )
    return list(dict.fromkeys(paths))


def compatible_call(function: Any, **available: Any) -> Any:
    signature = inspect.signature(function)
    kwargs = {
        name: available[name]
        for name, parameter in signature.parameters.items()
        if name in available
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name not in kwargs
    ]
    if required:
        raise TypeError(f"Adapter requires unsupported inputs: {', '.join(required)}")
    return function(**kwargs)


@st.cache_data(show_spinner=False)
def load_prepared_data() -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    payloads: list[Any] = []
    notes: list[str] = []
    loaded_prepared_dataset = False
    for path in discover_json():
        try:
            payloads.append(load_json(path))
            notes.append(f"Loaded `{path.relative_to(ROOT)}`")
            if path.resolve() == (ROOT / "data/sec/prepared_analytics.json").resolve():
                loaded_prepared_dataset = True
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"Skipped `{path.relative_to(ROOT)}`: {safe_message(exc)}")

    # Optional zero-argument analytics adapters.
    if not loaded_prepared_dataset:
        try:
            dataset = importlib.import_module("src.analytics.dataset")
            loader = getattr(dataset, "load_prepared_dataset", None)
            if callable(loader):
                payloads.append(loader())
                notes.append("Loaded `src.analytics.dataset.load_prepared_dataset()`")
        except Exception as exc:
            notes.append(f"Prepared analytics dataset unavailable: {safe_message(exc)}")

    for module_name in ("src.analytics.metrics", "src.metrics.metrics"):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        for function_name in ("get_metrics_data", "load_metrics", "get_metrics"):
            function = getattr(module, function_name, None)
            if callable(function):
                try:
                    payloads.append(compatible_call(function))
                    notes.append(f"Loaded `{module_name}.{function_name}()`")
                except Exception as exc:  # Adapter failures must not sink the UI.
                    notes.append(f"Adapter `{module_name}.{function_name}` unavailable: {safe_message(exc)}")
                break

    metrics = normalize_metrics(payloads)
    conflicts = collect_named_lists(
        payloads, {"conflicts", "discrepancies", "quality_issues", "anomalies"}
    )
    evaluations = collect_named_lists(payloads, {"cases", "evaluation", "evaluations", "eval_results", "scores"})
    return metrics, conflicts, evaluations, notes


def format_value(value: float, unit: str = "USD") -> str:
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    if unit.lower() in {"%", "percent", "percentage"}:
        return f"{value:,.1f}%"
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if absolute >= threshold:
            prefix = "$" if unit.upper() == "USD" else ""
            return f"{sign}{prefix}{absolute / threshold:,.2f}{suffix}"
    prefix = "$" if unit.upper() == "USD" else ""
    return f"{sign}{prefix}{absolute:,.2f}"


def latest_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    subset = frame[frame["metric"] == metric].copy()
    if subset.empty:
        return subset
    subset["_period_sort"] = subset["period"].str.extract(r"((?:19|20)\d{2})", expand=False).fillna(
        subset["period"]
    )
    subset = subset.sort_values(
        ["ticker", "_period_sort", "filed", "accession", "value"],
        kind="stable",
    )
    return subset.groupby("ticker", as_index=False).tail(1).drop(columns=["_period_sort"])


def one_row_per_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest authoritative row when a filing repeats a fact."""
    if frame.empty:
        return frame
    deduped = frame.copy()
    deduped["_filed_sort"] = deduped["filed"].fillna("").astype(str)
    deduped["_accession_sort"] = deduped["accession"].fillna("").astype(str)
    deduped = deduped.sort_values(
        ["ticker", "_filed_sort", "_accession_sort", "value"],
        kind="stable",
    )
    return deduped.groupby("ticker", as_index=False).tail(1).drop(
        columns=["_filed_sort", "_accession_sort"]
    )


def source_link(row: Mapping[str, Any]) -> str:
    url = str(row.get("source_url") or "")
    if url.startswith(("https://www.sec.gov/", "https://data.sec.gov/")):
        return url
    accession = str(row.get("accession") or "").replace("-", "")
    ticker = str(row.get("ticker") or "")
    cik = {"NVDA": "1045810", "AMD": "2488", "INTC": "50863"}.get(ticker)
    if cik and accession:
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
    if cik:
        return f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude"
    return ""


def readiness(metrics: pd.DataFrame, notes: list[str]) -> dict[str, Any]:
    present = set(metrics["ticker"]) if not metrics.empty else set()
    cited = (
        metrics.apply(lambda row: bool(source_link(row)), axis=1).mean() * 100
        if not metrics.empty
        else 0
    )
    return {
        "companies": len(present),
        "metrics": metrics["metric"].nunique() if not metrics.empty else 0,
        "rows": len(metrics),
        "citation_rate": cited,
        "notes": notes,
    }


def empty_state() -> None:
    st.markdown(
        """
        <div class="panel">
          <div class="eyebrow">Prepared data not found</div>
          <b>The dashboard shell is healthy; the evidence layer is not prepared yet.</b>
          <p class="small-muted">
          Add a prepared metrics JSON file or expose a supported analytics adapter.
          No placeholder financial values are shown because that would blur the line
          between reported facts and demo content.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "Expected JSON example:\n"
        '{"metrics":[{"ticker":"NVDA","metric":"Revenue","period":"2025",'
        '"value":123000000,"unit":"USD","form":"10-K","source_url":"https://www.sec.gov/..."}]}',
        language="json",
    )


def render_sidebar(metrics: pd.DataFrame, notes: list[str]) -> tuple[list[str], str | None]:
    status = readiness(metrics, notes)
    st.sidebar.markdown("## Silicon Signals")
    st.sidebar.caption("Evidence-first comparative financial analysis")
    available = [ticker for ticker in TARGET_TICKERS if ticker in set(metrics.get("ticker", []))]
    selected = st.sidebar.multiselect(
        "Companies",
        TARGET_TICKERS,
        default=available or list(TARGET_TICKERS),
        format_func=lambda ticker: f"{ticker} · {COMPANY_NAMES[ticker]}",
    )
    metrics_list = sorted(metrics["metric"].unique()) if not metrics.empty else []
    selected_metric = st.sidebar.selectbox(
        "Focus metric",
        metrics_list,
        index=0 if metrics_list else None,
        placeholder="Waiting for prepared data",
    )
    st.sidebar.markdown("---")
    dot = "" if status["rows"] else " off"
    st.sidebar.markdown(
        f'<span class="status-dot{dot}"></span><b>{status["rows"]:,}</b> normalized observations',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        f'{status["companies"]}/3 companies · {status["metrics"]} metrics · '
        f'{status["citation_rate"]:.0f}% source-linked'
    )
    with st.sidebar.expander("Data readiness"):
        st.write("Prepared inputs are read-only. The app does not fetch SEC data on page load.")
        for note in notes or ["No compatible prepared JSON or analytics adapter discovered."]:
            st.caption(note)
    st.sidebar.markdown("---")
    st.sidebar.caption("Public SEC filings only · Not investment advice")
    return selected, selected_metric


def render_overview(frame: pd.DataFrame, selected_metric: str | None) -> None:
    st.subheader("Decision snapshot")
    if frame.empty or not selected_metric:
        empty_state()
        return
    latest = latest_rows(frame, selected_metric)
    columns = st.columns(3)
    by_ticker = {row["ticker"]: row for _, row in latest.iterrows()}
    for column, ticker in zip(columns, TARGET_TICKERS):
        with column:
            row = by_ticker.get(ticker)
            if row is None:
                st.metric(f"{ticker} · {selected_metric}", "Not available")
                st.caption("No comparable observation in the current selection.")
            else:
                st.metric(
                    f"{ticker} · {selected_metric}",
                    format_value(row["value"], row["unit"]),
                    help=f"Latest available period in prepared data: {row['period']}",
                )
                st.caption(f"{row['period']} · {row['form'] or 'form not tagged'}")

    chart = frame[frame["metric"] == selected_metric].copy()
    if not chart.empty:
        figure = px.line(
            chart,
            x="period",
            y="value",
            color="ticker",
            markers=True,
            color_discrete_map=COLORS,
            custom_data=["company", "unit", "form", "filed"],
        )
        figure.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{x}<br>%{y:,.2f} %{customdata[1]}"
                "<br>%{customdata[2]} · filed %{customdata[3]}<extra></extra>"
            )
        )
        figure.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=30, b=10),
            legend_title_text="",
            yaxis_title=selected_metric,
            xaxis_title="Reported period",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.45)",
        )
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})


def render_comparison(frame: pd.DataFrame) -> None:
    st.subheader("Comparable metrics")
    st.caption("Choose a metric and compare one source-traceable observation per company.")
    if frame.empty:
        empty_state()
        return
    left, right = st.columns([2, 1])
    metric = left.selectbox("Metric", sorted(frame["metric"].unique()), key="comparison_metric")
    metric_frame = frame[frame["metric"] == metric]
    periods = ["Latest available", *sorted(metric_frame["period"].unique(), reverse=True)]
    period = right.selectbox("Period", periods, key="comparison_period")
    if period == "Latest available":
        comparison = latest_rows(metric_frame, metric)
    else:
        comparison = one_row_per_ticker(metric_frame[metric_frame["period"] == period])
    comparison = comparison.sort_values("value", ascending=False)
    if comparison["ticker"].nunique() < len(set(frame["ticker"])):
        st.info(
            "This selection does not have every selected company on the same exact period. "
            "Use `Latest available` for the broadest peer comparison."
        )
    figure = px.bar(
        comparison,
        x="ticker",
        y="value",
        color="ticker",
        color_discrete_map=COLORS,
        text_auto=".3s",
        custom_data=["company", "unit", "form", "filed"],
    )
    figure.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{y:,.2f} %{customdata[1]}"
        "<br>%{customdata[2]} · filed %{customdata[3]}<extra></extra>"
    )
    figure.update_layout(
        showlegend=False,
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title=metric,
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.45)",
    )
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
    st.dataframe(
        comparison[["ticker", "metric", "period", "value", "unit", "form", "filed"]],
        hide_index=True,
        use_container_width=True,
        column_config={"value": st.column_config.NumberColumn(format="%.2f")},
    )


def render_traceability(frame: pd.DataFrame) -> None:
    st.subheader("Metric source & input traceability")
    st.caption("Every displayed number should be reproducible from a public filing or an explicit calculation.")
    if frame.empty:
        empty_state()
        return
    trace = frame.copy()
    trace["source"] = trace.apply(source_link, axis=1)
    trace["source_status"] = trace["source"].map(lambda value: "Linked" if value else "Missing")
    selected = st.dataframe(
        trace[
            [
                "ticker", "metric", "period", "value", "unit", "input", "form",
                "filed", "accession", "source_status", "source",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "value": st.column_config.NumberColumn(format="%.2f"),
            "source": st.column_config.LinkColumn(display_text="Open SEC source"),
        },
        on_select="rerun",
        selection_mode="single-row",
        key="trace_table",
    )
    rows = selected.selection.rows
    if rows:
        row = trace.iloc[rows[0]]
        st.markdown(
            f'<div class="source-card"><b>{row["ticker"]} · {row["metric"]} · {row["period"]}</b>'
            f'<br><span class="small-muted">Input/calculation: {row["input"] or "Reported"} · '
            f'Form: {row["form"] or "not tagged"} · Accession: {row["accession"] or "not tagged"}</span></div>',
            unsafe_allow_html=True,
        )


def normalize_answer(result: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(result, str):
        return result, [], []
    if not isinstance(result, Mapping):
        return str(result), [], []
    answer = str(first(result, "answer", "response", "content", default="No answer returned."))
    citations = first(result, "citations", "sources", "docs", "documents", default=[])
    conflicts = first(result, "conflicts", "discrepancies", default=[])
    return (
        answer,
        list(citations) if isinstance(citations, Iterable) and not isinstance(citations, (str, bytes)) else [],
        list(conflicts) if isinstance(conflicts, list) else [],
    )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _metric_value(value: float, metric: str) -> str:
    lowered = metric.lower()
    if "growth" in lowered or "margin" in lowered:
        return _pct(value)
    if "debt to equity" in lowered:
        return f"{value:.2f}x"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"{value:,.2f}"


def _period_label(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "nan", "unknown"}:
        return "latest prepared period"
    return text


def _select_analytics_metric(question: str, available_metrics: set[str]) -> str | None:
    lowered = question.lower()
    if any(term in lowered for term in ("grow", "grown", "growth", "outperform", "faster")):
        if "gross" in lowered and "profit" in lowered:
            return "Gross Profit Yoy Growth"
        if "operating" in lowered and "income" in lowered:
            return "Operating Income Yoy Growth"
        if "net" in lowered and "income" in lowered:
            return "Net Income Yoy Growth"
        if "cash" in lowered:
            return "Operating Cash Flow Yoy Growth"
        return "Revenue Yoy Growth"
    if "margin" in lowered:
        if "gross" in lowered:
            return "Gross Margin"
        if "net" in lowered:
            return "Net Margin"
        if "operating" in lowered:
            return "Operating Margin"
        return "Operating Margin" if "Operating Margin" in available_metrics else "Gross Margin"
    if "free cash flow" in lowered or "fcf" in lowered:
        return "Free Cash Flow"
    if "debt" in lowered and "equity" in lowered:
        return "Debt To Equity"
    return None


def answer_from_analytics(
    question: str,
    frame: pd.DataFrame,
    ticker: str | None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Answer common comparison questions from computed metrics before RAG."""
    if frame.empty:
        return None

    metric = _select_analytics_metric(question, set(frame["metric"].dropna().unique()))
    if not metric:
        return None

    candidates = frame[frame["metric"].eq(metric)].copy()
    if ticker:
        candidates = candidates[candidates["ticker"].eq(ticker)]
    if candidates.empty:
        return None

    latest = (
        candidates.sort_values(["ticker", "period", "value"], kind="stable")
        .drop_duplicates(["ticker"], keep="last")
        .sort_values("value", ascending=False, kind="stable")
    )
    if latest.empty:
        return None

    leader = latest.iloc[0]
    metric_label = metric.replace(" Yoy Growth", " year-over-year growth").lower()
    comparisons = [
        f"{row.ticker}: {_metric_value(float(row.value), metric)} for {_period_label(row.period)} [{index}]"
        for index, row in enumerate(latest.itertuples(), start=1)
    ]
    values = [float(row.value) for row in latest.itertuples()]
    spread = values[0] - values[-1] if len(values) > 1 else 0.0
    answer_lines = [
        (
            f"**Short answer:** {leader['ticker']} shows the strongest latest available "
            f"{metric_label}: {_metric_value(float(leader['value']), metric)} for "
            f"{_period_label(leader['period'])} [1]."
        ),
        "",
        f"**Comparison:** {'; '.join(comparisons)}.",
    ]
    if len(values) > 1:
        answer_lines.append(
            f"**Magnitude:** {leader['ticker']} is ahead of the lowest peer by {_metric_value(spread, metric)} on this metric."
        )
    if "Yoy Growth" in metric:
        method = f"`{metric}` uses `(current - prior_year) / abs(prior_year)` from SEC companyfacts inputs."
    elif "Margin" in metric:
        method = f"`{metric}` is calculated from SEC companyfacts income-statement inputs as profit divided by revenue."
    elif metric == "Free Cash Flow":
        method = "`Free Cash Flow` is operating cash flow minus capital expenditures from SEC companyfacts inputs."
    elif metric == "Debt To Equity":
        method = "`Debt To Equity` is total debt divided by stockholders' equity from SEC companyfacts inputs."
    else:
        method = f"`{metric}` comes from the prepared SEC-derived analytics dataset."
    answer_lines.extend([
        f"**How this was calculated:** {method}",
        "**Caveat:** This uses each company's latest available comparable period in the prepared data, so fiscal period dates may differ slightly across companies.",
    ])
    answer = "\n\n".join(answer_lines)

    citations = []
    for index, row in enumerate(latest.to_dict("records"), start=1):
        citations.append(
            {
                "id": index,
                "ticker": row["ticker"],
                "metric": row["metric"],
                "period": row["period"],
                "source_url": row.get("source_url", ""),
                "quote": (
                    f"{row['ticker']} {row['metric']} = {_metric_value(float(row['value']), metric)} "
                    f"for {_period_label(row['period'])}; formula: {row.get('input') or 'reported/computed metric'}."
                ),
                "metadata": row,
            }
        )
    return answer, citations, []


@st.cache_resource(show_spinner=False)
def ensure_rag_index(
    persist_dir: str = "chroma_db",
    manifest_path: str = "data/sec/manifest.json",
) -> tuple[bool, str]:
    """Build the local Chroma index from SEC filings if it is missing or empty."""
    try:
        from src.ingestion.parser import parse_all_filings
        from src.rag.vectorstore import VectorStore

        store = VectorStore(persist_dir=str(ROOT / persist_dir))
        existing = store.collection.count()
        if existing > 0:
            return True, f"RAG index ready with {existing:,} filing chunks."

        manifest = ROOT / manifest_path
        if not manifest.exists():
            return False, f"RAG index is empty and `{manifest_path}` was not found."

        documents = parse_all_filings(str(manifest))
        if not documents:
            return False, "RAG index is empty and no filing chunks could be parsed from the SEC manifest."

        store.clear()
        added = store.add_documents(documents, batch_size=100)
        total = store.collection.count()
        if total <= 0:
            return False, "RAG index rebuild finished but no chunks were stored."
        return True, f"RAG index rebuilt with {total:,} filing chunks from {added:,} new chunks."
    except Exception as exc:
        return False, f"RAG index could not be prepared: {safe_message(exc)}"


def contextualize_question(question: str, history: list[dict[str, Any]]) -> str:
    """Give lightweight context to follow-up questions without hiding the user text."""
    if not history:
        return question
    lowered = question.lower()
    followup_markers = (
        "what about", "how about", "why", "explain", "compare", "that",
        "it", "they", "their", "same", "more detail", "tell me more",
    )
    if not any(marker in lowered for marker in followup_markers):
        return question
    previous = history[-1]
    return (
        f"Follow-up question: {question}\n\n"
        f"Previous question: {previous.get('question', '')}\n"
        f"Previous answer: {previous.get('answer', '')[:1200]}"
    )


def append_chat_turn(
    question: str,
    ticker: str | None,
    answer: str,
    citations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    history = st.session_state.setdefault("qa_history", [])
    history.append(
        {
            "question": question,
            "ticker": ticker,
            "answer": answer,
            "citations": citations,
            "conflicts": conflicts,
        }
    )
    st.session_state["qa_history"] = history[-10:]


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.warning("No citations were returned. Treat the answer as unsupported.")
        return
    st.markdown(f"**Citations · {len(citations)} source item(s)**")
    for index, citation in enumerate(citations, 1):
        label, url, excerpt = citation_fields(citation)
        with st.expander(f"[{index}] {label}"):
            if url.startswith(("https://www.sec.gov/", "https://data.sec.gov/")):
                st.link_button("Open SEC source", url)
            st.write(excerpt or "Citation metadata returned without an excerpt.")


def render_chat_turn(turn: Mapping[str, Any]) -> None:
    with st.chat_message("user"):
        suffix = f"  \n_Filter: {turn['ticker']}_" if turn.get("ticker") else ""
        st.markdown(f"{turn.get('question', '')}{suffix}")
    with st.chat_message("assistant"):
        st.markdown(str(turn.get("answer", "")))
        render_citations(list(turn.get("citations", [])))
        conflicts = turn.get("conflicts", [])
        if conflicts:
            st.warning(f"The retrieval layer flagged {len(conflicts)} possible conflict(s). Review Data quality.")


def set_pending_question(question: str) -> None:
    st.session_state["qa_pending_question"] = question


def render_suggested_questions(disabled: bool) -> None:
    suggestions = [
        "Which company has grown more?",
        "Compare operating margin across the three companies.",
        "What did Intel say about revenue compared with 2024?",
        "What changed in AMD net revenue?",
    ]
    cols = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        cols[index % 2].button(
            suggestion,
            key=f"qa_suggestion_{index}",
            disabled=disabled,
            on_click=set_pending_question,
            args=(suggestion,),
            use_container_width=True,
        )


def run_qa(
    question: str,
    ticker: str | None,
    metrics: pd.DataFrame,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    contextual_question = contextualize_question(question, history or [])
    analytics_answer = answer_from_analytics(contextual_question, metrics, ticker)
    if analytics_answer is not None:
        return analytics_answer

    index_ready, index_message = ensure_rag_index()
    if not index_ready:
        return index_message, [], []

    try:
        module = importlib.import_module("src.rag.qa")
    except (ImportError, ModuleNotFoundError):
        return (
            "The RAG Cited Q&A adapter is not available yet. Prepare the RAG index and expose "
            "`src.rag.qa.ask_question(...)` or `answer_question(...)` to enable answers.",
            [],
            [],
        )
    for name in ("ask_question", "answer_question", "query", "answer"):
        function = getattr(module, name, None)
        if callable(function):
            try:
                result = compatible_call(
                    function,
                    question=contextual_question,
                    query=contextual_question,
                    ticker=ticker,
                    ticker_filter=ticker,
                )
                answer, citations, conflicts = normalize_answer(result)
                if citations:
                    return answer, citations, conflicts
                return f"{answer}\n\n_Index status: {index_message}_", citations, conflicts
            except Exception as exc:
                return f"Q&A could not complete: {safe_message(exc)}", [], []
    return "The Q&A module loaded but does not expose a compatible answer function.", [], []


def citation_fields(citation: Any) -> tuple[str, str, str]:
    if not isinstance(citation, Mapping):
        return str(citation), "", ""
    metadata = citation.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    merged = {**metadata, **citation}
    label = " · ".join(
        filter(
            None,
            [
                str(first(merged, "ticker", default="")),
                str(first(merged, "filing_type", "form", default="")),
                str(first(merged, "filing_date", "filed", "year", default="")),
            ],
        )
    )
    url = str(first(merged, "url", "source_url", default=""))
    excerpt = str(first(merged, "quote", "content", "excerpt", "text", default=""))
    return label or "Retrieved SEC evidence", url, excerpt[:600]


def render_qa(frame: pd.DataFrame) -> None:
    st.subheader("Ask the filings")
    st.caption("Chat with the SEC evidence. The app keeps the latest 10 turns and shows citations under each answer.")
    history = st.session_state.setdefault("qa_history", [])
    top_left, top_right = st.columns([2, 1])
    ticker_choice = top_left.selectbox("Optional company filter", ["All", *TARGET_TICKERS], key="qa_ticker_filter")
    if top_right.button("Clear chat", use_container_width=True):
        st.session_state["qa_history"] = []
        st.rerun()

    remaining = max(0, 10 - len(history))
    st.caption(f"{remaining} question(s) remaining in this chat window.")
    render_suggested_questions(disabled=remaining == 0)

    for turn in history:
        render_chat_turn(turn)

    prompt = st.chat_input(
        "Ask a source-cited question or a follow-up...",
        disabled=remaining == 0,
    )
    pending = st.session_state.pop("qa_pending_question", None)
    question = (prompt or pending or "").strip()
    if question and remaining > 0:
        ticker = None if ticker_choice == "All" else ticker_choice
        with st.spinner("Retrieving evidence and preparing a cited answer..."):
            answer, citations, conflicts = run_qa(question, ticker, frame, history)
        append_chat_turn(question, ticker, answer, citations, conflicts)
        st.rerun()
    elif remaining == 0:
        st.info("This chat has reached 10 questions. Clear chat to start a new one.")


def issue_summary(issue: Mapping[str, Any]) -> tuple[str, str]:
    title = str(first(issue, "title", "key", "metric", "type", default="Potential discrepancy"))
    detail = str(first(issue, "note", "message", "description", "reason", default=json.dumps(issue, default=str)))
    return title, detail


def render_quality(frame: pd.DataFrame, conflicts: list[dict[str, Any]]) -> None:
    st.subheader("Conflicts & data quality")
    missing_sources = frame[~frame.apply(lambda row: bool(source_link(row)), axis=1)] if not frame.empty else frame
    duplicates = (
        frame[frame.duplicated(["ticker", "metric", "period"], keep=False)]
        if not frame.empty
        else frame
    )
    a, b, c = st.columns(3)
    a.metric("Detected conflicts", len(conflicts))
    b.metric("Rows missing SEC links", len(missing_sources))
    c.metric("Duplicate comparison keys", len(duplicates))
    st.caption(
        "A conflict is a review signal, not proof of an error: amendments, fiscal calendars, "
        "units, and annual-versus-quarterly contexts can all create legitimate differences."
    )
    if conflicts:
        for issue in conflicts:
            title, detail = issue_summary(issue)
            with st.expander(title):
                st.write(detail)
                st.json(issue)
    else:
        st.success("No prepared conflict records were found.")
    if not missing_sources.empty:
        with st.expander("Rows requiring source-link review"):
            st.dataframe(
                missing_sources[["ticker", "metric", "period", "value", "form", "accession"]],
                hide_index=True,
                use_container_width=True,
            )
    if not duplicates.empty:
        with st.expander("Duplicate ticker / metric / period keys"):
            st.dataframe(duplicates, hide_index=True, use_container_width=True)


def render_evaluation(
    frame: pd.DataFrame, conflicts: list[dict[str, Any]], evaluations: list[dict[str, Any]]
) -> None:
    st.subheader("Evaluation transparency")
    status = readiness(frame, [])
    expected_pairs = len(TARGET_TICKERS) * max(status["metrics"], 1)
    observed_pairs = frame[["ticker", "metric"]].drop_duplicates().shape[0] if not frame.empty else 0
    coverage = observed_pairs / expected_pairs * 100 if expected_pairs else 0
    a, b, c, d = st.columns(4)
    a.metric("Company coverage", f'{status["companies"]}/3')
    b.metric("Metric-pair coverage", f"{coverage:.0f}%")
    c.metric("Source-link rate", f'{status["citation_rate"]:.0f}%')
    d.metric(
        "Data-quality review flags",
        len(conflicts),
        help="Review signals from duplicate facts, amendments, restatements, or fiscal-period differences; not failed evaluation cases.",
    )
    st.markdown(
        """
        <div class="panel"><b>How to read these checks</b><br>
        <span class="small-muted">Coverage and source-link rate are deterministic presentation checks,
        not claims of financial correctness. Data-quality review flags are intentionally surfaced so
        amended, restated, or duplicated SEC facts can be inspected instead of hidden. RAG quality is
        measured with the prepared question set for citation precision, answer faithfulness, and
        abstention behavior.</span></div>
        """,
        unsafe_allow_html=True,
    )
    if evaluations:
        st.markdown("#### Prepared evaluation results")
        st.dataframe(pd.json_normalize(evaluations), hide_index=True, use_container_width=True)
    else:
        st.info(
            "No prepared evaluation result set was discovered. The dashboard reports only live "
            "coverage checks and does not invent model-quality scores."
        )
    with st.expander("Suggested acceptance gates"):
        st.markdown(
            "- Every displayed metric has a filing accession or public SEC URL.\n"
            "- Numeric comparisons use the same metric definition, unit, and period basis.\n"
            "- Q&A claims include passage-level citations and abstain when evidence is insufficient.\n"
            "- Known amendments/restatements appear in conflict review.\n"
            "- Secrets and raw provider errors never render in the UI."
        )


def main() -> None:
    inject_css()
    metrics, conflicts, evaluations, notes = load_prepared_data()
    selected_tickers, selected_metric = render_sidebar(metrics, notes)
    filtered = metrics[metrics["ticker"].isin(selected_tickers)].copy() if selected_tickers else metrics.iloc[0:0]

    st.markdown('<div class="eyebrow">Public filing intelligence</div>', unsafe_allow_html=True)
    st.title("Three chipmakers.\nOne evidence trail.")
    st.markdown(
        '<p class="lede">Compare NVDA, AMD, and INTC financial signals, inspect every metric input, '
        "and ask source-cited questions without hiding uncertainty.</p>",
        unsafe_allow_html=True,
    )
    st.caption(f"Dashboard date: {date.today().isoformat()} · Prepared data is never silently refreshed")

    tabs = st.tabs(
        [
            "Overview",
            "Compare",
            "Traceability",
            "RAG Cited Q&A",
            "Data quality",
            "Evaluation",
        ]
    )
    with tabs[0]:
        render_overview(filtered, selected_metric)
    with tabs[1]:
        render_comparison(filtered)
    with tabs[2]:
        render_traceability(filtered)
    with tabs[3]:
        render_qa(filtered)
    with tabs[4]:
        render_quality(filtered, conflicts)
    with tabs[5]:
        render_evaluation(filtered, conflicts, evaluations)


if __name__ == "__main__":
    main()
