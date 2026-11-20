from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


class AnalysisState(TypedDict, total=False):
    question: str
    rows: list[dict[str, Any]]
    columns: list[str]
    plan: list[str]
    plan_text: str
    metric: str | None
    group: str | None
    product: str | None
    date: str | None
    dimension: str | None
    filters: dict[str, list[str]]
    focus_values: list[str]
    top_n: int
    aggregation: str
    period: str | None
    seasonal: bool
    date_note: str | None
    pending: list[str]
    results: list[dict[str, Any]]
    answer: str


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def load_kaggle_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "Global Superstore.csv"
    return pd.read_csv(data_path)


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    prepared = frame.copy()
    prepared.columns = [str(column) for column in prepared.columns]
    saved_note = prepared.attrs.get("date_note")
    if saved_note and "Analysis Week" in prepared:
        return prepared, saved_note

    date_columns = [
        column
        for column in prepared.columns
        if any(word in column.casefold() for word in ("date", "time"))
    ]
    for column in date_columns:
        parsed = pd.to_datetime(prepared[column], errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.8 and parsed.nunique() > 1:
            prepared[column] = parsed
            return prepared, None

    if "Year" in prepared and "weeknum" in prepared:
        years = pd.to_numeric(prepared["Year"], errors="coerce")
        weeks = pd.to_numeric(prepared["weeknum"], errors="coerce")
        year_start = pd.to_datetime(
            years.astype("Int64").astype("string") + "-01-01",
            errors="coerce",
        )
        year_end = pd.to_datetime(
            years.astype("Int64").astype("string") + "-12-31",
            errors="coerce",
        )
        approximate = year_start + pd.to_timedelta((weeks - 1) * 7 + 3, unit="D")
        out_of_year = approximate.dt.year > years
        approximate = approximate.where(~out_of_year, year_end)
        valid = years.notna() & weeks.between(1, 53) & approximate.notna()
        prepared["Analysis Week"] = approximate.where(valid)
        note = (
            "The supplied Order Date and Ship Date contain no usable calendar dates. "
            "Time analysis uses approximate week midpoints reconstructed from Year and weeknum; "
            "quarter and seasonality results are estimates."
        )
        prepared.attrs["date_note"] = note
        return prepared, note

    return prepared, None


def _filter_frame(
    frame: pd.DataFrame, filters: dict[str, list[str]] | None
) -> pd.DataFrame:
    filtered = frame
    for column, wanted in (filters or {}).items():
        if column in filtered and wanted:
            choices = {str(value).casefold() for value in wanted}
            filtered = filtered[
                filtered[column].astype(str).str.casefold().isin(choices)
            ]
    return filtered


def _apply_period(
    frame: pd.DataFrame, date: str | None, period: str | None
) -> tuple[pd.DataFrame, str | None]:
    if period != "last_quarter" or not date or date not in frame:
        return frame, None
    dates = pd.to_datetime(frame[date], errors="coerce")
    usable = dates.dropna()
    if usable.empty:
        return frame.iloc[0:0], None
    latest_quarter = usable.max().to_period("Q")
    return (
        frame.loc[dates.dt.to_period("Q") == latest_quarter],
        f"latest available quarter, Q{latest_quarter.quarter} {latest_quarter.year}",
    )


def _json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, default=str)


@tool
def dataset_overview(
    rows: list[dict[str, Any]], filters: dict[str, list[str]]
) -> str:
    """Summarize dataset size, quality, and numeric ranges, optionally filtered."""
    frame = _filter_frame(_frame(rows), filters)
    numeric = frame.select_dtypes(include="number")
    summary: dict[str, Any] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "filters": filters,
        "missing_values": {
            str(name): int(count)
            for name, count in frame.isna().sum().items()
            if count > 0
        },
        "numeric_summary": {},
        "categorical_summary": {},
    }
    for name in numeric.columns:
        values = numeric[name].dropna()
        if not values.empty:
            summary["numeric_summary"][str(name)] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    for name in frame.select_dtypes(exclude="number").columns:
        modes = frame[name].mode(dropna=True)
        summary["categorical_summary"][str(name)] = {
            "unique_values": int(frame[name].nunique(dropna=True)),
            "most_common": str(modes.iloc[0]) if not modes.empty else None,
        }
    return _json_result(summary)


@tool
def compare_groups(
    rows: list[dict[str, Any]],
    metric: str,
    group: str,
    filters: dict[str, list[str]],
    focus_values: list[str],
    aggregation: str,
    period: str | None,
    date: str | None,
) -> str:
    """Compare group totals or averages, optionally within a time period."""
    frame, period_label = _apply_period(
        _filter_frame(_frame(rows), filters), date, period
    )
    if metric not in frame or group not in frame:
        return _json_result({"error": "The selected metric or group column is missing."})
    values = pd.to_numeric(frame[metric], errors="coerce")
    usable = frame.assign(_metric=values).dropna(subset=[group, "_metric"])
    if focus_values:
        selected = {value.casefold() for value in focus_values}
        usable = usable[usable[group].astype(str).str.casefold().isin(selected)]
    grouped = usable.groupby(group, observed=True)["_metric"].agg(
        ["mean", "count", "sum"]
    )
    if grouped.empty:
        return _json_result({"error": f"No usable values in {metric} for {group}."})
    sort_column = "mean" if aggregation == "mean" else "sum"
    grouped = grouped.sort_values(sort_column, ascending=False)
    return _json_result(
        {
            "metric": metric,
            "group": group,
            "aggregation": aggregation,
            "period": period_label,
            "groups": [
                {
                    "name": str(index),
                    "mean": float(row["mean"]),
                    "count": int(row["count"]),
                    "sum": float(row["sum"]),
                }
                for index, row in grouped.iterrows()
            ],
        }
    )


@tool
def rank_products(
    rows: list[dict[str, Any]],
    metric: str,
    product: str,
    date: str,
    filters: dict[str, list[str]],
    top_n: int,
    period: str | None,
) -> str:
    """Rank products by total metric, optionally in the dataset's latest quarter."""
    frame, period_label = _apply_period(
        _filter_frame(_frame(rows), filters), date, period
    )
    if metric not in frame or product not in frame:
        return _json_result({"error": "A numeric measure or product-name column is missing."})
    values = pd.to_numeric(frame[metric], errors="coerce")
    grouped = (
        frame.assign(_metric=values)
        .dropna(subset=[product, "_metric"])
        .groupby(product, observed=True)["_metric"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
        .head(max(1, min(top_n, 10)))
    )
    if grouped.empty:
        return _json_result({"error": "No products have usable values for this analysis."})
    total = float(pd.to_numeric(frame[metric], errors="coerce").sum())
    return _json_result(
        {
            "metric": metric,
            "product": product,
            "period": period_label,
            "total": total,
            "products": [
                {
                    "name": str(index),
                    "total": float(row["sum"]),
                    "count": int(row["count"]),
                    "share_pct": float(row["sum"] / total * 100) if total else None,
                }
                for index, row in grouped.iterrows()
            ],
        }
    )


@tool
def analyze_time_trend(
    rows: list[dict[str, Any]],
    metric: str,
    date: str,
    filters: dict[str, list[str]],
    aggregation: str,
    seasonal: bool,
) -> str:
    """Measure monthly movement and, on request, month-of-year seasonality."""
    frame = _filter_frame(_frame(rows), filters)
    if metric not in frame or date not in frame:
        return _json_result({"error": "The selected metric or date column is missing."})
    dates = pd.to_datetime(frame[date], errors="coerce")
    values = pd.to_numeric(frame[metric], errors="coerce")
    usable = pd.DataFrame({"date": dates, "value": values}).dropna()
    if usable.empty:
        return _json_result({"error": f"No usable dates and values for {metric}."})
    series = usable.set_index("date")["value"]
    monthly = (
        (series.resample("ME").sum() if aggregation == "sum" else series.resample("ME").mean())
        .dropna()
    )
    if monthly.empty:
        return _json_result({"error": f"No monthly values for {metric}."})
    first, last = float(monthly.iloc[0]), float(monthly.iloc[-1])
    change_pct = ((last - first) / abs(first) * 100) if first else None
    seasonality: list[dict[str, Any]] = []
    if seasonal:
        by_month = monthly.groupby(monthly.index.month).mean()
        seasonality = [
            {"month": int(month), "average": float(value)}
            for month, value in by_month.items()
        ]
    return _json_result(
        {
            "metric": metric,
            "aggregation": aggregation,
            "filters": filters,
            "seasonal": seasonal,
            "periods": int(len(monthly)),
            "years": int(monthly.index.year.nunique()),
            "first_period": str(monthly.index[0].date()),
            "first_value": first,
            "last_period": str(monthly.index[-1].date()),
            "last_value": last,
            "change_pct": float(change_pct) if change_pct is not None else None,
            "monthly_values": [
                {"period": str(index.date()), "value": float(value)}
                for index, value in monthly.items()
            ],
            "seasonality": seasonality,
        }
    )


@tool
def explain_drivers(
    rows: list[dict[str, Any]],
    metric: str,
    group: str,
    dimension: str,
    focus_values: list[str],
    filters: dict[str, list[str]],
) -> str:
    """Break down a two-group performance gap by product or sub-category."""
    frame = _filter_frame(_frame(rows), filters)
    if metric not in frame or group not in frame or dimension not in frame:
        return _json_result({"error": "A measure, comparison group, or driver column is missing."})
    if len(focus_values) != 2:
        return _json_result(
            {"error": f"Name two values of {group} to compare their performance drivers."}
        )
    left, right = focus_values
    values = pd.to_numeric(frame[metric], errors="coerce")
    usable = frame.assign(_metric=values).dropna(subset=[group, dimension, "_metric"])
    chosen = {left.casefold(), right.casefold()}
    usable = usable[usable[group].astype(str).str.casefold().isin(chosen)]
    grouped = usable.groupby([dimension, group], observed=True)["_metric"].sum().unstack(fill_value=0)
    left_column = next((column for column in grouped.columns if str(column).casefold() == left.casefold()), None)
    right_column = next((column for column in grouped.columns if str(column).casefold() == right.casefold()), None)
    if left_column is None or right_column is None:
        return _json_result({"error": f"No comparable values found for {left} and {right}."})
    detail = grouped.reindex(columns=[left_column, right_column], fill_value=0)
    detail = detail.assign(_gap=detail[left_column] - detail[right_column])
    detail = detail.sort_values("_gap", key=lambda column: column.abs(), ascending=False)
    return _json_result(
        {
            "metric": metric,
            "group": group,
            "dimension": dimension,
            "left": left,
            "right": right,
            "left_total": float(usable.loc[usable[group] == left_column, "_metric"].sum()),
            "right_total": float(usable.loc[usable[group] == right_column, "_metric"].sum()),
            "drivers": [
                {
                    "name": str(index),
                    "left": float(row[left_column]),
                    "right": float(row[right_column]),
                    "gap": float(row["_gap"]),
                }
                for index, row in detail.head(5).iterrows()
            ],
        }
    )


@tool
def detect_anomalies(
    rows: list[dict[str, Any]], metric: str, filters: dict[str, list[str]]
) -> str:
    """Find unusually high or low numeric values with the 1.5-IQR rule."""
    frame = _filter_frame(_frame(rows), filters)
    if metric not in frame:
        return _json_result({"error": f"Column {metric} was not found."})
    values = pd.to_numeric(frame[metric], errors="coerce")
    clean = values.dropna()
    if clean.empty:
        return _json_result({"error": f"Column {metric} has no usable numeric values."})
    q1, q3 = clean.quantile([0.25, 0.75])
    spread = q3 - q1
    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    indices = clean[(clean < low) | (clean > high)].index
    return _json_result(
        {
            "metric": metric,
            "count": int(len(indices)),
            "total": int(len(clean)),
            "low_bound": float(low),
            "high_bound": float(high),
            "anomalies": [
                {"row": int(index) + 1, "value": float(values.loc[index])}
                for index in indices[:20]
            ],
            "truncated": len(indices) > 20,
            "filters": filters,
        }
    )


TOOLS: dict[str, Callable[..., Any]] = {
    "overview": dataset_overview,
    "compare": compare_groups,
    "rank": rank_products,
    "trend": analyze_time_trend,
    "drivers": explain_drivers,
    "anomaly": detect_anomalies,
}

_TOOL_LABELS = {
    "overview": "Dataset overview",
    "compare": "Group comparison",
    "rank": "Product ranking",
    "trend": "Time trend and seasonality",
    "drivers": "Performance drivers",
    "anomaly": "Anomaly detection",
}


def _mentioned_column(question: str, columns: list[str]) -> str | None:
    lowered = question.casefold()
    matches = [
        column
        for column in columns
        if re.search(rf"(?<!\w){re.escape(column.casefold())}(?!\w)", lowered)
    ]
    return max(matches, key=len) if matches else None


def _find_column(columns: list[str], names: tuple[str, ...]) -> str | None:
    for name in names:
        for column in columns:
            lowered = column.casefold()
            if name in lowered and not any(
                excluded in lowered for excluded in (" id", "code", "postal")
            ):
                return column
    return None


def _pick_columns(question: str, frame: pd.DataFrame) -> dict[str, str | None]:
    columns = [str(column) for column in frame.columns]
    numeric = [
        str(column)
        for column in frame.select_dtypes(include="number").columns
        if not any(term in str(column).casefold() for term in ("id", "postal", "zip"))
    ]
    named = _mentioned_column(question, columns)
    metric = named if named in numeric else None
    if metric is None:
        metric_intents = (
            (("profit", "margin"), ("profit", "margin")),
            (("quantity", "units", "volume"), ("quantity", "unit")),
            (("discount",), ("discount",)),
            (("order count", "number of orders"), ("order", "count")),
            (("revenue", "sales"), ("revenue", "sales")),
        )
        text = question.casefold()
        for phrases, column_terms in metric_intents:
            if any(phrase in text for phrase in phrases):
                metric = _find_column(numeric, column_terms)
                if metric:
                    break
    if metric is None:
        metric = _find_column(numeric, ("sales", "revenue", "profit")) or (
            numeric[0] if numeric else None
        )

    date = "Analysis Week" if "Analysis Week" in columns else None
    if date is None:
        for column in columns:
            if not any(word in column.casefold() for word in ("date", "time")):
                continue
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8 and parsed.nunique() > 1:
                date = column
                break
    if date is None:
        for column in columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                continue
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8 and parsed.nunique() > 1:
                date = column
                break

    aliases = (
        (("sub-category", "subcategory", "sub category"), ("sub-category", "subcategory")),
        (("product", "item"), ("product name", "product", "item")),
        (("region",), ("region",)),
        (("category",), ("category",)),
        (("segment",), ("segment",)),
        (("state",), ("state",)),
        (("city",), ("city",)),
        (("ship mode",), ("ship mode",)),
    )
    group = None
    explicit_group = False
    lowered_question = question.casefold()
    for phrases, column_terms in aliases:
        if any(phrase in lowered_question for phrase in phrases):
            group = _find_column(columns, column_terms)
            if group:
                explicit_group = True
                break
    categorical = [
        column
        for column in columns
        if column != date
        and not pd.api.types.is_numeric_dtype(frame[column])
        and 1 < frame[column].nunique(dropna=True) <= 200
        and not any(term in column.casefold() for term in ("id", "name"))
    ]
    if not explicit_group:
        mentioned_groups = [
            (column, _matched_values(question, frame, column))
            for column in categorical
        ]
        matched = [(column, values) for column, values in mentioned_groups if len(values) >= 2]
        if matched:
            group = max(matched, key=lambda item: len(item[1]))[0]
    if group is None:
        group = categorical[0] if categorical else None
    product = _find_column(columns, ("product name", "product", "item name", "item"))
    if product is None:
        product = group
    dimension = _find_column(columns, ("sub-category", "subcategory", "product name", "product"))
    if "product" in lowered_question or "item" in lowered_question:
        dimension = product or dimension
    if dimension is None:
        dimension = group
    return {
        "metric": metric,
        "group": group,
        "product": product,
        "date": date,
        "dimension": dimension,
    }


def _matched_values(question: str, frame: pd.DataFrame, column: str) -> list[str]:
    text = question.casefold()
    values = [str(value) for value in frame[column].dropna().unique()]
    matches: list[tuple[int, str]] = []
    for value in values:
        pattern = rf"(?<!\w){re.escape(value.casefold())}(?!\w)"
        found = re.search(pattern, text)
        if found:
            matches.append((found.start(), value))
    return [value for _, value in sorted(matches)]


def _select_filters(
    question: str, frame: pd.DataFrame, group: str | None, plan: list[str]
) -> tuple[dict[str, list[str]], list[str]]:
    filters: dict[str, list[str]] = {}
    focus_values: list[str] = []
    for column in frame.columns:
        if (
            pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_datetime64_any_dtype(frame[column])
            or column.casefold().endswith("date")
        ):
            continue
        unique_count = frame[column].nunique(dropna=True)
        if unique_count > 2000 and "product name" not in column.casefold():
            continue
        values = _matched_values(question, frame, str(column))
        if not values:
            continue
        if str(column) == group and any(action in plan for action in ("compare", "drivers")):
            focus_values = values[:2]
        else:
            filters[str(column)] = values
    return filters, focus_values


def _plan_tools(question: str, columns: dict[str, str | None]) -> list[str]:
    text = question.casefold()
    selected: list[str] = []
    has_top_n = bool(re.search(r"\b(top|best|rank)\s*(\d+)?\b", text))
    product_requested = bool(re.search(r"\b(product|products|item|items)\b", text))
    if has_top_n and (product_requested or columns.get("product")):
        selected.append("rank")
    if re.search(r"\b(trend|seasonal|seasonality|over time|month|monthly|year|yearly|increase|decrease|growth|change)\b", text):
        selected.append("trend")
    if re.search(
        r"\b(compare|comparison|versus|vs\.?|highest|lowest|rank|performance)\b"
        r"|\bby\s+(region|category|segment|product|state|city|sub[\s-]?category)\b",
        text,
    ):
        if "rank" not in selected:
            selected.append("compare")
    if re.search(r"\b(driv\w*|why|cause|behind|explain)\b", text):
        selected.append("drivers")
    if re.search(r"\b(anomal\w*|outlier\w*|unusual|spike|drop|suspicious)\b", text):
        selected.append("anomaly")
    if re.search(r"\b(overview|summar\w*|describe|dataset|data|missing|column|profile)\b", text):
        selected.append("overview")
    if not selected:
        selected.append("overview")
    return selected


def _plan(state: AnalysisState) -> dict[str, Any]:
    frame = _frame(state["rows"])
    columns = _pick_columns(state["question"], frame)
    planned = _plan_tools(state["question"], columns)
    filters, focus_values = _select_filters(
        state["question"], frame, columns["group"], planned
    )
    question = state["question"].casefold()
    top_match = re.search(r"\btop\s+(\d+)\b", question)
    top_n = min(int(top_match.group(1)), 10) if top_match else 3
    aggregation = "mean" if re.search(r"\b(average|mean)\b", question) else "sum"
    period = "last_quarter" if re.search(r"\b(last|latest)\s+quarter\b", question) else None
    seasonal = bool(re.search(r"\b(seasonal|seasonality)\b", question))
    descriptions = {
        "overview": "profile data volume, quality, and key measures",
        "compare": f"compare {aggregation} {columns['metric'] or 'measure'} by {columns['group'] or 'group'}",
        "rank": f"rank the top {top_n} products by total {columns['metric'] or 'measure'}",
        "trend": (
            f"measure {columns['metric'] or 'measure'} over {columns['date'] or 'time'}"
            + (" and compare the same months across years" if seasonal else "")
        ),
        "drivers": f"break down the difference by {columns['dimension'] or 'category'}",
        "anomaly": f"flag unusual {columns['metric'] or 'numeric'} values",
    }
    if period:
        descriptions["rank"] += " in the latest quarter present in the data"
        descriptions["compare"] += " in the latest quarter present in the data"
    if filters:
        descriptions["trend"] += " for " + ", ".join(
            f"{column}={', '.join(values)}" for column, values in filters.items()
        )
    return {
        "plan": planned,
        "pending": planned.copy(),
        "results": [],
        **columns,
        "filters": filters,
        "focus_values": focus_values,
        "top_n": top_n,
        "aggregation": aggregation,
        "period": period,
        "seasonal": seasonal,
        "plan_text": "; ".join(
            f"{_TOOL_LABELS[name]}: {descriptions[name]}" for name in planned
        ),
    }


def _dispatch(state: AnalysisState) -> str:
    pending = state.get("pending", [])
    return pending[0] if pending else "synthesize"


def _make_tool_node(name: str) -> Callable[[AnalysisState], dict[str, Any]]:
    selected_tool = TOOLS[name]

    def run(state: AnalysisState) -> dict[str, Any]:
        args: dict[str, Any] = {
            "rows": state["rows"],
            "filters": state.get("filters", {}),
        }
        error = None
        if name != "overview":
            if not state.get("metric"):
                error = "This analysis needs a numeric column, but none was found."
            else:
                args["metric"] = state["metric"]
        if name == "compare":
            if not state.get("group"):
                error = "This analysis needs a category column to compare."
            args.update(
                group=state.get("group") or "",
                focus_values=state.get("focus_values", []),
                aggregation=state.get("aggregation", "sum"),
                period=state.get("period"),
                date=state.get("date"),
            )
        elif name == "rank":
            if not state.get("product") or not state.get("date"):
                error = "Product ranking needs a product-name column and an order-date column."
            args.update(
                product=state.get("product") or "",
                date=state.get("date") or "",
                top_n=state.get("top_n", 3),
                period=state.get("period"),
            )
        elif name == "trend":
            if not state.get("date"):
                error = "This analysis needs a date or time column."
            args.update(
                date=state.get("date") or "",
                aggregation=state.get("aggregation", "sum"),
                seasonal=state.get("seasonal", False),
            )
        elif name == "drivers":
            if not state.get("group") or not state.get("dimension"):
                error = "Driver analysis needs a group column and a product/category breakdown."
            args.update(
                group=state.get("group") or "",
                dimension=state.get("dimension") or "",
                focus_values=state.get("focus_values", []),
            )
        if error:
            result = {"error": error}
        else:
            result = json.loads(selected_tool.invoke(args))
        return {
            "pending": state["pending"][1:],
            "results": state["results"] + [{"tool": name, "data": result}],
        }

    return run


def _format_metric(metric: str, value: float) -> str:
    formatted = f"{value:,.2f}"
    return f"${formatted}" if any(word in metric.casefold() for word in ("sales", "revenue", "profit")) else formatted


def _synthesize(state: AnalysisState) -> dict[str, str]:
    insights: list[str] = []
    for result in state.get("results", []):
        name, data = result["tool"], result["data"]
        if "error" in data:
            insights.append(data["error"])
        elif name == "overview":
            missing = sum(data["missing_values"].values())
            numeric = data["numeric_summary"]
            detail = f"{len(numeric)} numeric measures were summarized; {missing:,} missing cells were found."
            insights.append(f"The dataset has {data['rows']:,} rows and {data['columns']} columns. {detail}")
            metric = next(
                (column for column in numeric if any(word in column.casefold() for word in ("sales", "revenue", "profit"))),
                next(iter(numeric), None),
            )
            if metric:
                stats = numeric[metric]
                insights.append(
                    f"{metric} averages {_format_metric(metric, stats['mean'])} "
                    f"(range {_format_metric(metric, stats['min'])}–{_format_metric(metric, stats['max'])})."
                )
        elif name == "compare":
            groups = data["groups"]
            if len(groups) == 1:
                insights.append(f"Only one group is available for {data['group']}.")
            else:
                high, low = groups[0], groups[-1]
                value_key = "mean" if data["aggregation"] == "mean" else "sum"
                label = "average" if value_key == "mean" else "total"
                high_value = _format_metric(data["metric"], high[value_key])
                low_value = _format_metric(data["metric"], low[value_key])
                insights.append(
                    f"{high['name']} has the highest {label} {data['metric']} "
                    f"({high_value}); {low['name']} is lowest ({low_value})."
                )
                if low[value_key] != 0:
                    gap = (high[value_key] - low[value_key]) / abs(low[value_key]) * 100
                    insights.append(f"The gap is {gap:,.1f}% of the lowest group's {label}.")
                if data.get("period"):
                    insights.append(f"Scope: {data['period']}.")
        elif name == "rank":
            products = data["products"]
            if products:
                ranking = "; ".join(
                    f"{index + 1}. {item['name']} ({_format_metric(data['metric'], item['total'])})"
                    for index, item in enumerate(products)
                )
                insights.append(
                    f"Top products by {data['metric']}"
                    + (f" in the {data['period']}" if data.get("period") else "")
                    + f": {ranking}."
                )
                if products[0]["share_pct"] is not None:
                    share = sum(item["share_pct"] or 0 for item in products)
                    insights.append(f"Together they account for {share:.1f}% of sales in this scope.")
        elif name == "trend":
            change = data["change_pct"]
            direction = (
                f"rose by {change:,.1f}%" if change is not None and change > 0
                else f"fell by {abs(change):,.1f}%" if change is not None and change < 0
                else "was broadly flat" if change == 0
                else "could not be compared as a percentage"
            )
            insights.append(
                f"{data['aggregation'].title()} {data['metric']} {direction} from "
                f"{data['first_period']} ({_format_metric(data['metric'], data['first_value'])}) "
                f"to {data['last_period']} ({_format_metric(data['metric'], data['last_value'])}), "
                f"across {data['periods']} months."
            )
            if data["seasonal"] and data["seasonality"]:
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                ordered = sorted(data["seasonality"], key=lambda item: item["average"])
                low, high = ordered[0], ordered[-1]
                insights.append(
                    f"Seasonally, {month_names[high['month'] - 1]} is strongest "
                    f"({_format_metric(data['metric'], high['average'])} average) and "
                    f"{month_names[low['month'] - 1]} weakest "
                    f"({_format_metric(data['metric'], low['average'])}); this compares "
                    f"month-of-year averages across {data['years']} year(s)."
                )
            elif len(data["monthly_values"]) >= 3:
                series = data["monthly_values"]
                changes = [series[i]["value"] - series[i - 1]["value"] for i in range(1, len(series))]
                biggest = max(changes, key=abs)
                index = changes.index(biggest) + 1
                insights.append(
                    f"The largest month-to-month move was "
                    f"{'up' if biggest > 0 else 'down'} "
                    f"({_format_metric(data['metric'], abs(biggest))}) in {series[index]['period']}."
                )
        elif name == "drivers":
            left, right = data["left"], data["right"]
            left_total = _format_metric(data["metric"], data["left_total"])
            right_total = _format_metric(data["metric"], data["right_total"])
            direction = "higher" if data["left_total"] > data["right_total"] else "lower"
            insights.append(
                f"{left} recorded {left_total} in {data['metric']} versus "
                f"{right} at {right_total}. The largest {data['dimension']} differences were "
                + "; ".join(
                    f"{item['name']} ({left} {_format_metric(data['metric'], item['left'])}, "
                    f"{right} {_format_metric(data['metric'], item['right'])})"
                    for item in data["drivers"][:3]
                )
                + f". Overall, {left} was {direction} than {right}."
            )
        elif name == "anomaly":
            count, total = data["count"], data["total"]
            if count:
                examples = ", ".join(
                    f"{_format_metric(data['metric'], item['value'])} (row {item['row']})"
                    for item in data["anomalies"][:3]
                )
                insights.append(
                    f"Found {count} unusually high or low {data['metric']} value(s) "
                    f"({count / total * 100:.1f}% of {total:,} rows) using the IQR rule. "
                    f"Examples: {examples}."
                )
            else:
                insights.append(
                    f"No IQR outliers were detected in {data['metric']} across {total:,} usable rows."
                )
    if state.get("date_note") and (
        any(name in ("rank", "trend") for name in state.get("plan", []))
        or state.get("period")
    ):
        insights.append(f"Time-data caveat: {state['date_note']}")
    return {
        "answer": " ".join(insights)
        if insights
        else "I couldn't derive an insight from this data."
    }


def _build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("plan", _plan)
    graph.add_node("dispatch", lambda state: {})
    for name in TOOLS:
        graph.add_node(name, _make_tool_node(name))
        graph.add_edge(name, "dispatch")
    graph.add_node("synthesize", _synthesize)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "dispatch")
    graph.add_conditional_edges(
        "dispatch",
        _dispatch,
        {**{name: name for name in TOOLS}, "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


ANALYST_GRAPH = _build_graph()


def analyze(question: str, frame: pd.DataFrame) -> AnalysisState:
    clean, date_note = prepare_dataset(frame)
    rows = json.loads(clean.to_json(orient="records", date_format="iso"))
    return ANALYST_GRAPH.invoke(
        {
            "question": question,
            "rows": rows,
            "columns": list(clean.columns),
            "date_note": date_note,
        }
    )


def make_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    months = pd.date_range("2014-01-01", periods=48, freq="MS")
    catalog = {
        "Furniture": [("Chairs", "Ergonomic Chair", 245), ("Tables", "Conference Table", 410), ("Bookcases", "Oak Bookcase", 180)],
        "Office Supplies": [("Storage", "File Organizer", 48), ("Binders", "Binder Set", 32), ("Art", "Desk Lamp", 65)],
        "Technology": [("Phones", "Wireless Phone", 330), ("Accessories", "Laptop Dock", 155), ("Copiers", "Office Copier", 520)],
    }
    regions = {"Central": 0.94, "East": 1.16, "South": 0.88, "West": 1.04}
    rows: list[dict[str, Any]] = []
    for month_index, month in enumerate(months):
        season = 1 + 0.18 * np.sin((month.month - 1) * np.pi / 6 - np.pi / 3)
        growth = 1 + month_index * 0.006
        for region, factor in regions.items():
            for category, products in catalog.items():
                for subcategory, product, base in products:
                    sales = base * factor * season * growth * rng.normal(1, 0.12)
                    if month.year == 2017 and month.month == 10 and region == "East" and product == "Wireless Phone":
                        sales *= 3.5
                    quantity = int(max(1, rng.normal(sales / 80, 2)))
                    discount = float(rng.choice([0, 0.05, 0.1, 0.2, 0.3]))
                    profit = sales * (0.18 - discount * 0.55) * rng.normal(1, 0.15)
                    rows.append(
                        {
                            "Order Date": month + pd.Timedelta(days=14),
                            "Region": region,
                            "Category": category,
                            "Sub-Category": subcategory,
                            "Product Name": product,
                            "Sales": round(float(sales), 2),
                            "Profit": round(float(profit), 2),
                            "Quantity": quantity,
                            "Discount": discount,
                            "Segment": str(rng.choice(["Consumer", "Corporate", "Home Office"])),
                        }
                    )
    return pd.DataFrame(rows)
