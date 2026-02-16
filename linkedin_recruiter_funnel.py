"""
Compute recruiter funnel metrics from LinkedIn + Zoho CSV exports.

Usage (per recruiter export):
  python linkedin_recruiter_funnel.py \
    --candidates ./Candidates_2026_01_13.csv \
    --connections ./Connections.csv \
    --invitations ./Invitations.csv \
    --deals-x-cand ./Deals_X_Cand.csv \
    --deals ./Deals.csv \
    --month 2 --year 2026 \
    --outdir ./out

Batch mode with per-recruiter LinkedIn exports:
  python linkedin_recruiter_funnel.py \
    --candidates ./Candidates_2026_01_13.csv \
    --deals-x-cand ./Deals_X_Cand.csv \
    --deals ./Deals.csv \
    --recruiters-config ./recruiters.json \
    --all-months \
    --outdir ./out

Google Sheets sync:
  export GOOGLE_SHEETS_CREDENTIALS=/path/to/service_account.json
  python linkedin_recruiter_funnel.py ... --sheet-id <spreadsheet_id>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import unicodedata
from datetime import datetime
from calendar import monthrange
from typing import cast

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def normalize_linkedin_url(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    x = x.replace({"nan": "", "none": "", "null": ""})
    x = x.str.replace(r"[\?#].*$", "", regex=True)
    x = x.str.replace(r"^http://", "https://", regex=True)
    x = x.str.replace(r"/+$", "", regex=True)
    return x


def find_column(df: pd.DataFrame, preferred: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in cols:
            return cols[p.lower()]
    return None


def parse_dt_flex(series: pd.Series) -> pd.Series:
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%y, %I:%M %p",
        "%d-%b-%y",
    ]
    s = series.astype(str)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        parsed = pd.to_datetime(s, errors="coerce", format=fmt)
        out = out.fillna(parsed)
    # Fallback for any remaining values without emitting warnings
    remaining = out.isna()
    if remaining.any():
        parsed = pd.to_datetime(s[remaining], errors="coerce")
        out.loc[remaining] = parsed
    return out


def safe_rate(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return (num / den) * 100.0


def weeks_in_month(year: int, month: int) -> int:
    """Unused — kept for reference. Connection requests now use flat 1000/month."""
    days = monthrange(year, month)[1]
    return int((days + 6) // 7)


def format_month_label(year: int, month: int) -> str:
    return datetime(year, month, 1).strftime("%b %Y")


def slugify_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return cleaned or "recruiter"


def load_recruiters_config(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("recruiters") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("Recruiters config must be a list or {\"recruiters\": [...]}.")

    normalized = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Each recruiter config entry must be an object.")
        name = item.get("recruiter_name") or item.get("name") or item.get("linkedin_from_name")
        connections = item.get("connections") or item.get("connections_path")
        invitations = item.get("invitations") or item.get("invitations_path")
        if not name:
            raise ValueError("Recruiter entry missing recruiter_name.")
        if not connections or not invitations:
            raise ValueError(f"Recruiter '{name}' is missing connections/invitations path.")
        normalized.append(
            {
                "recruiter_name": name,
                "linkedin_from_name": item.get("linkedin_from_name"),
                "connections": connections,
                "invitations": invitations,
                "vendor_filter": item.get("vendor") or item.get("vendor_name"),
                "vendor_id_filter": item.get("vendor_id") or item.get("vendor_id_filter"),
            }
        )
    return normalized


def build_summary_comparison(
    combined_df: pd.DataFrame,
    recruiter_order: list[str],
) -> pd.DataFrame:
    df = combined_df.copy()
    df["cohort_start"] = pd.to_datetime(df["cohort_start"], errors="coerce")
    df = df[df["cohort_start"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    latest_start = df["cohort_start"].max()
    latest_label = format_month_label(latest_start.year, latest_start.month)

    metrics = [
        ("connections_accepted", "Connections Accepted"),
        ("acceptance_rate_pct", "Acceptance Rate"),
        ("candidates_applied", "Candidates Who Applied"),
        ("application_rate_pct", "Application Rate"),
        ("screening_pass_rate_pct", "Screening Pass Rate"),
        ("presented_to_client", "Presented to Client"),
        ("client_acceptance_rate_pct", "Client Acceptance Rate"),
        ("hired", "Hired"),
        ("hire_rate_pct", "Hire Rate (Top-of-Funnel)"),
    ]

    count_keys = {
        "connections_accepted",
        "candidates_applied",
        "presented_to_client",
        "accepted_by_client",
        "rejected_by_client",
        "hired",
    }

    def format_value(key: str, value: float | None) -> str:
        if value is None:
            return ""
        if key.endswith("_rate_pct"):
            return f"{value:.1f}%"
        return f"{int(round(value))}"

    rows: list[dict] = []
    rows.append({"Metric": f"Current Month ({latest_label})"})

    current_df = cast(pd.DataFrame, df[df["cohort_start"] == latest_start])
    for key, label in metrics:
        row = {"Metric": label}
        values = []
        for recruiter in recruiter_order:
            v = cast(pd.Series, current_df.loc[current_df["recruiter_name"] == recruiter, key])
            val = float(v.iloc[0]) if not v.empty else None
            row[recruiter] = format_value(key, val)
            if val is not None:
                values.append(val)
        team_avg = sum(values) / len(values) if values else None
        row["Team Avg"] = format_value(key, team_avg)
        rows.append(row)

    rows.append({"Metric": ""})

    cohort_start_series = cast(pd.Series, df["cohort_start"])
    unique_months = sorted(cohort_start_series.dt.to_period("M").unique())
    for window in (3, 6):
        if len(unique_months) < window:
            continue
        start_period = unique_months[-window]
        end_period = unique_months[-1]
        start_label = format_month_label(start_period.year, start_period.month)
        end_label = format_month_label(end_period.year, end_period.month)
        rows.append({"Metric": f"{window}-Month Rolling (Avg Rates / Sum Counts) ({start_label} – {end_label})"})

        for key, label in metrics:
            row = {"Metric": label}
            values = []
            for recruiter in recruiter_order:
                r_df = cast(pd.DataFrame, df[df["recruiter_name"] == recruiter]).sort_values("cohort_start")
                recent = r_df.tail(window)
                if len(recent) < window:
                    row[recruiter] = ""
                    continue
                if key in count_keys:
                    val = float(recent[key].sum())
                else:
                    val = float(recent[key].mean())
                row[recruiter] = format_value(key, val)
                values.append(val)
            team_avg = sum(values) / len(values) if values else None
            row["Team Avg"] = format_value(key, team_avg)
            rows.append(row)

        rows.append({"Metric": ""})

    summary_df = pd.DataFrame(rows)
    ordered_cols = ["Metric"] + recruiter_order + ["Team Avg"]
    summary_df = summary_df.reindex(columns=ordered_cols)
    return summary_df


def build_definitions_table() -> pd.DataFrame:
    rows = [
        {
            "Metric": "Connection Requests Sent (est.)",
            "Definition": "Flat estimate of 1,000 requests per month (250 per week x 4 weeks).",
            "Source": "Assumption",
        },
        {
            "Metric": "Connections Accepted",
            "Definition": "Connections in cohort month initiated by recruiter (Sent invites). Connections missing from Invitations are included and flagged internally.",
            "Source": "Connections.csv + Invitations.csv",
        },
        {
            "Metric": "Acceptance Rate",
            "Definition": "Connections Accepted / Connection Requests Sent (est.).",
            "Source": "Calculated",
        },
        {
            "Metric": "Candidates Who Applied",
            "Definition": "Candidates matched by LinkedIn URL to cohort connections, with Created Time on/after cohort month.",
            "Source": "Candidates.csv",
        },
        {
            "Metric": "Application Rate",
            "Definition": "Candidates Who Applied / Connections Accepted.",
            "Source": "Calculated",
        },
        {
            "Metric": "Screening — Passed",
            "Definition": "Cohort candidates with Personal Interview Status = Passed.",
            "Source": "Candidates.csv",
        },
        {
            "Metric": "Screening — Failed",
            "Definition": "Cohort candidates with Personal Interview Status = Fail.",
            "Source": "Candidates.csv",
        },
        {
            "Metric": "Screening Pass Rate",
            "Definition": "Passed / (Passed + Failed).",
            "Source": "Calculated",
        },
        {
            "Metric": "Presented to Client",
            "Definition": "Cohort candidates with Status Candidate in presented/advanced statuses (Review by Customer, Screening Process, Job Offer, Accepted, Hired, Rejected).",
            "Source": "Deals_X_Cand.csv",
        },
        {
            "Metric": "Presentation Rate",
            "Definition": "Share of screened-passed candidates that are actually submitted to a client (Presented to Client / Screening — Passed).",
            "Source": "Calculated",
        },
        {
            "Metric": "Accepted by Client",
            "Definition": "Cohort candidates with Status Candidate = 300 - Accepted by Customer.",
            "Source": "Deals_X_Cand.csv",
        },
        {
            "Metric": "Rejected by Client",
            "Definition": "Cohort candidates with Status Candidate = 500 - Rejected by Customer.",
            "Source": "Deals_X_Cand.csv",
        },
        {
            "Metric": "Client Acceptance Rate",
            "Definition": "Accepted by Client / Presented to Client.",
            "Source": "Calculated",
        },
        {
            "Metric": "Hired",
            "Definition": "Cohort candidates with Status Candidate = 400 - Hired.",
            "Source": "Deals_X_Cand.csv",
        },
        {
            "Metric": "Hire Rate (Top-of-Funnel)",
            "Definition": "Hired / Connections Accepted.",
            "Source": "Calculated",
        },
    ]
    return pd.DataFrame(rows)


def build_trade_breakdown(
    *,
    cohort_cand: pd.DataFrame,
    dx_in_cohort: pd.DataFrame,
    screening_agg: pd.DataFrame,
    trade_col: str,
    month_label: str,
    month_start: datetime,
    month_end: datetime,
    presented_statuses: set[str],
    accepted_status: str,
    rejected_status: str,
    hired_status: str,
) -> pd.DataFrame:
    trade_series = cast(pd.Series, cohort_cand[trade_col]).fillna("").astype(str).str.strip()
    trade_label = trade_series.replace({"": "Unknown"})
    cohort_cand = cohort_cand.copy()
    cohort_cand["_trade"] = trade_label

    trade_by_linkedin = cast(pd.Series, cohort_cand.groupby("linkedin_clean")["_trade"].first())
    trades = sorted(trade_by_linkedin.dropna().unique().tolist())
    trade_groups = cast(pd.Series, cohort_cand.groupby("_trade")["linkedin_clean"].unique())

    screening_agg = cast(pd.DataFrame, screening_agg)
    passed_set = set(screening_agg[screening_agg["passed"]].index)
    failed_set = set(screening_agg[(~screening_agg["passed"]) & screening_agg["failed"]].index)

    status_series = cast(pd.Series, dx_in_cohort["_status"]) if not dx_in_cohort.empty else pd.Series([], dtype=str)
    linkedin_series = (
        cast(pd.Series, dx_in_cohort["_linkedin"]).fillna("") if not dx_in_cohort.empty else pd.Series([], dtype=str)
    )

    rows = []
    for trade in trades:
        trade_values = cast(list, trade_groups.get(trade, []))
        trade_linkedin = set(trade_values)
        candidates_applied = len(trade_linkedin)

        screening_passed = len(trade_linkedin & passed_set)
        screening_failed = len(trade_linkedin & failed_set)

        presented_linkedin = set(
            linkedin_series[status_series.isin(list(presented_statuses))].tolist()
        ) & trade_linkedin
        accepted_linkedin = set(linkedin_series[status_series == accepted_status].tolist()) & trade_linkedin
        rejected_linkedin = set(linkedin_series[status_series == rejected_status].tolist()) & trade_linkedin
        hired_linkedin = set(linkedin_series[status_series == hired_status].tolist()) & trade_linkedin

        presented_linkedin = presented_linkedin & passed_set
        accepted_linkedin = accepted_linkedin & passed_set
        rejected_linkedin = rejected_linkedin & passed_set
        hired_linkedin = hired_linkedin & passed_set

        presented_count = len(presented_linkedin)
        accepted_count = len(accepted_linkedin)
        rejected_count = len(rejected_linkedin)
        hired_count = len(hired_linkedin)

        screening_pass_rate = safe_rate(screening_passed, screening_passed + screening_failed)
        presentation_rate = safe_rate(presented_count, screening_passed)
        client_accept_rate = safe_rate(accepted_count, presented_count)
        hire_rate_from_applied = safe_rate(hired_count, candidates_applied)

        rows.append(
            {
                "cohort_month": month_label,
                "cohort_start": str(month_start.date()),
                "cohort_end": str(month_end.date()),
                "trade_of_service": trade,
                "candidates_applied": candidates_applied,
                "screening_passed": screening_passed,
                "screening_failed": screening_failed,
                "screening_pass_rate_pct": round(screening_pass_rate, 1),
                "presented_to_client": presented_count,
                "presentation_rate_pct": round(presentation_rate, 1),
                "accepted_by_client": accepted_count,
                "rejected_by_client": rejected_count,
                "client_acceptance_rate_pct": round(client_accept_rate, 1),
                "hired": hired_count,
                "hire_rate_from_applied_pct": round(hire_rate_from_applied, 1),
            }
        )

    return pd.DataFrame(rows)


def build_trade_rollup(trade_df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = trade_df.copy()
    df["cohort_start"] = pd.to_datetime(df["cohort_start"], errors="coerce")
    df = df[df["cohort_start"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    cohort_start_series = cast(pd.Series, df["cohort_start"])
    periods = sorted(cohort_start_series.dt.to_period("M").unique())
    if len(periods) < window:
        return pd.DataFrame()

    window_periods = periods[-window:]
    start_period = window_periods[0]
    end_period = window_periods[-1]
    start_label = format_month_label(start_period.year, start_period.month)
    end_label = format_month_label(end_period.year, end_period.month)
    window_label = f"{window}M ({start_label} – {end_label})"

    period_mask = cohort_start_series.dt.to_period("M").isin(window_periods)
    window_df = df[period_mask].copy()

    window_df = cast(pd.DataFrame, window_df)
    grouped = window_df.groupby("trade_of_service", dropna=False)
    rows = []
    for trade, g in grouped:
        applied = int(g["candidates_applied"].sum())
        passed = int(g["screening_passed"].sum())
        failed = int(g["screening_failed"].sum())
        presented = int(g["presented_to_client"].sum())
        accepted = int(g["accepted_by_client"].sum())
        rejected = int(g["rejected_by_client"].sum())
        hired = int(g["hired"].sum())

        screening_pass_rate = safe_rate(passed, passed + failed)
        presentation_rate = safe_rate(presented, passed)
        client_accept_rate = safe_rate(accepted, presented)
        hire_rate = safe_rate(hired, applied)

        rows.append(
            {
                "window_label": window_label,
                "window_start": str(datetime(start_period.year, start_period.month, 1).date()),
                "window_end": str(datetime(end_period.year, end_period.month, 1).date()),
                "trade_of_service": trade,
                "candidates_applied": applied,
                "screening_passed": passed,
                "screening_failed": failed,
                "screening_pass_rate_pct": round(screening_pass_rate, 1),
                "presented_to_client": presented,
                "presentation_rate_pct": round(presentation_rate, 1),
                "accepted_by_client": accepted,
                "rejected_by_client": rejected,
                "client_acceptance_rate_pct": round(client_accept_rate, 1),
                "hired": hired,
                "hire_rate_from_applied_pct": round(hire_rate, 1),
            }
        )

    return pd.DataFrame(rows).sort_values("candidates_applied", ascending=False)


def write_trade_rollups(
    *,
    outdir: str,
    sheet_id: str | None,
    tab_base: str,
    summary_only: bool,
) -> None:
    pattern = os.path.join(outdir, "recruiter_funnel_trades_????_??.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return

    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    trade_df = pd.concat(frames, ignore_index=True)

    trade_df["cohort_start"] = pd.to_datetime(trade_df["cohort_start"], errors="coerce")
    cohort_start_series = cast(pd.Series, trade_df["cohort_start"])
    latest_start = cohort_start_series.max()
    if pd.isna(latest_start):
        return
    latest_label = format_month_label(latest_start.year, latest_start.month)

    current_df = trade_df[trade_df["cohort_start"] == latest_start].copy()
    current_df = current_df.assign(
        window_label=f"Current Month ({latest_label})",
        window_start=current_df["cohort_start"],
        window_end=current_df["cohort_end"],
    )

    window_frames = [current_df]
    for window in (3, 6):
        rollup_df = build_trade_rollup(trade_df, window)
        if not rollup_df.empty:
            window_frames.append(rollup_df)

    summary_df = pd.concat(window_frames, ignore_index=True)
    summary_cols = [
        "window_label",
        "window_start",
        "window_end",
        "trade_of_service",
        "candidates_applied",
        "screening_passed",
        "screening_failed",
        "screening_pass_rate_pct",
        "presented_to_client",
        "presentation_rate_pct",
        "accepted_by_client",
        "rejected_by_client",
        "client_acceptance_rate_pct",
        "hired",
        "hire_rate_from_applied_pct",
    ]
    summary_df = summary_df.reindex(columns=summary_cols)

    summary_path = os.path.join(outdir, "recruiter_funnel_trades_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Trade Summary CSV: {summary_path}")

    if sheet_id and not summary_only:
        write_dataframe_to_sheet(sheet_id, tab_base, summary_df)


def load_gspread_client():
    import importlib

    try:
        gspread = importlib.import_module("gspread")
    except ImportError as exc:
        raise ImportError("gspread is required for Google Sheets sync. Run: pip install gspread") from exc

    creds = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not creds:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS is required for Google Sheets sync.")
    if os.path.isfile(creds):
        return gspread.service_account(filename=creds)
    info = json.loads(creds)
    return gspread.service_account_from_dict(info)


def apply_sheet_formatting(ws, rows: int, cols: int, headers: list[str]) -> None:
    sheet_id = ws._properties.get("sheetId")
    if sheet_id is None:
        return
    sh = ws.spreadsheet
    requests: list[dict] = [
        {
            "updateSheetProperties": {
                "properties": {"sheetId": sheet_id, "gridProperties": {"frozenRowCount": 1}},
                "fields": "gridProperties.frozenRowCount",
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": cols,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.92, "green": 0.95, "blue": 0.98},
                        "textFormat": {"bold": True},
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
            }
        },
    ]

    percent_headers = {"acceptance_rate", "application_rate", "screening_pass_rate", "presentation_rate", "client_acceptance_rate", "hire_rate"}
    count_headers = {
        "connection_requests_est",
        "connections_accepted",
        "candidates_applied",
        "screening_passed",
        "screening_failed",
        "presented_to_client",
        "accepted_by_client",
        "rejected_by_client",
        "hired",
    }
    date_headers = {"cohort_start", "cohort_end", "window_start", "window_end"}

    for idx, header in enumerate(headers):
        h = header.strip().lower()
        if h in {"metric", "definition"}:
            width = 360
        elif "trade" in h:
            width = 200
        elif "month" in h:
            width = 140
        elif h in date_headers or "date" in h:
            width = 130
        else:
            width = 120

        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": idx,
                        "endIndex": idx + 1,
                    },
                    "properties": {"pixelSize": width},
                    "fields": "pixelSize",
                }
            }
        )

        alignment = "LEFT" if h in {"metric", "definition", "trade_of_service"} else "RIGHT"
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": rows,
                        "startColumnIndex": idx,
                        "endColumnIndex": idx + 1,
                    },
                    "cell": {"userEnteredFormat": {"horizontalAlignment": alignment}},
                    "fields": "userEnteredFormat(horizontalAlignment)",
                }
            }
        )

        if h in date_headers or "date" in h:
            requests.append(
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1,
                            "endRowIndex": rows,
                            "startColumnIndex": idx,
                            "endColumnIndex": idx + 1,
                        },
                        "cell": {
                            "userEnteredFormat": {"numberFormat": {"type": "DATE", "pattern": "yyyy-mm-dd"}}
                        },
                        "fields": "userEnteredFormat.numberFormat",
                    }
                }
            )
        if any(k in h for k in percent_headers) or h.endswith("_pct") or "rate" in h or "pct" in h:
            requests.append(
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1,
                            "endRowIndex": rows,
                            "startColumnIndex": idx,
                            "endColumnIndex": idx + 1,
                        },
                        "cell": {
                            "userEnteredFormat": {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}}
                        },
                        "fields": "userEnteredFormat.numberFormat",
                    }
                }
            )
        elif h in count_headers:
            requests.append(
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1,
                            "endRowIndex": rows,
                            "startColumnIndex": idx,
                            "endColumnIndex": idx + 1,
                        },
                        "cell": {
                            "userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": "#,##0"}}
                        },
                        "fields": "userEnteredFormat.numberFormat",
                    }
                }
            )

    metadata = sh.fetch_sheet_metadata()
    has_banding = False
    for sheet in metadata.get("sheets", []):
        props = sheet.get("properties", {})
        if props.get("sheetId") == sheet_id:
            has_banding = bool(sheet.get("bandedRanges"))
            break

    if not has_banding:
        requests.append(
            {
                "addBanding": {
                    "bandedRange": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 0,
                            "endRowIndex": rows,
                            "startColumnIndex": 0,
                            "endColumnIndex": cols,
                        },
                        "rowProperties": {
                            "headerColor": {"red": 0.92, "green": 0.95, "blue": 0.98},
                            "firstBandColor": {"red": 1, "green": 1, "blue": 1},
                            "secondBandColor": {"red": 0.98, "green": 0.98, "blue": 0.99},
                        },
                    }
                }
            }
        )

    sh.batch_update({"requests": requests})


def write_dataframe_to_sheet(sheet_id: str, tab_name: str, df: pd.DataFrame) -> None:
    def sanitize_for_sheet(dataframe: pd.DataFrame) -> pd.DataFrame:
        out = dataframe.copy()
        percent_cols = []
        for col in out.columns:
            h = str(col).strip().lower()
            if h.endswith("_pct") or "rate" in h or "pct" in h:
                percent_cols.append(col)
        for col in out.columns:
            if is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d")
            elif out[col].dtype == "object":
                out[col] = out[col].apply(
                    lambda v: v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v
                )
            if col in percent_cols:
                series = cast(pd.Series, out[col])
                if series.dtype == "object":
                    s = series.astype(str).str.replace("%", "", regex=False)
                    s = s.str.replace(",", "", regex=False).str.strip()
                    numeric = pd.to_numeric(s, errors="coerce")
                else:
                    numeric = pd.to_numeric(series, errors="coerce")
                numeric_series = cast(pd.Series, pd.Series(numeric, index=series.index))
                if numeric_series.notna().any():
                    # All _pct / rate columns use 0-100 scale from safe_rate();
                    # always convert to 0-1 for Google Sheets PERCENT format.
                    numeric_series = numeric_series.apply(
                        lambda v: v / 100 if pd.notna(v) else v
                    )
                    out[col] = numeric_series
        return out

    gc = load_gspread_client()
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab_name)
    except Exception:
        rows = max(100, len(df) + 1)
        cols = max(20, len(df.columns))
        ws = sh.add_worksheet(title=tab_name, rows=str(rows), cols=str(cols))
    rows = len(df) + 1
    cols = len(df.columns)
    ws.resize(rows=max(100, rows), cols=max(20, cols))
    ws.clear()
    safe_df = sanitize_for_sheet(df)
    data = [safe_df.columns.tolist()] + safe_df.where(pd.notna(safe_df), "").values.tolist()
    ws.update(data, value_input_option="USER_ENTERED")
    apply_sheet_formatting(ws, rows=rows, cols=cols, headers=safe_df.columns.tolist())


def sync_outputs_to_sheet(
    *,
    sheet_id: str,
    summary_path: str,
    detail_path: str | None,
    summary_tab: str,
    detail_tab: str,
    summary_only: bool,
    definitions_tab: str,
) -> None:
    summary_df = pd.read_csv(summary_path, low_memory=False)
    write_dataframe_to_sheet(sheet_id, summary_tab, summary_df)
    definitions_df = build_definitions_table()
    write_dataframe_to_sheet(sheet_id, definitions_tab, definitions_df)
    if summary_only or not detail_path:
        return
    detail_df = pd.read_csv(detail_path, low_memory=False)
    write_dataframe_to_sheet(sheet_id, detail_tab, detail_df)


def compute_funnel_for_month(
    *,
    year: int,
    month: int,
    cand: pd.DataFrame,
    conn: pd.DataFrame,
    inv: pd.DataFrame,
    deals_x: pd.DataFrame,
    outdir: str,
    cand_link_col: str,
    cand_id_col: str,
    cand_created_col: str,
    screening_col: str,
    conn_link_col: str,
    conn_date_col: str,
    inv_dir_col: str,
    inv_inviter_col: str,
    inv_invitee_col: str,
    dx_candidate_col: str,
    dx_status_col: str,
    vendor_filter: str | None,
    vendor_id_filter: str | None,
    vendor_col: str | None,
    vendor_id_col: str | None,
    interview_date_col: str | None,
    dx_modified_col: str | None,
    trade_breakdown: bool,
    trade_col: str | None,
) -> dict:
    os.makedirs(outdir, exist_ok=True)

    month_start = datetime(year, month, 1)
    month_end = datetime(year, month, monthrange(year, month)[1], 23, 59, 59)

    cand_work = cand.copy()

    cand_link_series = cast(pd.Series, cand_work[cand_link_col])
    cand_work["linkedin_clean"] = normalize_linkedin_url(cand_link_series)
    cand_work = cand_work[cand_work["linkedin_clean"].notna() & (cand_work["linkedin_clean"] != "")].copy()

    if vendor_filter and vendor_col:
        vendor_series = cast(pd.Series, cand_work[vendor_col])
        cand_work = cand_work[vendor_series.astype(str).str.strip() == vendor_filter].copy()
    if vendor_id_filter and vendor_id_col:
        vendor_id_series = cast(pd.Series, cand_work[vendor_id_col])
        cand_work = cand_work[vendor_id_series.astype(str).str.strip() == vendor_id_filter].copy()

    conn_link_series = cast(pd.Series, conn[conn_link_col])
    conn_date_series = cast(pd.Series, conn[conn_date_col])
    conn_work = conn.copy()
    conn_work["linkedin_clean"] = normalize_linkedin_url(conn_link_series)
    conn_dt = parse_dt_flex(conn_date_series)
    conn_work["_connected_dt"] = conn_dt
    conn_month = conn_work[
        conn_work["linkedin_clean"].notna()
        & (conn_work["linkedin_clean"] != "")
        & (conn_work["_connected_dt"].notna())
        & (conn_work["_connected_dt"] >= month_start)
        & (conn_work["_connected_dt"] <= month_end)
    ].copy()

    conn_month_urls = set(conn_month["linkedin_clean"].tolist())
    if len(conn_month_urls) == 0:
        print(f"No connections found for {year:04d}-{month:02d} in Connections.csv")

    inv_dir_series = cast(pd.Series, inv[inv_dir_col])
    inv_inviter_series = cast(pd.Series, inv[inv_inviter_col])
    inv_invitee_series = cast(pd.Series, inv[inv_invitee_col])
    inv_dir = inv_dir_series.fillna("").astype(str).str.strip().str.upper()
    inv_inviter = normalize_linkedin_url(inv_inviter_series)
    inv_invitee = normalize_linkedin_url(inv_invitee_series)

    outgoing_mask = inv_dir.isin(["OUTGOING", "SENT"])
    incoming_mask = inv_dir.isin(["INCOMING", "RECEIVED"])

    outgoing_set = set(inv_invitee.loc[outgoing_mask].dropna().tolist())
    incoming_set = set(inv_inviter.loc[incoming_mask].dropna().tolist())

    accepted_outgoing = conn_month_urls & outgoing_set
    accepted_incoming = conn_month_urls & incoming_set
    accepted_unknown = conn_month_urls - accepted_outgoing - accepted_incoming

    connections_accepted = len(accepted_outgoing) + len(accepted_unknown)

    # Flat estimate per client brief: 250 requests/week * 4 weeks = 1000/month
    requests_est = 1000

    cand_created_series = cast(pd.Series, cand_work[cand_created_col])
    cand_created_dt = parse_dt_flex(cand_created_series)
    cand_work["_created_dt"] = cand_created_dt
    linkedin_clean_series = cast(pd.Series, cand_work["linkedin_clean"])
    created_dt_series = cast(pd.Series, cand_work["_created_dt"])
    cohort_mask = (
        linkedin_clean_series.isin(list(accepted_outgoing | accepted_unknown))
        & created_dt_series.notna()
        & (created_dt_series >= month_start)
    )
    cohort_cand = cand_work[cohort_mask].copy()

    linkedin_clean_series = cast(pd.Series, cohort_cand["linkedin_clean"])
    cohort_linkedin = set(linkedin_clean_series.tolist())
    candidates_applied = len(cohort_linkedin)

    screening_series = cast(pd.Series, cohort_cand[screening_col])
    screening_raw = screening_series.fillna("").astype(str).str.strip()
    screening_df = pd.DataFrame(
        {
            "linkedin_clean": linkedin_clean_series,
            "passed": screening_raw == "Passed",
            "failed": screening_raw == "Fail",
        }
    )
    screening_agg = screening_df.groupby("linkedin_clean")[["passed", "failed"]].any()
    screening_agg = cast(pd.DataFrame, screening_agg)

    passed_set = set(screening_agg[screening_agg["passed"]].index)
    failed_set = set(screening_agg[(~screening_agg["passed"]) & screening_agg["failed"]].index)
    screening_passed = len(passed_set)
    screening_failed = len(failed_set)

    cohort_id_series = cast(pd.Series, cohort_cand[cand_id_col])
    cohort_ids = set(cohort_id_series.astype(str).str.strip().tolist())
    cand_id_to_linkedin = dict(zip(cohort_id_series.astype(str).str.strip().tolist(), linkedin_clean_series.tolist()))

    dx = deals_x.copy()
    dx_candidate_series = cast(pd.Series, dx[dx_candidate_col])
    dx_status_series = cast(pd.Series, dx[dx_status_col])
    dx["_cand_id"] = dx_candidate_series.astype(str).str.strip()
    dx_status = dx_status_series.fillna("").astype(str).str.strip()
    dx_cand_id_series = cast(pd.Series, dx["_cand_id"])
    dx_in_cohort = dx.loc[dx_cand_id_series.isin(list(cohort_ids))].copy()
    dx_in_cohort["_status"] = dx_status[dx_in_cohort.index]
    dx_in_cohort["_linkedin"] = dx_in_cohort["_cand_id"].map(cand_id_to_linkedin)
    dx_in_cohort = dx_in_cohort[dx_in_cohort["_linkedin"].notna()].copy()

    presented_statuses = {
        "200 - Review by Customer",
        "300 - Accepted by Customer",
        "400 - Hired",
        "500 - Rejected by Customer",
        "301 - Screening Process",
        "302 - Job Offer",
    }
    accepted_status = "300 - Accepted by Customer"
    rejected_status = "500 - Rejected by Customer"
    hired_status = "400 - Hired"

    status_series = cast(pd.Series, dx_in_cohort["_status"]) if not dx_in_cohort.empty else pd.Series([], dtype=str)
    linkedin_series = (
        cast(pd.Series, dx_in_cohort["_linkedin"]).fillna("") if not dx_in_cohort.empty else pd.Series([], dtype=str)
    )
    presented_linkedin = set(linkedin_series[status_series.isin(list(presented_statuses))].tolist())
    accepted_linkedin = set(linkedin_series[status_series == accepted_status].tolist())
    rejected_linkedin = set(linkedin_series[status_series == rejected_status].tolist())
    hired_linkedin = set(linkedin_series[status_series == hired_status].tolist())

    presented_linkedin = presented_linkedin & passed_set
    accepted_linkedin = accepted_linkedin & passed_set
    rejected_linkedin = rejected_linkedin & passed_set
    hired_linkedin = hired_linkedin & passed_set

    presented_count = len(presented_linkedin)
    accepted_count = len(accepted_linkedin)
    rejected_count = len(rejected_linkedin)
    hired_count = len(hired_linkedin)

    acceptance_rate = safe_rate(connections_accepted, requests_est)
    application_rate = safe_rate(candidates_applied, connections_accepted)
    screening_pass_rate = safe_rate(screening_passed, screening_passed + screening_failed)
    presentation_rate = safe_rate(presented_count, screening_passed)
    client_accept_rate = safe_rate(accepted_count, presented_count)
    hire_rate = safe_rate(hired_count, connections_accepted)

    month_label = format_month_label(year, month)
    rows = [
        ("Cohort Month", month_label),
        ("Cohort Date Range", f"{month_label} ({month_start.date()} to {month_end.date()})"),
        ("Connection Requests Sent (est.)", requests_est),
        ("Connections Accepted", connections_accepted),
        ("Acceptance Rate", f"{acceptance_rate:.1f}%"),
        ("Candidates Who Applied", candidates_applied),
        ("Application Rate", f"{application_rate:.1f}%"),
        ("Screening — Passed", screening_passed),
        ("Screening — Failed", screening_failed),
        ("Screening Pass Rate", f"{screening_pass_rate:.1f}%"),
        ("Presented to Client", presented_count),
        ("Presentation Rate", f"{presentation_rate:.1f}%"),
        ("Accepted by Client", accepted_count),
        ("Rejected by Client", rejected_count),
        ("Client Acceptance Rate", f"{client_accept_rate:.1f}%"),
        ("Hired", hired_count),
        ("Hire Rate (Top-of-Funnel)", f"{hire_rate:.1f}%"),
    ]

    extra = {
        "connections_outgoing": len(accepted_outgoing),
        "connections_incoming": len(accepted_incoming),
        "connections_unknown": len(accepted_unknown),
        "cohort_month": f"{year:04d}-{month:02d}",
        "cohort_month_label": month_label,
        "vendor_filter": vendor_filter,
        "vendor_id_filter": vendor_id_filter,
        "screening_date_column": interview_date_col or "",
        "deals_stage_timestamp_column": dx_modified_col or "",
    }

    out_csv = os.path.join(outdir, f"recruiter_funnel_{year:04d}_{month:02d}.csv")
    out_json = os.path.join(outdir, f"recruiter_funnel_{year:04d}_{month:02d}.json")

    metrics = [k for k, _ in rows]
    values = [v for _, v in rows]
    pd.DataFrame({"Metric": metrics, "Value": values}).to_csv(out_csv, index=False)

    payload = {
        "month": f"{year:04d}-{month:02d}",
        "metrics": {k: v for k, v in rows},
        "extra": extra,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== Funnel Output ===")
    for k, v in rows:
        print(f"{k}: {v}")
    print("")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")

    summary_row = {
        "cohort_month": month_label,
        "cohort_start": str(month_start.date()),
        "cohort_end": str(month_end.date()),
        "connection_requests_est": requests_est,
        "connections_accepted": connections_accepted,
        "acceptance_rate_pct": round(acceptance_rate, 1),
        "candidates_applied": candidates_applied,
        "application_rate_pct": round(application_rate, 1),
        "screening_passed": screening_passed,
        "screening_failed": screening_failed,
        "screening_pass_rate_pct": round(screening_pass_rate, 1),
        "presented_to_client": presented_count,
        "presentation_rate_pct": round(presentation_rate, 1),
        "accepted_by_client": accepted_count,
        "rejected_by_client": rejected_count,
        "client_acceptance_rate_pct": round(client_accept_rate, 1),
        "hired": hired_count,
        "hire_rate_pct": round(hire_rate, 1),
        "vendor_filter": vendor_filter or "",
        "vendor_id_filter": vendor_id_filter or "",
    }

    if trade_breakdown:
        if not trade_col:
            raise ValueError("Trade of Service column not found for trade breakdown.")
        trade_df = build_trade_breakdown(
            cohort_cand=cast(pd.DataFrame, cohort_cand),
            dx_in_cohort=dx_in_cohort,
            screening_agg=screening_agg,
            trade_col=trade_col,
            month_label=month_label,
            month_start=month_start,
            month_end=month_end,
            presented_statuses=presented_statuses,
            accepted_status=accepted_status,
            rejected_status=rejected_status,
            hired_status=hired_status,
        )
        trade_out = os.path.join(outdir, f"recruiter_funnel_trades_{year:04d}_{month:02d}.csv")
        trade_df.to_csv(trade_out, index=False)
        print(f"Trade Breakdown CSV: {trade_out}")

    return summary_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--connections", required=False)
    parser.add_argument("--invitations", required=False)
    parser.add_argument("--deals-x-cand", required=True)
    parser.add_argument("--deals", required=True)
    parser.add_argument("--month", type=int, required=False)
    parser.add_argument("--year", type=int, required=False)
    parser.add_argument("--outdir", default="./out")
    parser.add_argument("--recruiters-config", default=None, help="JSON config for batch recruiter runs.")
    parser.add_argument("--vendor", default=None, help="Optional vendor name to filter candidates.")
    parser.add_argument("--vendor-id", default=None, help="Optional vendor id to filter candidates.")
    parser.add_argument(
        "--all-months",
        action="store_true",
        help="Recompute all months found in Connections.csv (lagging refresh).",
    )
    parser.add_argument("--sheet-id", default=None, help="Google Sheet ID for syncing outputs.")
    parser.add_argument("--sheet-summary-tab", default="Summary", help="Tab name for summary table.")
    parser.add_argument("--sheet-detail-tab", default="Funnel", help="Tab name for detail table.")
    parser.add_argument("--sheet-definitions-tab", default="Definitions", help="Tab name for definitions table.")
    parser.add_argument("--sheet-trade-tab", default="Trades", help="Tab name for trade breakdown table.")
    parser.add_argument("--trade-breakdown", action="store_true", help="Output trade-of-service breakdowns.")
    parser.add_argument("--sheet-summary-only", action="store_true", help="Only sync summary tab.")
    args = parser.parse_args()

    cand = cast(pd.DataFrame, pd.read_csv(args.candidates, low_memory=False))
    deals_x = cast(pd.DataFrame, pd.read_csv(args.deals_x_cand, low_memory=False))
    pd.read_csv(args.deals, low_memory=False)

    cand_link_col = find_column(cand, ["linkedin", "linkedin url", "linkedin_url", "profile url", "profile"])
    cand_id_col = find_column(cand, ["record id", "candidate id", "id"])
    cand_created_col = find_column(cand, ["created time", "created date", "created"])
    screening_col = find_column(cand, ["personal interview status"])
    interview_date_col = find_column(cand, ["interview date"])
    vendor_col = find_column(cand, ["vendor", "vendor name", "recruiter"])
    vendor_id_col = find_column(cand, ["vendor.id", "vendor id", "vendor_id"])
    trade_col = find_column(cand, ["trade of service", "trade of services", "trade", "service trade"])

    if cand_link_col is None:
        raise ValueError("Candidates CSV: LinkedIn column not found.")
    if cand_id_col is None:
        raise ValueError("Candidates CSV: Record Id column not found.")
    if cand_created_col is None:
        raise ValueError("Candidates CSV: Created Time column not found.")
    if screening_col is None:
        raise ValueError("Candidates CSV: Personal Interview Status column not found.")
    if args.trade_breakdown and trade_col is None:
        raise ValueError("Candidates CSV: Trade of Service column not found for trade breakdown.")

    dx_candidate_col = find_column(deals_x, ["candidates.id", "candidate id", "candidate_id"])
    dx_status_col = find_column(deals_x, ["status candidate"])
    dx_modified_col = find_column(deals_x, ["modified time", "modified"])
    if dx_candidate_col is None or dx_status_col is None:
        raise ValueError("Deals X Cand CSV: candidates.id or status column not found.")

    if args.recruiters_config:
        if not args.all_months and (args.month is None or args.year is None):
            raise ValueError("--month and --year are required unless --all-months is used.")
        recruiters = load_recruiters_config(args.recruiters_config)
        combined_rows = []

        for recruiter in recruiters:
            recruiter_name = recruiter["recruiter_name"]
            recruiter_slug = slugify_name(recruiter_name)
            recruiter_outdir = os.path.join(args.outdir, recruiter_slug)

            conn = cast(pd.DataFrame, pd.read_csv(recruiter["connections"], low_memory=False))
            inv = cast(pd.DataFrame, pd.read_csv(recruiter["invitations"], low_memory=False))

            conn_link_col = find_column(conn, ["url", "linkedin", "linkedin url", "profile url", "profile"])
            conn_date_col = find_column(conn, ["connected on", "connected_on", "connected"])
            if conn_link_col is None or conn_date_col is None:
                raise ValueError("Connections CSV: URL or Connected On column not found.")

            inv_dir_col = find_column(inv, ["direction"])
            inv_inviter_col = find_column(inv, ["inviterprofileurl", "inviter profile url", "inviter_profile_url"])
            inv_invitee_col = find_column(inv, ["inviteeprofileurl", "invitee profile url", "invitee_profile_url"])
            if inv_dir_col is None or inv_inviter_col is None or inv_invitee_col is None:
                raise ValueError("Invitations CSV: direction/inviter/invitee columns not found.")

            summary_rows = []
            if args.all_months:
                conn_date_series = cast(pd.Series, conn[conn_date_col])
                conn_dt = parse_dt_flex(conn_date_series)
                months = (
                    conn_dt.dropna()
                    .dt.to_period("M")
                    .sort_values()
                    .unique()
                    .tolist()
                )
                if not months:
                    raise ValueError("No valid connection dates found for month discovery.")
                for period in months:
                    summary_rows.append(
                        compute_funnel_for_month(
                            year=period.year,
                            month=period.month,
                            cand=cand,
                            conn=conn,
                            inv=inv,
                            deals_x=deals_x,
                            outdir=recruiter_outdir,
                            cand_link_col=cand_link_col,
                            cand_id_col=cand_id_col,
                            cand_created_col=cand_created_col,
                            screening_col=screening_col,
                            conn_link_col=conn_link_col,
                            conn_date_col=conn_date_col,
                            inv_dir_col=inv_dir_col,
                            inv_inviter_col=inv_inviter_col,
                            inv_invitee_col=inv_invitee_col,
                            dx_candidate_col=dx_candidate_col,
                            dx_status_col=dx_status_col,
                            vendor_filter=recruiter.get("vendor_filter") or args.vendor,
                            vendor_id_filter=recruiter.get("vendor_id_filter") or args.vendor_id,
                            vendor_col=vendor_col,
                            vendor_id_col=vendor_id_col,
                            interview_date_col=interview_date_col,
                            dx_modified_col=dx_modified_col,
                            trade_breakdown=args.trade_breakdown,
                            trade_col=trade_col,
                        )
                    )
            else:
                summary_rows.append(
                    compute_funnel_for_month(
                        year=args.year,
                        month=args.month,
                        cand=cand,
                        conn=conn,
                        inv=inv,
                        deals_x=deals_x,
                        outdir=recruiter_outdir,
                        cand_link_col=cand_link_col,
                        cand_id_col=cand_id_col,
                        cand_created_col=cand_created_col,
                        screening_col=screening_col,
                        conn_link_col=conn_link_col,
                        conn_date_col=conn_date_col,
                        inv_dir_col=inv_dir_col,
                        inv_inviter_col=inv_inviter_col,
                        inv_invitee_col=inv_invitee_col,
                        dx_candidate_col=dx_candidate_col,
                        dx_status_col=dx_status_col,
                        vendor_filter=recruiter.get("vendor_filter") or args.vendor,
                        vendor_id_filter=recruiter.get("vendor_id_filter") or args.vendor_id,
                        vendor_col=vendor_col,
                        vendor_id_col=vendor_id_col,
                        interview_date_col=interview_date_col,
                        dx_modified_col=dx_modified_col,
                        trade_breakdown=args.trade_breakdown,
                        trade_col=trade_col,
                    )
                )

            summary_df = pd.DataFrame(summary_rows).sort_values("cohort_start")
            summary_out = os.path.join(recruiter_outdir, "recruiter_funnel_summary.csv")
            summary_df.to_csv(summary_out, index=False)
            print(f"Summary CSV: {summary_out}")

            if args.trade_breakdown and args.all_months:
                write_trade_rollups(
                    outdir=recruiter_outdir,
                    sheet_id=args.sheet_id,
                    tab_base=f"{recruiter_name} {args.sheet_trade_tab}",
                    summary_only=args.sheet_summary_only,
                )

            for row in summary_rows:
                combined_rows.append({"recruiter_name": recruiter_name, **row})

            if args.sheet_id and not args.sheet_summary_only:
                write_dataframe_to_sheet(args.sheet_id, recruiter_name, summary_df)
                if args.trade_breakdown and not args.all_months:
                    latest_start = pd.to_datetime(summary_df["cohort_start"], errors="coerce").max()
                    if pd.isna(latest_start):
                        trade_month = datetime(args.year, args.month, 1)
                    else:
                        trade_month = datetime(latest_start.year, latest_start.month, 1)
                    trade_path = os.path.join(
                        recruiter_outdir,
                        f"recruiter_funnel_trades_{trade_month.year:04d}_{trade_month.month:02d}.csv",
                    )
                    if os.path.exists(trade_path):
                        trade_df = pd.read_csv(trade_path, low_memory=False)
                        write_dataframe_to_sheet(
                            args.sheet_id,
                            f"{recruiter_name} {args.sheet_trade_tab}",
                            trade_df,
                        )

        combined_df = pd.DataFrame(combined_rows).sort_values(["recruiter_name", "cohort_start"])
        combined_out = os.path.join(args.outdir, "recruiter_funnel_summary_all.csv")
        combined_df.to_csv(combined_out, index=False)
        print(f"Combined Summary CSV: {combined_out}")

        summary_cmp = build_summary_comparison(
            combined_df,
            recruiter_order=[r["recruiter_name"] for r in recruiters],
        )
        summary_cmp_out = os.path.join(args.outdir, "recruiter_summary_comparison.csv")
        summary_cmp.to_csv(summary_cmp_out, index=False)
        print(f"Summary Comparison CSV: {summary_cmp_out}")

        if args.sheet_id:
            write_dataframe_to_sheet(args.sheet_id, args.sheet_summary_tab, summary_cmp)
            definitions_df = build_definitions_table()
            write_dataframe_to_sheet(args.sheet_id, args.sheet_definitions_tab, definitions_df)
        return

    if not args.connections or not args.invitations:
        raise ValueError("--connections and --invitations are required unless --recruiters-config is used.")

    if not args.all_months and (args.month is None or args.year is None):
        raise ValueError("--month and --year are required unless --all-months is used.")

    conn = cast(pd.DataFrame, pd.read_csv(args.connections, low_memory=False))
    inv = cast(pd.DataFrame, pd.read_csv(args.invitations, low_memory=False))

    conn_link_col = find_column(conn, ["url", "linkedin", "linkedin url", "profile url", "profile"])
    conn_date_col = find_column(conn, ["connected on", "connected_on", "connected"])
    if conn_link_col is None or conn_date_col is None:
        raise ValueError("Connections CSV: URL or Connected On column not found.")

    inv_dir_col = find_column(inv, ["direction"])
    inv_inviter_col = find_column(inv, ["inviterprofileurl", "inviter profile url", "inviter_profile_url"])
    inv_invitee_col = find_column(inv, ["inviteeprofileurl", "invitee profile url", "invitee_profile_url"])
    if inv_dir_col is None or inv_inviter_col is None or inv_invitee_col is None:
        raise ValueError("Invitations CSV: direction/inviter/invitee columns not found.")

    summary_rows = []

    if args.all_months:
        conn_date_series = cast(pd.Series, conn[conn_date_col])
        conn_dt = parse_dt_flex(conn_date_series)
        months = (
            conn_dt.dropna()
            .dt.to_period("M")
            .sort_values()
            .unique()
            .tolist()
        )
        if not months:
            raise ValueError("No valid connection dates found for month discovery.")
        for period in months:
            summary_rows.append(
                compute_funnel_for_month(
                    year=period.year,
                    month=period.month,
                    cand=cand,
                    conn=conn,
                    inv=inv,
                    deals_x=deals_x,
                    outdir=args.outdir,
                    cand_link_col=cand_link_col,
                    cand_id_col=cand_id_col,
                    cand_created_col=cand_created_col,
                    screening_col=screening_col,
                    conn_link_col=conn_link_col,
                    conn_date_col=conn_date_col,
                    inv_dir_col=inv_dir_col,
                    inv_inviter_col=inv_inviter_col,
                    inv_invitee_col=inv_invitee_col,
                    dx_candidate_col=dx_candidate_col,
                    dx_status_col=dx_status_col,
                    vendor_filter=args.vendor,
                    vendor_id_filter=args.vendor_id,
                    vendor_col=vendor_col,
                    vendor_id_col=vendor_id_col,
                    interview_date_col=interview_date_col,
                    dx_modified_col=dx_modified_col,
                    trade_breakdown=args.trade_breakdown,
                    trade_col=trade_col,
                )
            )
        summary_df = pd.DataFrame(summary_rows).sort_values("cohort_start")
        summary_out = os.path.join(args.outdir, "recruiter_funnel_summary.csv")
        summary_df.to_csv(summary_out, index=False)
        print(f"Summary CSV: {summary_out}")
        if args.sheet_id:
            latest_period = months[-1]
            detail_path = os.path.join(
                args.outdir,
                f"recruiter_funnel_{latest_period.year:04d}_{latest_period.month:02d}.csv",
            )
            sync_outputs_to_sheet(
                sheet_id=args.sheet_id,
                summary_path=summary_out,
                detail_path=detail_path,
                summary_tab=args.sheet_summary_tab,
                detail_tab=args.sheet_detail_tab,
                summary_only=args.sheet_summary_only,
                definitions_tab=args.sheet_definitions_tab,
            )
            if args.trade_breakdown and not args.all_months:
                trade_path = os.path.join(
                    args.outdir,
                    f"recruiter_funnel_trades_{latest_period.year:04d}_{latest_period.month:02d}.csv",
                )
                if os.path.exists(trade_path):
                    trade_df = pd.read_csv(trade_path, low_memory=False)
                    write_dataframe_to_sheet(args.sheet_id, args.sheet_trade_tab, trade_df)
            if args.trade_breakdown and args.all_months:
                write_trade_rollups(
                    outdir=args.outdir,
                    sheet_id=args.sheet_id,
                    tab_base=args.sheet_trade_tab,
                    summary_only=args.sheet_summary_only,
                )
        return

    summary_rows.append(
        compute_funnel_for_month(
            year=args.year,
            month=args.month,
            cand=cand,
            conn=conn,
            inv=inv,
            deals_x=deals_x,
            outdir=args.outdir,
            cand_link_col=cand_link_col,
            cand_id_col=cand_id_col,
            cand_created_col=cand_created_col,
            screening_col=screening_col,
            conn_link_col=conn_link_col,
            conn_date_col=conn_date_col,
            inv_dir_col=inv_dir_col,
            inv_inviter_col=inv_inviter_col,
            inv_invitee_col=inv_invitee_col,
            dx_candidate_col=dx_candidate_col,
            dx_status_col=dx_status_col,
            vendor_filter=args.vendor,
            vendor_id_filter=args.vendor_id,
            vendor_col=vendor_col,
            vendor_id_col=vendor_id_col,
            interview_date_col=interview_date_col,
            dx_modified_col=dx_modified_col,
            trade_breakdown=args.trade_breakdown,
            trade_col=trade_col,
        )
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(args.outdir, "recruiter_funnel_summary.csv")
    summary_df.to_csv(summary_out, index=False)
    print(f"Summary CSV: {summary_out}")
    if args.sheet_id:
        detail_path = os.path.join(args.outdir, f"recruiter_funnel_{args.year:04d}_{args.month:02d}.csv")
        sync_outputs_to_sheet(
            sheet_id=args.sheet_id,
            summary_path=summary_out,
            detail_path=detail_path,
            summary_tab=args.sheet_summary_tab,
            detail_tab=args.sheet_detail_tab,
            summary_only=args.sheet_summary_only,
            definitions_tab=args.sheet_definitions_tab,
        )
        if args.trade_breakdown:
            trade_path = os.path.join(
                args.outdir,
                f"recruiter_funnel_trades_{args.year:04d}_{args.month:02d}.csv",
            )
            if os.path.exists(trade_path):
                trade_df = pd.read_csv(trade_path, low_memory=False)
                write_dataframe_to_sheet(args.sheet_id, args.sheet_trade_tab, trade_df)


if __name__ == "__main__":
    main()
