"""
Segmented matching + charting across MULTIPLE CSVs using LinkedIn profile URLs.

Inputs (required):
- Candidates CSV (expects a column like "Linkedin")
- Connections CSV (expects a column like "URL")
- Messages CSV (expects columns like "SENDER PROFILE URL" and "RECIPIENT PROFILE URLS")

What it does:
1) Normalizes LinkedIn URLs across all sources
2) Creates three boolean flags per candidate:
   - Is Connected (in Connections.csv)
   - Is Contacted (appears in Messages.csv sender/recipient URLs)
   - Is Connected AND Contacted
3) Exports:
   - candidates_matching_any.csv  -> candidates present in at least 1 external CSV (connections or messages)
   - candidates_matching_all.csv  -> candidates present in EVERY CSV (candidates + connections + messages)
   - messages_involving_candidates.csv -> messages involving any candidate LinkedIn URL
4) Produces highly segmented charts with clearer naming:
   - "Candidate Match Segmentation (Connected vs Contacted)"
   - "Connected Candidates (Matched) Over Time — Monthly"
   - "Contacted Candidates (Matched) Over Time — Monthly"
   - "Top Companies — Connected Candidates (Matched)"
   - "Top Positions — Connected Candidates (Matched)"
   - "Seniority Buckets — Connected Candidates (Matched)"
   - "Connection Recency — Connected Candidates (Matched)"
   - "Message Folders — Contacted Candidates (Matched Messages)"

Requirements:
    pip install pandas matplotlib

Usage:
    python chart_linkedin_matches.py \
      --candidates "/path/to/Candidates_2026_01_13.csv" \
      --connections "/path/to/Connections.csv" \
      --messages "/path/to/messages.csv" \
      --outdir "./charts" \
      --topn 20
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Helpers
# -----------------------------
def normalize_linkedin_url(s: pd.Series) -> pd.Series:
    """Normalize LinkedIn URLs for consistent matching."""
    x = s.astype(str).str.strip().str.lower()
    x = x.replace({"nan": "", "none": "", "null": ""})
    x = x.str.replace(r"[\?#].*$", "", regex=True)   # drop query/fragment
    x = x.str.replace(r"^http://", "https://", regex=True)
    x = x.str.replace(r"/+$", "", regex=True)
    return x


def find_column(df: pd.DataFrame, preferred: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in cols:
            return cols[p.lower()]
    return None


def safe_value_counts(s: pd.Series) -> pd.Series:
    s2 = s.fillna("").astype(str).str.strip()
    s2 = s2[s2 != ""]
    return s2.value_counts()


def format_count(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "0"
    return f"{int(round(v)):,}"


def format_compact(value, _pos=None) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{sign}{v / 1_000:.1f}k"
    if v >= 1:
        return f"{sign}{int(round(v))}"
    return f"{sign}{v:.2f}".rstrip("0").rstrip(".")


def add_footer(fig, note: str | None) -> None:
    if not note:
        return
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9, color="0.35")


def add_callout(ax, text: str | None) -> None:
    if not text:
        return
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8"},
    )


def bar_chart(
    series,
    title,
    xlabel,
    ylabel,
    outpath,
    topn=None,
    show_pct=False,
    total=None,
    note=None,
    callout=None,
):
    if series is None or len(series) == 0:
        return

    data = series.copy()
    if topn is not None:
        data = data.head(topn)

    fig, ax = plt.subplots(figsize=(14, 7))

    data.sort_values(ascending=True).plot(kind="barh", ax=ax)

    # --- FIX: wrap + pad title ---
    import textwrap
    wrapped_title = "\n".join(textwrap.wrap(title, width=55))
    ax.set_title(wrapped_title, fontsize=14, pad=20)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    max_val = float(data.max()) if len(data) else 0.0
    if max_val > 0:
        ax.set_xlim(0, max_val * 1.15)

    ax.xaxis.set_major_formatter(FuncFormatter(format_compact))

    pct_total = total if total is not None else (data.sum() if show_pct else None)
    for bar in ax.patches:
        value = bar.get_width()
        if value == 0:
            continue
        label = format_count(value)
        if show_pct and pct_total:
            pct = (value / pct_total) * 100
            label = f"{label} ({pct:.1f}%)"
        ax.text(
            value + (max_val * 0.01 if max_val else 0.01),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9,
        )

    add_callout(ax, callout)
    add_footer(fig, note)

    # Ensure enough space for title and footer
    plt.subplots_adjust(top=0.88, left=0.35, bottom=0.12)

    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def configure_year_month_axis(ax, min_dt: datetime, max_dt: datetime) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(
        axis="x",
        which="major",
        pad=16,
        labelsize=10,
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
    )
    ax.tick_params(axis="x", which="minor", labelsize=8, rotation=45, bottom=True, labelbottom=True)
    ax.grid(which="minor", axis="x", linestyle=":", alpha=0.15)
    for year in range(min_dt.year, max_dt.year + 1):
        ax.axvline(datetime(year, 1, 1), color="0.85", lw=1, zorder=0)


def line_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: str, note=None, callout=None):
    if series is None or len(series) == 0:
        return

    idx = series.index
    if isinstance(idx, pd.PeriodIndex):
        dt_index = idx.to_timestamp()
    else:
        dt_index = pd.to_datetime(idx.astype(str), errors="coerce")
    series_dt = series.copy()
    series_dt.index = dt_index
    series_dt = series_dt[series_dt.index.notna()].sort_index()
    if series_dt.empty:
        return

    min_dt = series_dt.index.min()
    max_dt = series_dt.index.max()
    full_index = pd.date_range(min_dt.to_period("M").to_timestamp(), max_dt.to_period("M").to_timestamp(), freq="MS")
    series_dt = series_dt.reindex(full_index).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series_dt.index, series_dt.values, marker="o", linewidth=2, markersize=4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(format_compact))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    configure_year_month_axis(ax, min_dt, max_dt)

    values = series_dt.values
    points = len(values)
    max_val = float(values.max()) if points else 0.0
    y_top = max_val * 1.2 if max_val > 0 else 1
    ax.set_ylim(0, y_top)

    quarter = series_dt.index.to_period("Q")
    quarter_max = {}
    for i, (q, y) in enumerate(zip(quarter, values)):
        if q not in quarter_max or y > quarter_max[q][1]:
            quarter_max[q] = (i, y)

    for q in sorted(quarter_max.keys()):
        i, y = quarter_max[q]
        if y < 10:
            continue
        x = series_dt.index[i]
        ax.annotate(
            format_count(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            arrowprops={"arrowstyle": "-|>", "color": "0.3", "lw": 0.8},
        )

    add_callout(ax, callout)
    footer_note = note or ""
    if "Months with no activity shown as 0." not in footer_note:
        footer_note = (footer_note + " " if footer_note else "") + "Months with no activity shown as 0."
    if "Quarterly peaks are labeled." not in footer_note:
        footer_note = (footer_note + " " if footer_note else "") + "Quarterly peaks are labeled."
    add_footer(fig, footer_note)

    plt.subplots_adjust(top=0.85, bottom=0.18)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def line_charts_by_year(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: str, note=None) -> None:
    if series is None or len(series) == 0:
        return

    idx = series.index
    if isinstance(idx, pd.PeriodIndex):
        dt_index = idx.to_timestamp()
    else:
        dt_index = pd.to_datetime(idx.astype(str), errors="coerce")
    series_dt = series.copy()
    series_dt.index = dt_index
    series_dt = series_dt[series_dt.index.notna()].sort_index()
    if series_dt.empty:
        return

    years = sorted(series_dt.index.year.unique().tolist())
    base, ext = os.path.splitext(outpath)
    for year in years:
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 1)
        full_index = pd.date_range(start, end, freq="MS")
        year_series = series_dt[series_dt.index.year == year].reindex(full_index).fillna(0)
        year_title = f"{title} ({year})"
        year_outpath = f"{base}_{year}{ext}"
        year_callout = f"Total in {year}: {format_count(year_series.sum())}"
        line_chart(year_series, year_title, xlabel, ylabel, year_outpath, note=note, callout=year_callout)


def seniority_bucket(title: str) -> str:
    """Heuristic seniority bucketing from a job title string."""
    t = (title or "").lower()
    if any(k in t for k in ["chief", "cxo", "cto", "ceo", "coo", "cfo", "vp", "vice president", "head of"]):
        return "Executive / VP / Head"
    if any(k in t for k in ["director", "principal", "staff"]):
        return "Director / Principal / Staff"
    if any(k in t for k in ["manager", "lead", "supervisor"]):
        return "Manager / Lead"
    if any(k in t for k in ["senior", "sr."]):
        return "Senior IC"
    if any(k in t for k in ["junior", "jr.", "entry", "intern", "trainee"]):
        return "Junior / Entry"
    if t.strip() == "":
        return "Unknown"
    return "IC (Other)"


def explode_profile_urls(series: pd.Series) -> pd.Series:
    """
    Split multi-recipient URL cells into a flat Series.
    Default split is comma; add semicolon if your export uses it.
    """
    s = series.fillna("").astype(str)
    parts = s.str.split(",")
    exploded = parts.explode().astype(str).str.strip()
    exploded = exploded[exploded != ""]
    return exploded


def parse_dt_flex(series: pd.Series) -> pd.Series:
    """Parse date/time with a few common fallbacks."""
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().any():
        return dt
    # e.g., LinkedIn "Connected On": 9-Oct-23
    dt2 = pd.to_datetime(series, errors="coerce", format="%d-%b-%y")
    return dt2


def file_signature(path: str) -> dict | None:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return {"path": os.path.abspath(path), "mtime": int(st.st_mtime), "size": st.st_size}


def build_manifest(args: argparse.Namespace) -> dict:
    return {
        "script": os.path.basename(__file__),
        "inputs": {
            "candidates": file_signature(args.candidates),
            "connections": file_signature(args.connections),
            "messages": file_signature(args.messages),
            "topn": args.topn,
            "today": args.today,
        },
    }


def max_mtime(paths: list[str]) -> int | None:
    mtimes = []
    for p in paths:
        try:
            mtimes.append(int(os.path.getmtime(p)))
        except OSError:
            return None
    return max(mtimes) if mtimes else None


def outputs_fresh(output_paths: list[str], input_paths: list[str]) -> bool:
    inputs_mtime = max_mtime(input_paths)
    outputs_mtime = max_mtime(output_paths)
    if inputs_mtime is None or outputs_mtime is None:
        return False
    return outputs_mtime >= inputs_mtime


def list_chart_paths(outdir: str) -> list[str]:
    if not os.path.isdir(outdir):
        return []
    return sorted(
        os.path.join(outdir, f)
        for f in os.listdir(outdir)
        if f.lower().endswith(".png")
    )


def ensure_match_segment(enriched: pd.DataFrame) -> None:
    if "match_segment" in enriched.columns:
        return
    if "is_connected_match" not in enriched.columns or "is_contacted_match" not in enriched.columns:
        return

    def segment_row(r):
        c = bool(r["is_connected_match"])
        m = bool(r["is_contacted_match"])
        if c and m:
            return "Connected + Contacted (Matched)"
        if c and not m:
            return "Connected Only (Matched)"
        if (not c) and m:
            return "Contacted Only (Matched)"
        return "Not Matched (Neither)"

    enriched["match_segment"] = enriched.apply(segment_row, axis=1)


def render_charts(
    enriched: pd.DataFrame,
    msg_extract: pd.DataFrame,
    conn_date_col: str | None,
    conn_company_col: str | None,
    conn_position_col: str | None,
    msg_date_col: str | None,
    msg_folder_col: str | None,
    args: argparse.Namespace,
) -> None:
    ensure_match_segment(enriched)
    matched_note = "Matched = candidate LinkedIn URL found in Candidates + referenced file."

    # Chart 1: Match segmentation
    seg_counts = enriched["match_segment"].value_counts()
    bar_chart(
        seg_counts,
        title="Candidate Match Segmentation (Connected vs Contacted)",
        xlabel="Candidates",
        ylabel="Match segment",
        outpath=os.path.join(args.outdir, "candidate_match_segmentation_connected_vs_contacted.png"),
        topn=None,
        show_pct=True,
        total=len(enriched),
        note="Share of candidates matched against connections or messages. " + matched_note,
        callout=f"Total candidates: {format_count(len(enriched))}",
    )

    # Chart 2: Connected candidates (matched) over time — monthly
    if conn_date_col is not None and conn_date_col in enriched.columns:
        connected_dt = parse_dt_flex(enriched[conn_date_col])
        enriched["_connected_dt"] = connected_dt

        connected_only = enriched[enriched["is_connected_match"] & enriched["_connected_dt"].notna()].copy()
        if len(connected_only) > 0:
            monthly_conn = (
                connected_only.assign(month=lambda d: d["_connected_dt"].dt.to_period("M").astype(str))
                .groupby("month")
                .size()
            )
            line_charts_by_year(
                monthly_conn,
                title="Connected Candidates (Matched) Over Time — Monthly",
                xlabel="Month",
                ylabel="Connected candidates",
                outpath=os.path.join(args.outdir, "connected_candidates_matched_over_time_monthly.png"),
                note="Monthly count of connected candidates present in the input list. " + matched_note,
            )

            # Connection recency buckets
            today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else datetime.today().date()
            d = connected_only.copy()
            d["_days_ago"] = (pd.Timestamp(today) - d["_connected_dt"]).dt.days

            def recency_bucket(days: int) -> str:
                if days < 0:
                    return "Future/Invalid"
                if days <= 30:
                    return "0–30 days"
                if days <= 90:
                    return "31–90 days"
                if days <= 180:
                    return "91–180 days"
                if days <= 365:
                    return "181–365 days"
                return "365+ days"

            d["_recency_bucket"] = d["_days_ago"].map(recency_bucket)
            recency_counts = d["_recency_bucket"].value_counts().reindex(
                ["0–30 days", "31–90 days", "91–180 days", "181–365 days", "365+ days", "Future/Invalid"]
            ).dropna()

            bar_chart(
                recency_counts,
                title="Connection Recency — Connected Candidates (Matched)",
                xlabel="Connected candidates",
                ylabel="Recency bucket",
                outpath=os.path.join(args.outdir, "connection_recency_connected_candidates_matched.png"),
                topn=None,
                show_pct=True,
                total=len(connected_only),
                note="How recently matched connections were accepted. " + matched_note,
                callout=f"Total connected matches: {format_count(len(connected_only))}",
            )

    # Chart 3: Contacted candidates (matched) over time — monthly (based on message date, if available)
    if msg_date_col is not None and msg_date_col in msg_extract.columns:
        msg_extract["_msg_dt"] = parse_dt_flex(msg_extract[msg_date_col])
        if msg_extract["_msg_dt"].notna().any():
            monthly_msgs = (
                msg_extract.dropna(subset=["_msg_dt"])
                .assign(month=lambda d: d["_msg_dt"].dt.to_period("M").astype(str))
                .groupby("month")
                .size()
            )
            line_charts_by_year(
                monthly_msgs,
                title="Contacted Candidates (Matched) Over Time — Monthly",
                xlabel="Month",
                ylabel="Messages involving matched candidates",
                outpath=os.path.join(args.outdir, "contacted_candidates_matched_over_time_monthly.png"),
                note="Monthly count of messages that include a matched candidate. " + matched_note,
            )

    # Chart 4: Top companies among CONNECTED candidates (matched)
    if conn_company_col is not None and conn_company_col in enriched.columns:
        connected = enriched[enriched["is_connected_match"]].copy()
        company_counts = safe_value_counts(connected[conn_company_col])
        bar_chart(
            company_counts,
            title=f"Top Companies — Connected Candidates (Matched) (Top {args.topn})",
            xlabel="Connected candidates",
            ylabel="Company",
            outpath=os.path.join(args.outdir, "top_companies_connected_candidates_matched.png"),
            topn=args.topn,
            note="Most common companies among matched connections. " + matched_note,
            callout=f"Total connected matches: {format_count(len(connected))}",
        )

    # Chart 5: Top positions among CONNECTED candidates (matched) + seniority
    if conn_position_col is not None and conn_position_col in enriched.columns:
        connected = enriched[enriched["is_connected_match"]].copy()
        pos_counts = safe_value_counts(connected[conn_position_col])
        bar_chart(
            pos_counts,
            title=f"Top Positions — Connected Candidates (Matched) (Top {args.topn})",
            xlabel="Connected candidates",
            ylabel="Position",
            outpath=os.path.join(args.outdir, "top_positions_connected_candidates_matched.png"),
            topn=args.topn,
            note="Most common titles among matched connections. " + matched_note,
            callout=f"Total connected matches: {format_count(len(connected))}",
        )

        connected["_seniority_bucket"] = connected[conn_position_col].fillna("").astype(str).map(seniority_bucket)
        seniority_counts = connected["_seniority_bucket"].value_counts()
        bar_chart(
            seniority_counts,
            title="Seniority Buckets — Connected Candidates (Matched)",
            xlabel="Connected candidates",
            ylabel="Seniority bucket",
            outpath=os.path.join(args.outdir, "seniority_buckets_connected_candidates_matched.png"),
            topn=None,
            show_pct=True,
            total=len(connected),
            note="Distribution of seniority levels for matched connections. " + matched_note,
            callout=f"Total connected matches: {format_count(len(connected))}",
        )

    # Chart 6: Message folders distribution for candidate-involved messages
    if msg_folder_col is not None and msg_folder_col in msg_extract.columns:
        folder_counts = safe_value_counts(msg_extract[msg_folder_col])
        bar_chart(
            folder_counts,
            title="Message Folders — Contacted Candidates (Matched Messages)",
            xlabel="Messages",
            ylabel="Folder",
            outpath=os.path.join(args.outdir, "message_folders_contacted_candidates_matched.png"),
            topn=30,
            show_pct=True,
            total=len(msg_extract),
            note="Share of matched messages by folder. " + matched_note,
            callout=f"Total matched messages: {format_count(len(msg_extract))}",
        )


def try_regenerate_charts_from_cache(args: argparse.Namespace, manifest_path: str) -> bool:
    input_paths = [args.candidates, args.connections, args.messages]
    enriched_out = os.path.join(args.outdir, "candidates_enriched_matches.csv")
    any_out = os.path.join(args.outdir, "candidates_matching_any.csv")
    all_out = os.path.join(args.outdir, "candidates_matching_all.csv")
    msg_extract_out = os.path.join(args.outdir, "messages_involving_candidates.csv")
    expected_outputs = [enriched_out, msg_extract_out]

    if not outputs_fresh(expected_outputs, input_paths):
        return False

    try:
        enriched = pd.read_csv(enriched_out)
        msg_extract = pd.read_csv(msg_extract_out)
    except OSError:
        return False

    conn_date_col = find_column(enriched, ["connected on", "connected_on", "connected"])
    conn_company_col = find_column(enriched, ["company", "current company", "organization"])
    conn_position_col = find_column(enriched, ["position", "title", "role"])
    msg_date_col = find_column(msg_extract, ["date", "sent date", "timestamp", "time"])
    msg_folder_col = find_column(msg_extract, ["folder", "inbox folder", "label"])

    print("Analysis already exists for these inputs; updating charts from cached outputs.")
    render_charts(
        enriched=enriched,
        msg_extract=msg_extract,
        conn_date_col=conn_date_col,
        conn_company_col=conn_company_col,
        conn_position_col=conn_position_col,
        msg_date_col=msg_date_col,
        msg_folder_col=msg_folder_col,
        args=args,
    )

    manifest = build_manifest(args)
    manifest["outputs"] = {
        "csv": [enriched_out, any_out, all_out, msg_extract_out],
        "charts": list_chart_paths(args.outdir),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Charts:")
    for p in list_chart_paths(args.outdir):
        print(p)
    return True


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--connections", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--outdir", default="./charts")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--today", default=None, help="Override today as YYYY-MM-DD for recency bucketing.")
    args = parser.parse_args()

    manifest_path = os.path.join(args.outdir, "analysis_manifest_linkedin_metrics.json")
    if try_regenerate_charts_from_cache(args, manifest_path):
        return

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    cand = pd.read_csv(args.candidates)
    conn = pd.read_csv(args.connections)
    msg = pd.read_csv(args.messages)

    # Identify key columns
    cand_link_col = find_column(cand, ["linkedin", "linkedin url", "linkedin_url", "profile url", "profile"])
    conn_link_col = find_column(conn, ["url", "linkedin", "linkedin url", "profile url", "profile"])
    conn_date_col = find_column(conn, ["connected on", "connected_on", "connected"])

    conn_company_col = find_column(conn, ["company", "current company", "organization"])
    conn_position_col = find_column(conn, ["position", "title", "role"])

    msg_sender_col = find_column(msg, ["sender profile url", "sender_profile_url"])
    msg_recips_col = find_column(msg, ["recipient profile urls", "recipient_profile_urls"])
    msg_date_col = find_column(msg, ["date", "sent date", "timestamp", "time"])
    msg_folder_col = find_column(msg, ["folder", "inbox folder", "label"])

    if cand_link_col is None:
        raise ValueError("Candidates CSV: LinkedIn column not found (expected something like 'Linkedin').")
    if conn_link_col is None:
        raise ValueError("Connections CSV: URL column not found (expected something like 'URL').")
    if msg_sender_col is None or msg_recips_col is None:
        raise ValueError("Messages CSV: could not find sender/recipient LinkedIn URL columns.")

    # Normalize candidate URLs
    cand["linkedin_clean"] = normalize_linkedin_url(cand[cand_link_col])
    cand = cand[cand["linkedin_clean"].notna() & (cand["linkedin_clean"] != "")].copy()

    # Normalize connection URLs
    conn["linkedin_clean"] = normalize_linkedin_url(conn[conn_link_col])
    conn = conn[conn["linkedin_clean"].notna() & (conn["linkedin_clean"] != "")].copy()
    conn_url_set = set(conn["linkedin_clean"].tolist())

    # Normalize message URLs (sender + recipients)
    msg_sender_urls = normalize_linkedin_url(msg[msg_sender_col])
    msg_recips_urls = normalize_linkedin_url(explode_profile_urls(msg[msg_recips_col]))

    msg_url_set = set(pd.concat([msg_sender_urls, msg_recips_urls], ignore_index=True).dropna().tolist())
    msg_url_set.discard("")
    msg_url_set.discard("nan")

    # Enrich candidates with flags (Connected / Contacted)
    enriched = cand.copy()
    enriched["is_connected_match"] = enriched["linkedin_clean"].isin(conn_url_set)
    enriched["is_contacted_match"] = enriched["linkedin_clean"].isin(msg_url_set)
    enriched["is_connected_and_contacted_match"] = enriched["is_connected_match"] & enriched["is_contacted_match"]

    # Segment label
    def segment_row(r):
        c = bool(r["is_connected_match"])
        m = bool(r["is_contacted_match"])
        if c and m:
            return "Connected + Contacted (Matched)"
        if c and not m:
            return "Connected Only (Matched)"
        if (not c) and m:
            return "Contacted Only (Matched)"
        return "Not Matched (Neither)"

    enriched["match_segment"] = enriched.apply(segment_row, axis=1)

    # Merge connection metadata (company/position/date) for connected matches
    conn_cols_to_keep = ["linkedin_clean"]
    for col in [conn_company_col, conn_position_col, conn_date_col]:
        if col is not None and col in conn.columns:
            conn_cols_to_keep.append(col)
    conn_small = conn[conn_cols_to_keep].drop_duplicates("linkedin_clean")

    enriched = enriched.merge(conn_small, on="linkedin_clean", how="left")

    # Export CSVs
    enriched_out = os.path.join(args.outdir, "candidates_enriched_matches.csv")
    enriched.to_csv(enriched_out, index=False)

    # Candidates matching ANY external CSV (connections OR messages)
    any_match = enriched[enriched["is_connected_match"] | enriched["is_contacted_match"]].copy()
    any_out = os.path.join(args.outdir, "candidates_matching_any.csv")
    any_match.to_csv(any_out, index=False)

    # Candidates matching EVERY CSV: Candidates AND Connections AND Messages
    all_match = enriched[enriched["is_connected_match"] & enriched["is_contacted_match"]].copy()
    all_out = os.path.join(args.outdir, "candidates_matching_all.csv")
    all_match.to_csv(all_out, index=False)

    # Build a messages extract involving ANY candidate URL
    cand_set = set(enriched["linkedin_clean"].tolist())
    msg2 = msg.copy()
    msg2["_sender_clean"] = normalize_linkedin_url(msg2[msg_sender_col])

    tmp = msg2[[msg_recips_col]].copy()
    tmp["_msg_row"] = tmp.index
    tmp["_recipient"] = tmp[msg_recips_col].fillna("").astype(str).str.split(",")
    tmp = tmp.explode("_recipient")
    tmp["_recipient_clean"] = normalize_linkedin_url(tmp["_recipient"].astype(str).str.strip())

    sender_in = msg2["_sender_clean"].isin(cand_set)
    recip_in = tmp.groupby("_msg_row")["_recipient_clean"].apply(lambda s: s.isin(cand_set).any())
    recip_in = recip_in.reindex(msg2.index).fillna(False)

    msg2["involves_candidate_match"] = sender_in | recip_in

    msg_extract = msg2[msg2["involves_candidate_match"]].copy()
    msg_extract_out = os.path.join(args.outdir, "messages_involving_candidates.csv")
    msg_extract.to_csv(msg_extract_out, index=False)

    # -----------------------------
    # Charts (clearer naming)
    # -----------------------------
    render_charts(
        enriched=enriched,
        msg_extract=msg_extract,
        conn_date_col=conn_date_col,
        conn_company_col=conn_company_col,
        conn_position_col=conn_position_col,
        msg_date_col=msg_date_col,
        msg_folder_col=msg_folder_col,
        args=args,
    )

    manifest = build_manifest(args)
    manifest["outputs"] = {
        "csv": [enriched_out, any_out, all_out, msg_extract_out],
        "charts": list_chart_paths(args.outdir),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print run summary
    print("=== Run Summary ===")
    print(f"Candidates (valid LinkedIn): {len(enriched):,}")
    print(f"Connected matches: {int(enriched['is_connected_match'].sum()):,}")
    print(f"Contacted matches: {int(enriched['is_contacted_match'].sum()):,}")
    print(f"Connected + Contacted (matches all CSVs): {int(enriched['is_connected_and_contacted_match'].sum()):,}")
    print("")
    print("=== Outputs ===")
    print(f"Enriched candidates: {enriched_out}")
    print(f"Candidates matching any external CSV: {any_out}")
    print(f"Candidates matching ALL CSVs: {all_out}")
    print(f"Messages involving candidates: {msg_extract_out}")
    print(f"Charts directory: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
