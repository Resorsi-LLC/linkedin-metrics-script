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


def bar_chart(series, title, xlabel, ylabel, outpath, topn=None):
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

    # Ensure enough space for title
    plt.subplots_adjust(top=0.88, left=0.35)

    plt.savefig(outpath, dpi=200)
    plt.close(fig)



def line_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: str):
    if series is None or len(series) == 0:
        return
    plt.figure()
    series.sort_index().plot(kind="line")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


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


def load_manifest(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


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


def maybe_exit_if_analyzed(args: argparse.Namespace, manifest_path: str) -> None:
    input_paths = [args.candidates, args.connections, args.messages]
    expected_outputs = [
        os.path.join(args.outdir, "candidates_enriched_matches.csv"),
        os.path.join(args.outdir, "candidates_matching_any.csv"),
        os.path.join(args.outdir, "candidates_matching_all.csv"),
        os.path.join(args.outdir, "messages_involving_candidates.csv"),
    ]
    if not os.path.exists(manifest_path):
        charts = list_chart_paths(args.outdir)
        if charts and outputs_fresh(expected_outputs, input_paths):
            print("Analysis already exists for these inputs; skipping recompute.")
            print("Charts:")
            for p in charts:
                print(p)
            raise SystemExit(0)
        return
    existing = load_manifest(manifest_path)
    if not existing:
        return
    current = build_manifest(args)
    if existing.get("inputs") != current.get("inputs"):
        return
    charts = list_chart_paths(args.outdir)
    if not charts:
        return
    print("Analysis already exists for these inputs; skipping recompute.")
    print("Charts:")
    for p in charts:
        print(p)
    raise SystemExit(0)


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
    maybe_exit_if_analyzed(args, manifest_path)

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
    # Chart 1: Match segmentation
    seg_counts = enriched["match_segment"].value_counts()
    bar_chart(
        seg_counts,
        title="Candidate Match Segmentation (Connected vs Contacted)",
        xlabel="Candidates",
        ylabel="Match segment",
        outpath=os.path.join(args.outdir, "candidate_match_segmentation_connected_vs_contacted.png"),
        topn=None,
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
            line_chart(
                monthly_conn,
                title="Connected Candidates (Matched) Over Time — Monthly",
                xlabel="Month",
                ylabel="Connected candidates",
                outpath=os.path.join(args.outdir, "connected_candidates_matched_over_time_monthly.png"),
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
            line_chart(
                monthly_msgs,
                title="Contacted Candidates (Matched) Over Time — Monthly",
                xlabel="Month",
                ylabel="Messages involving matched candidates",
                outpath=os.path.join(args.outdir, "contacted_candidates_matched_over_time_monthly.png"),
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
