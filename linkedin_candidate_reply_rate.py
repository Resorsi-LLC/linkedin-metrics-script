import argparse
import json
import logging
import os
import re
import textwrap
import time
from collections import Counter
from contextlib import contextmanager
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Logging helpers
# -----------------------------
def setup_logger(outdir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("linkedin_metrics")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(outdir, "run.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


@contextmanager
def step(logger: logging.Logger, name: str):
    t0 = time.time()
    logger.info(f"[START] {name}")
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info(f"[DONE ] {name} | {dt:,.2f}s")


# -----------------------------
# Normalization / parsing
# -----------------------------
def normalize_linkedin_url(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    x = x.replace({"nan": "", "none": "", "null": ""})
    x = x.str.replace(r"[\?#].*$", "", regex=True)
    x = x.str.replace(r"^http://", "https://", regex=True)
    x = x.str.replace(r"/+$", "", regex=True)
    return x


def is_people_profile(url: str) -> bool:
    return isinstance(url, str) and ("linkedin.com/in/" in url)


def parse_dt_flex(series: pd.Series) -> pd.Series:
    """
    Consistent parsing:
    - try LinkedIn %d-%b-%y first
    - fallback to dateutil parse
    Always returns tz-aware UTC (or NaT).
    """
    s = series.astype(str)

    dt = pd.to_datetime(s, errors="coerce", format="%d-%b-%y")
    dt2 = pd.to_datetime(s, errors="coerce")
    dt = dt.fillna(dt2)

    # Normalize timezone
    if getattr(dt.dt, "tz", None) is not None:
        return dt.dt.tz_convert("UTC")
    return dt.dt.tz_localize("UTC")


def to_utc_aware(dt_series: pd.Series) -> pd.Series:
    """
    Convert any datetime-like series to tz-aware UTC.
    """
    s = pd.to_datetime(dt_series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert("UTC")
    return s.dt.tz_localize("UTC")


def find_column(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in cols:
            return cols[p.lower()]
    return None


def explode_recipients(cell: str) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]


def contains_zoho_form(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return "forms.zohopublic.com" in text.lower()


# -----------------------------
# Chart helpers (title-safe)
# -----------------------------
def bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: str, topn: Optional[int] = None):
    if series is None or len(series) == 0:
        return

    data = series.copy()
    if topn is not None:
        data = data.head(topn)

    fig, ax = plt.subplots(figsize=(14, 7))
    data.sort_values(ascending=True).plot(kind="barh", ax=ax)

    wrapped_title = "\n".join(textwrap.wrap(title, width=55))
    ax.set_title(wrapped_title, fontsize=14, pad=20)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.subplots_adjust(top=0.88, left=0.35, right=0.98, bottom=0.10)

    plt.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------
# Candidate / non-candidate rules
# -----------------------------
NON_CANDIDATE_TITLE_KEYWORDS = [
    "founder", "co-founder", "owner", "partner",
    "ceo", "cfo", "coo", "cto", "cxo",
    "vp", "vice president", "head of", "director",
    "recruiter", "talent acquisition", "talent partner", "hr", "people operations", "people ops",
    "hiring manager"
]


def looks_non_candidate_by_title(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in NON_CANDIDATE_TITLE_KEYWORDS)


# -----------------------------
# Messages -> per-person threads
# -----------------------------
def infer_recruiter_profile_url(messages_df: pd.DataFrame, sender_col: str) -> Optional[str]:
    senders = messages_df[sender_col].dropna().astype(str).tolist()
    senders = [s.strip().lower() for s in senders if s and "linkedin.com/in/" in s.lower()]
    if not senders:
        return None
    return Counter(senders).most_common(1)[0][0]


def build_threads(
    messages_df: pd.DataFrame,
    sender_col: str,
    recipients_col: str,
    date_col: Optional[str],
    body_col: Optional[str],
    recruiter_url: str,
    logger: logging.Logger,
    progress_every: int = 50_000,
) -> Dict[str, List[dict]]:
    df = messages_df.copy()

    with step(logger, "Normalize message sender/recipients"):
        df["_sender"] = normalize_linkedin_url(df[sender_col])
        df["_recips_list"] = df[recipients_col].apply(explode_recipients)
        df["_recips_list"] = df["_recips_list"].apply(lambda lst: [u.lower().strip() for u in lst])
        df["_recips_list"] = df["_recips_list"].apply(lambda lst: [re.sub(r"[\?#].*$", "", u) for u in lst])
        df["_recips_list"] = df["_recips_list"].apply(lambda lst: [re.sub(r"/+$", "", u) for u in lst])

    with step(logger, "Parse message timestamps"):
        if date_col and date_col in df.columns:
            # Messages timestamps may be tz-aware already; normalize to UTC-aware
            df["_dt"] = to_utc_aware(pd.to_datetime(df[date_col], errors="coerce"))
        else:
            df["_dt"] = pd.NaT

    with step(logger, "Load message bodies (if available)"):
        if body_col and body_col in df.columns:
            df["_text"] = df[body_col].fillna("").astype(str)
        else:
            df["_text"] = ""

    recruiter_url_norm = normalize_linkedin_url(pd.Series([recruiter_url])).iloc[0]
    threads: Dict[str, List[dict]] = {}

    with step(logger, "Build per-person message threads (this can be the slowest step)"):
        n = len(df)

        # Use itertuples(name=None) so columns are positional, not attribute-based
        for i, (sender, recips, dt, text) in enumerate(
            df[["_sender", "_recips_list", "_dt", "_text"]].itertuples(index=False, name=None),
            start=1
        ):
            involved = set()
            if is_people_profile(sender):
                involved.add(sender)
            for r in recips:
                if is_people_profile(r):
                    involved.add(r)
            if recruiter_url_norm not in involved:
                if i % progress_every == 0:
                    logger.info(f"Threads progress: {i:,}/{n:,} rows scanned")
                continue
            others = [u for u in involved if u != recruiter_url_norm]
            for person_url in others:
                if person_url not in threads:
                    threads[person_url] = []
                if sender == recruiter_url_norm:
                    direction = "RECRUITER->THEM"
                elif sender == person_url:
                    direction = "THEM->RECRUITER"
                else:
                    direction = "OTHER"
                threads[person_url].append({"dt": dt, "direction": direction, "text": text})
            if i % progress_every == 0:
                logger.info(f"Threads progress: {i:,}/{n:,} rows scanned | threads: {len(threads):,}")


    with step(logger, "Sort threads by timestamp"):
        for person_url, msgs in threads.items():
            msgs.sort(key=lambda m: (pd.isna(m["dt"]), m["dt"] if not pd.isna(m["dt"]) else pd.Timestamp.min))

    return threads


def thread_stats(msgs: List[dict]) -> dict:
    inbound = sum(1 for m in msgs if m["direction"] == "THEM->RECRUITER")
    outbound = sum(1 for m in msgs if m["direction"] == "RECRUITER->THEM")
    zoho = any(contains_zoho_form(m.get("text", "")) for m in msgs)

    first_inbound_dt = next(
        (m["dt"] for m in msgs if m["direction"] == "THEM->RECRUITER" and not pd.isna(m["dt"])),
        pd.NaT
    )
    last_dt = next((m["dt"] for m in reversed(msgs) if not pd.isna(m["dt"])), pd.NaT)

    return {
        "inbound_count": inbound,
        "outbound_count": outbound,
        "contains_zoho_form": zoho,
        "first_inbound_dt": first_inbound_dt,
        "last_message_dt": last_dt,
    }


def select_reasonable_messages_for_prompt(msgs: List[dict], max_messages: int = 40, max_chars: int = 3500) -> List[dict]:
    if not msgs:
        return []

    zoho_msgs = [m for m in msgs if contains_zoho_form(m.get("text", ""))]
    first_out = next((m for m in msgs if m["direction"] == "RECRUITER->THEM"), None)
    first_in = next((m for m in msgs if m["direction"] == "THEM->RECRUITER"), None)
    last_in = next((m for m in reversed(msgs) if m["direction"] == "THEM->RECRUITER"), None)
    last_out = next((m for m in reversed(msgs) if m["direction"] == "RECRUITER->THEM"), None)

    chosen: List[dict] = []

    def add_unique(m):
        if m is None:
            return
        if m in chosen:
            return
        chosen.append(m)

    for m in zoho_msgs:
        add_unique(m)
    add_unique(first_out)
    add_unique(first_in)
    add_unique(last_in)
    add_unique(last_out)

    for m in reversed(msgs):
        if len(chosen) >= max_messages:
            break
        add_unique(m)

    chosen.sort(key=lambda m: (pd.isna(m["dt"]), m["dt"] if not pd.isna(m["dt"]) else pd.Timestamp.min))

    def format_msg(m: dict) -> str:
        prefix = "RECRUITER -> THEM" if m["direction"] == "RECRUITER->THEM" else \
                 "THEM -> RECRUITER" if m["direction"] == "THEM->RECRUITER" else "OTHER"
        txt = (m.get("text") or "").strip()
        txt = re.sub(r"\s+", " ", txt)
        return f"{prefix}: {txt}"

    total = sum(len(format_msg(m)) + 1 for m in chosen)
    if total <= max_chars:
        return chosen

    keep: List[dict] = []
    for m in chosen[:10]:
        keep.append(m)
    for m in chosen[-10:]:
        if m not in keep:
            keep.append(m)
    for m in zoho_msgs:
        if m not in keep:
            keep.append(m)

    keep.sort(key=lambda m: (pd.isna(m["dt"]), m["dt"] if not pd.isna(m["dt"]) else pd.Timestamp.min))

    while True:
        total2 = sum(len(format_msg(m)) + 1 for m in keep)
        if total2 <= max_chars or len(keep) <= 5:
            break
        drop_idx = None
        for i in range(5, len(keep) - 5):
            if not contains_zoho_form(keep[i].get("text", "")):
                drop_idx = i
                break
        if drop_idx is None:
            break
        keep.pop(drop_idx)

    return keep


# -----------------------------
# AI classification (optional)
# -----------------------------
def classify_with_ai_openai(prompt: str, model: str, api_key: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You classify LinkedIn connections for a recruiter.\n"
                "Return JSON only with keys: label, confidence, reasons.\n"
                "label must be one of: CANDIDATE, NON_CANDIDATE, UNCERTAIN.\n"
                "confidence must be one of: HIGH, MEDIUM, LOW.\n"
                "reasons must be a short list of 1-4 short strings.\n"
                "CANDIDATE: individual contributor or job-seeker type interaction.\n"
                "NON_CANDIDATE: client, executive, recruiter/HR, vendor, partnership contact.\n"
                "UNCERTAIN: insufficient evidence.\n"
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return {"label": "UNCERTAIN", "confidence": "LOW", "reasons": ["Failed to parse model output as JSON."]}


def build_ai_prompt(person_url: str, title: str, company: str, connected_on, stats: dict, selected_msgs: List[dict]) -> str:
    def fmt_dt(dt):
        if dt is None or pd.isna(dt):
            return "Unknown"
        if isinstance(dt, pd.Timestamp):
            return dt.strftime("%Y-%m-%d")
        return str(dt)

    lines = [
        f"LinkedIn URL: {person_url}",
        f"Title: {title or 'Unknown'}",
        f"Company: {company or 'Unknown'}",
        f"Connected On: {fmt_dt(connected_on)}",
        "",
        "Conversation Stats:",
        f"- inbound_count (THEM -> RECRUITER): {stats['inbound_count']}",
        f"- outbound_count (RECRUITER -> THEM): {stats['outbound_count']}",
        f"- contains_zoho_form_link: {stats['contains_zoho_form']}",
        f"- first_inbound_date: {fmt_dt(stats['first_inbound_dt'])}",
        f"- last_message_date: {fmt_dt(stats['last_message_dt'])}",
        "",
        "Selected Conversation (chronological):"
    ]

    for m in selected_msgs:
        prefix = "RECRUITER -> THEM" if m["direction"] == "RECRUITER->THEM" else \
                 "THEM -> RECRUITER" if m["direction"] == "THEM->RECRUITER" else "OTHER"
        txt = (m.get("text") or "").strip()
        txt = re.sub(r"\s+", " ", txt)
        if len(txt) > 500:
            txt = txt[:500] + "…"
        lines.append(f"- {prefix}: {txt}")

    lines += [
        "",
        "Task:",
        "Classify this person as CANDIDATE, NON_CANDIDATE, or UNCERTAIN.",
        "Important cues:",
        "- If the conversation includes forms.zohopublic.com links, that strongly suggests a candidate workflow.",
        "- If title/company indicate HR/recruiter/executive/partnership contact, that suggests NON_CANDIDATE unless strong candidate evidence exists.",
        "- If not enough evidence, choose UNCERTAIN."
    ]

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--connections", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--outdir", default="./out")
    parser.add_argument("--use_ai", action="store_true")
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--openai_model", default="gpt-4.1-mini")
    parser.add_argument("--max_prompt_messages", type=int, default=40)
    parser.add_argument("--max_prompt_chars", type=int, default=3500)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--limit_messages", type=int, default=0, help="For debugging: limit messages rows read (0 = no limit).")
    parser.add_argument("--limit_connections", type=int, default=0, help="For debugging: limit connections rows read (0 = no limit).")
    args = parser.parse_args()

    logger = setup_logger(args.outdir, args.log_level)

    with step(logger, "Read CSVs"):
        cand = pd.read_csv(args.candidates)
        conn = pd.read_csv(args.connections)
        msg = pd.read_csv(args.messages)

        if args.limit_messages > 0:
            msg = msg.head(args.limit_messages)
            logger.warning(f"limit_messages enabled: using first {args.limit_messages:,} message rows")

        if args.limit_connections > 0:
            conn = conn.head(args.limit_connections)
            logger.warning(f"limit_connections enabled: using first {args.limit_connections:,} connection rows")

    # Identify columns
    cand_link_col = find_column(cand, ["linkedin", "linkedin url", "linkedin_url", "profile url", "profile"])
    conn_url_col = find_column(conn, ["url", "linkedin", "linkedin url", "profile url", "profile"])
    conn_date_col = find_column(conn, ["connected on", "connected_on", "connected"])
    conn_company_col = find_column(conn, ["company", "current company", "organization"])
    conn_position_col = find_column(conn, ["position", "title", "role"])

    msg_sender_col = find_column(msg, ["sender profile url", "sender_profile_url"])
    msg_recip_col = find_column(msg, ["recipient profile urls", "recipient_profile_urls"])
    msg_date_col = find_column(msg, ["date", "sent date", "timestamp", "time"])
    msg_body_col = find_column(msg, ["message", "body", "content", "text"])

    logger.info(f"Detected columns | Candidates: {cand_link_col} | Connections: {conn_url_col}, {conn_date_col}, {conn_company_col}, {conn_position_col} | Messages: {msg_sender_col}, {msg_recip_col}, {msg_date_col}, {msg_body_col}")

    if conn_url_col is None or conn_date_col is None:
        raise ValueError("Connections.csv must include URL and Connected On columns.")
    if msg_sender_col is None or msg_recip_col is None:
        raise ValueError("messages.csv must include sender and recipient profile URL columns.")
    if cand_link_col is None:
        raise ValueError("Candidates.csv must include a LinkedIn column (e.g., 'Linkedin').")

    with step(logger, "Build CRM candidate URL set"):
        cand["_linkedin_clean"] = normalize_linkedin_url(cand[cand_link_col])
        crm_candidate_set = set(cand["_linkedin_clean"].dropna().tolist())
        logger.info(f"CRM candidate URLs: {len(crm_candidate_set):,}")

    with step(logger, "Normalize connections and parse Connected On"):
        conn["_linkedin_clean"] = normalize_linkedin_url(conn[conn_url_col])
        conn = conn[conn["_linkedin_clean"].apply(is_people_profile)].copy()
        conn["_connected_dt"] = parse_dt_flex(conn[conn_date_col])
        logger.info(f"Accepted connections (people profiles): {len(conn):,}")

    with step(logger, "Infer recruiter LinkedIn URL"):
        recruiter_url = infer_recruiter_profile_url(msg, msg_sender_col)
        if not recruiter_url:
            raise ValueError("Could not infer recruiter profile URL from messages sender column.")
        logger.info(f"Inferred recruiter URL: {recruiter_url}")

    threads = build_threads(
        messages_df=msg,
        sender_col=msg_sender_col,
        recipients_col=msg_recip_col,
        date_col=msg_date_col,
        body_col=msg_body_col,
        recruiter_url=recruiter_url,
        logger=logger,
        progress_every=50_000
    )
    logger.info(f"Threads built: {len(threads):,}")

    with step(logger, "Build base connection-level table"):
        base_cols = ["_linkedin_clean", "_connected_dt"]
        if conn_company_col and conn_company_col in conn.columns:
            base_cols.append(conn_company_col)
        if conn_position_col and conn_position_col in conn.columns:
            base_cols.append(conn_position_col)

        base = conn[base_cols].copy().rename(columns={
            "_linkedin_clean": "linkedin_url",
            "_connected_dt": "connected_on",
            conn_company_col: "company",
            conn_position_col: "title",
        })

        base["company"] = base.get("company", pd.Series([""] * len(base))).fillna("")
        base["title"] = base.get("title", pd.Series([""] * len(base))).fillna("")

    with step(logger, "Compute message stats for each connection"):
        stats_rows = []
        urls = base["linkedin_url"].tolist()
        n = len(urls)
        for i, person_url in enumerate(urls, start=1):
            msgs_for_person = threads.get(person_url, [])
            s = thread_stats(msgs_for_person) if msgs_for_person else {
                "inbound_count": 0,
                "outbound_count": 0,
                "contains_zoho_form": False,
                "first_inbound_dt": pd.NaT,
                "last_message_dt": pd.NaT,
            }
            stats_rows.append(s)

            if i % 10_000 == 0:
                logger.info(f"Stats progress: {i:,}/{n:,}")

        stats_df = pd.DataFrame(stats_rows)
        base = pd.concat([base.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)

        # Normalize datetimes to tz-aware UTC for strict comparisons
        base["connected_on"] = to_utc_aware(base["connected_on"])
        base["first_inbound_dt"] = to_utc_aware(base["first_inbound_dt"])

    with step(logger, "Compute replied_after_connect (strict, vectorized)"):
        base["replied_after_connect"] = (
            (base["inbound_count"] > 0)
            & base["connected_on"].notna()
            & base["first_inbound_dt"].notna()
            & (base["first_inbound_dt"] >= base["connected_on"])
        )

    with step(logger, "Candidate classification (deterministic rules)"):
        base["candidate_label"] = "UNCERTAIN"
        base["candidate_confidence"] = "LOW"
        base["candidate_reasons"] = ""

        zoho_mask = base["contains_zoho_form"] == True
        base.loc[zoho_mask, "candidate_label"] = "CANDIDATE"
        base.loc[zoho_mask, "candidate_confidence"] = "HIGH"
        base.loc[zoho_mask, "candidate_reasons"] = "Zoho form link in message history"

        crm_mask = base["linkedin_url"].isin(crm_candidate_set)
        base.loc[crm_mask, "candidate_label"] = "CANDIDATE"
        base.loc[crm_mask, "candidate_confidence"] = "HIGH"
        base.loc[crm_mask, "candidate_reasons"] = base.loc[crm_mask, "candidate_reasons"].where(
            base.loc[crm_mask, "candidate_reasons"] != "",
            "Present in CRM Candidates export"
        )

        noncand_mask = base["title"].apply(looks_non_candidate_by_title)
        override_mask = noncand_mask & (base["candidate_label"] != "CANDIDATE")
        base.loc[override_mask, "candidate_label"] = "NON_CANDIDATE"
        base.loc[override_mask, "candidate_confidence"] = "HIGH"
        base.loc[override_mask, "candidate_reasons"] = "Title indicates executive/HR/recruiter/partnership contact"

        logger.info(f"Deterministic labels | CANDIDATE: {(base['candidate_label']=='CANDIDATE').sum():,} | NON_CANDIDATE: {(base['candidate_label']=='NON_CANDIDATE').sum():,} | UNCERTAIN: {(base['candidate_label']=='UNCERTAIN').sum():,}")

    if args.use_ai:
        if not args.openai_api_key:
            raise ValueError("AI enabled but no OpenAI API key provided.")

        ai_mask = (base["candidate_label"] == "UNCERTAIN")
        ai_rows = base[ai_mask]
        logger.info(f"AI classification needed for {len(ai_rows):,} connections")

        with step(logger, "AI classification (ambiguous only)"):
            ai_results = {}
            total = len(ai_rows)
            for j, (idx, row) in enumerate(ai_rows.iterrows(), start=1):
                person_url = row["linkedin_url"]
                msgs_for_person = threads.get(person_url, [])

                selected = select_reasonable_messages_for_prompt(
                    msgs_for_person,
                    max_messages=args.max_prompt_messages,
                    max_chars=args.max_prompt_chars
                )

                prompt = build_ai_prompt(
                    person_url=person_url,
                    title=row.get("title", ""),
                    company=row.get("company", ""),
                    connected_on=row.get("connected_on", pd.NaT),
                    stats={
                        "inbound_count": row.get("inbound_count", 0),
                        "outbound_count": row.get("outbound_count", 0),
                        "contains_zoho_form": row.get("contains_zoho_form", False),
                        "first_inbound_dt": row.get("first_inbound_dt", pd.NaT),
                        "last_message_dt": row.get("last_message_dt", pd.NaT),
                    },
                    selected_msgs=selected
                )

                out = classify_with_ai_openai(prompt, args.openai_model, args.openai_api_key)
                ai_results[idx] = out

                if j % 50 == 0 or j == total:
                    logger.info(f"AI progress: {j:,}/{total:,}")

            for idx, out in ai_results.items():
                label = out.get("label", "UNCERTAIN")
                conf = out.get("confidence", "LOW")
                reasons = out.get("reasons", [])
                reasons_str = "; ".join([str(r) for r in reasons]) if isinstance(reasons, list) else str(reasons)

                base.loc[idx, "candidate_label"] = label
                base.loc[idx, "candidate_confidence"] = conf
                base.loc[idx, "candidate_reasons"] = reasons_str

    with step(logger, "Compute final metric + outputs"):
        candidates_only = base[base["candidate_label"] == "CANDIDATE"].copy()
        denom = len(candidates_only)
        num = int(candidates_only["replied_after_connect"].sum())
        reply_rate = (num / denom) if denom > 0 else 0.0

        strict_excluded = candidates_only[
            (candidates_only["inbound_count"] > 0) &
            (candidates_only["connected_on"].isna() | candidates_only["first_inbound_dt"].isna())
        ]
        strict_excluded_count = int(len(strict_excluded))

        out_all = os.path.join(args.outdir, "connections_enriched_candidate_labels.csv")
        out_candidates = os.path.join(args.outdir, "candidate_connections_only.csv")
        base.to_csv(out_all, index=False)
        candidates_only.to_csv(out_candidates, index=False)

        summary = {
            "recruiter_url_inferred": recruiter_url,
            "accepted_connections_people": int(len(base)),
            "candidate_connections": int(denom),
            "candidate_replied_strict_after_connect": int(num),
            "candidate_reply_rate_strict": reply_rate,
            "strict_excluded_candidates_with_inbound_but_missing_dates": strict_excluded_count,
            "ai_used": bool(args.use_ai),
            "model": args.openai_model if args.use_ai else None
        }
        with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Charts
        seg_counts = base["candidate_label"].value_counts()
        bar_chart(seg_counts,
                  "Accepted Connections — Classification (Candidate vs Non-Candidate vs Uncertain)",
                  "Connections", "Class",
                  os.path.join(args.outdir, "classification_breakdown.png"))

        rr_counts = pd.Series({"Candidate replied (strict)": num, "Candidate did not reply (strict)": denom - num})
        bar_chart(rr_counts,
                  "Candidate Connections — Replies After Acceptance (Strict Date Enforcement)",
                  "Candidates", "Outcome",
                  os.path.join(args.outdir, "candidate_reply_outcomes_strict.png"))

        logger.info("=== FINAL METRIC ===")
        logger.info(f"Accepted connections (people): {len(base):,}")
        logger.info(f"Candidate connections: {denom:,}")
        logger.info(f"Candidate replied (strict): {num:,}")
        logger.info(f"Reply rate (strict): {reply_rate:.2%}")
        logger.info(f"Strict excluded (candidate inbound but missing dates): {strict_excluded_count:,}")
        logger.info(f"Outputs: {out_all} | {out_candidates} | summary.json | charts")


if __name__ == "__main__":
    main()

