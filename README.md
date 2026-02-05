# LinkedIn Metrics Scripts

A small, script-driven analytics toolkit for LinkedIn exports. It processes candidate, connection, and message CSVs to produce match metrics, reply-rate analysis, and charts.

## Requirements
- Python 3.10+
- Packages: `pandas`, `matplotlib` (add `openai` only if using AI classification)

Install:
```bash
pip install pandas matplotlib
```

## Scripts
### 1) `linkedin_metrics.py`
Matches candidate LinkedIn URLs against Connections and Messages, then exports enriched CSVs and charts.

Run:
```bash
python linkedin_metrics.py \
  --candidates ./Candidates_2026_01_13.csv \
  --connections ./Connections.csv \
  --messages ./messages.csv \
  --outdir ./charts \
  --topn 20
```
Optional (accepted invitations chart):
```bash
python linkedin_metrics.py \
  --candidates ./Candidates_2026_01_13.csv \
  --connections ./Connections.csv \
  --messages ./messages.csv \
  --invitations ./Invitations.csv \
  --outdir ./charts \
  --topn 20
```

Inputs (expected columns, case-insensitive):
- Candidates: `Linkedin` (or `linkedin`, `profile url`)
- Connections: `URL`, `Connected On`, plus optional `Company`, `Position`
- Messages: `SENDER PROFILE URL`, `RECIPIENT PROFILE URLS`, optional `DATE`, `FOLDER`

Outputs (in `--outdir`):
- `candidates_enriched_matches.csv`
- `candidates_matching_any.csv`
- `candidates_matching_all.csv`
- `messages_involving_candidates.csv`
- Charts (PNG), including:
  - Segmentation (connected/contacted)
  - Connected over time (per-year images)
  - Contacted over time (per-year images)
  - Top companies/positions, seniority buckets, recency, folders
  - Accepted invitations (sent vs received), if `--invitations` is provided

Behavior notes:
- URLs are normalized (lowercased, query/fragment stripped, trailing slash removed).
- “Matched” means candidate LinkedIn URL appears in the referenced CSV.
- If outputs are newer than inputs, charts are regenerated from cached CSVs without recomputing matches.

### 2) `linkedin_candidate_reply_rate.py`
Computes candidate reply-rate after connection acceptance. Uses deterministic rules and (optionally) OpenAI for ambiguous cases.

Run (deterministic only):
```bash
python linkedin_candidate_reply_rate.py \
  --candidates ./Candidates_2026_01_13.csv \
  --connections ./Connections.csv \
  --messages ./messages.csv \
  --outdir ./out
```

Run with AI:
```bash
OPENAI_API_KEY=... python linkedin_candidate_reply_rate.py \
  --candidates ./Candidates_2026_01_13.csv \
  --connections ./Connections.csv \
  --messages ./messages.csv \
  --outdir ./out \
  --use_ai \
  --openai_model gpt-4.1-mini
```

Inputs (expected columns, case-insensitive):
- Connections: `URL`, `Connected On`, optional `Company`, `Position`
- Messages: `SENDER PROFILE URL`, `RECIPIENT PROFILE URLS`, optional `DATE`, `MESSAGE`
- Candidates: `Linkedin` (or `linkedin`, `profile url`)

Outputs (in `--outdir`):
- `connections_enriched_candidate_labels.csv`
- `candidate_connections_only.csv`
- `summary.json`
- Charts (PNG): classification breakdown and reply outcomes

Behavior notes:
- Reply-rate is “strict”: counts only inbound replies after connect time.
- If outputs are newer than inputs, charts are regenerated from cached CSVs without recomputing or re-running AI.

## Caching Behavior
Both scripts skip heavy recomputation when existing outputs are newer than the input CSVs. In that case they regenerate charts from cached CSVs and exit. This keeps charts up to date without reprocessing or re-running AI.

## Security
LinkedIn exports can contain sensitive data. Keep CSVs local and avoid committing them.
