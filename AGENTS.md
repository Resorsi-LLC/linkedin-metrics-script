# Repository Guidelines

## Project Structure & Module Organization
This repo is a small, script-driven analytics toolkit. The root contains two Python entry points: `linkedin_metrics.py` (candidate/connection/message matching plus charts) and `linkedin_candidate_reply_rate.py` (reply-rate and candidate classification). CSV inputs are typically kept at the repo root (examples: `Candidates_*.csv`, `Connections.csv`, `messages.csv`). Generated artifacts are written to `charts/` and `out/` (treated as outputs, not source).

## Build, Test, and Development Commands
Install dependencies locally:
- `pip install pandas matplotlib` (add `openai` if using AI classification).

Run the matching + charts workflow:
- `python linkedin_metrics.py --candidates ./Candidates_2026_01_13.csv --connections ./Connections.csv --messages ./messages.csv --outdir ./charts --topn 20`

Run reply-rate + classification:
- `python linkedin_candidate_reply_rate.py --candidates ./Candidates_2026_01_13.csv --connections ./Connections.csv --messages ./messages.csv --outdir ./out`
- AI mode: `OPENAI_API_KEY=... python linkedin_candidate_reply_rate.py ... --use_ai --openai_model gpt-4.1-mini`

## Coding Style & Naming Conventions
Use 4-space indentation and Pythonic snake_case for functions/variables. When adding new columns, follow existing conventions (e.g., `_connected_dt`, `linkedin_clean`) and prefer vectorized pandas operations. Keep helper functions small and readable; add brief comments only for non-obvious logic.

## Testing Guidelines
No automated tests are present. Validate changes by running scripts on a small CSV subset and verifying:
- Outputs exist in `charts/` and/or `out/`.
- Console/log summaries look reasonable (e.g., `=== Run Summary ===` or `out/summary.json`).

## Commit & Pull Request Guidelines
There is no established commit convention yet; use concise, imperative subjects (e.g., "Improve URL normalization"). PRs should describe the goal, list commands run, and note any new outputs. Avoid committing raw LinkedIn CSV exports or other sensitive data.

## Security & Configuration Tips
Treat LinkedIn exports as sensitive and keep them local. When using AI mode, supply the API key via `OPENAI_API_KEY` and do not hardcode secrets.
