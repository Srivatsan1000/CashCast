# CashCast — AI Finance Controller

Razorpay AI Buildathon · Track 04

A single-file Python app that closes two finance-ops loops on synthetic data, with a tool-calling chat agent (IBM Granite) on top.

## What it does

1. **Synthetic data** — 220 transactions, 60 receivables, 45 payables, plus a two-sided internal-ledger/bank-statement pair with realistic mismatches injected.
2. **Reconciliation** — matches internal ledger vs bank statement, reports match rate, auto-match precision/recall, throughput, and a full itemized exception list grouped into 4 root-cause categories.
3. **Monte Carlo cash forecast** — 3,000-path simulation of 30-day forward cash position (P10/P50/P90 fan chart), calibrated against a rolling-origin, no-look-ahead backtest.
4. **Chat agent** — a Gradio chat UI where IBM Granite (`granite-4.1-8b`) answers questions by calling real Python tools (`get_reconciliation_overview`, `list_exceptions`, `get_forecast_overview`, `list_payables`, `propose_action`, etc.) instead of receiving one static data dump. Every action it can "recommend" only logs to a pending-approval queue — it never executes anything.

## Run it

```bash
pip install numpy pandas scipy matplotlib transformers accelerate bitsandbytes sentencepiece gradio
python cashcast.py
```

Runs `launch_ui()` by default, which builds the world state (data → reconciliation → forecast → calibration) once, loads Granite, and opens a Gradio chat UI (`share=True` → public URL). Needs a GPU for Granite in reasonable time; the data/reconciliation/forecast pipeline itself runs fine on CPU.

To dry-run the UI wiring without a GPU/model download, call `launch_ui(use_mock_agent=True)` instead.

## Key design choices

- **Grounded, not memorized**: the agent only ever answers using numbers returned by its own tool calls this turn. A verifier (`verify_against_trace`) checks every cited ID and currency reference against the actual tool output and flags anything that doesn't match.
- **No arithmetic left for the model**: exception categories are pre-grouped and pre-summed in Python (`categorize_exceptions`) so the model reads four correct numbers instead of adding up raw records itself.
- **Bounded actions**: `propose_action` is the only way the agent can suggest a concrete step — it logs to `pending_actions`, status `pending_human_approval`, and nothing more.
- **Honest calibration**: the backtest withholds future data from each forecast origin (no look-ahead) and reports the real coverage rate, including the calibration multiplier applied to correct it.

## Files this produces

- `transactions.csv`, `receivables.csv`, `payables.csv`, `internal_ledger.csv`, `bank_statement.csv` — synthetic data
- `cashcast_fan_chart.png` — Monte Carlo forecast chart (only if run via `main()`, not the chat UI)
