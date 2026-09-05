import json
import random
import re
from datetime import timedelta

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:
    StoppingCriteria = object                                                                                   
    StoppingCriteriaList = None


def _extract_first_json_object(text):
    
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


class _FirstJSONObjectStop(StoppingCriteria):
    
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0][self.prompt_len:]
        if generated_ids.shape[0] < 2:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _extract_first_json_object(text) is not None


                                                                               
                              
                                                                               
N_TRANSACTIONS = 220
N_RECEIVABLES = 60
N_PAYABLES = 45
START_DATE = pd.Timestamp("2026-05-01")
HISTORY_DAYS = 90
FORECAST_HORIZON = 30
TODAY = START_DATE + timedelta(days=HISTORY_DAYS)

VENDORS = ["Acme Logistics", "BluePeak Cloud", "Nimbus Payroll Services",
           "Orion Office Supplies", "Vertex Marketing", "Delta Tax Consultants",
           "Foxtrot Legal", "Granite Facilities", "Helix Analytics Inc"]
CUSTOMERS = [f"CUST-{i:04d}" for i in range(1, 61)]
PAYMENT_METHODS = ["UPI", "Card", "NetBanking", "Wallet"]


def generate_transactions(n=N_TRANSACTIONS):
    rows = []
    for i in range(n):
        txn_id = f"TXN{100000+i}"
        payment_date = START_DATE + timedelta(
            days=int(np.random.randint(0, HISTORY_DAYS)), hours=int(np.random.randint(0, 24)))
        amount = round(np.random.lognormal(mean=8.5, sigma=0.9), 2)
        method = random.choice(PAYMENT_METHODS)

        roll = np.random.rand()
        status = "success" if roll < 0.90 else ("failed" if roll < 0.96 else "refunded")

        settlement_lag_days, settlement_date = None, None
        if status == "success":
            settlement_lag_days = float(np.clip(np.random.lognormal(mean=0.65, sigma=0.55), 0.5, 10))
            settlement_date = payment_date + timedelta(days=settlement_lag_days)

        rows.append({
            "txn_id": txn_id, "customer_id": random.choice(CUSTOMERS),
            "payment_date": payment_date, "amount": amount, "method": method,
            "status": status,
            "settlement_lag_days": round(settlement_lag_days, 2) if settlement_lag_days else None,
            "settlement_date": settlement_date,
        })
    return pd.DataFrame(rows)


def generate_receivables(n=N_RECEIVABLES, today=TODAY):
    rows = []
    aging_buckets = {"0-15d": (0, 15, 0.92), "16-30d": (16, 30, 0.80),
                      "31-60d": (31, 60, 0.55), "60d+": (61, 120, 0.30)}
    bucket_names = list(aging_buckets.keys())
    for i in range(n):
        inv_id = f"INV{5000+i}"
        bucket = random.choices(bucket_names, weights=[0.45, 0.30, 0.15, 0.10])[0]
        lo, hi, base_prob = aging_buckets[bucket]
        age_days = np.random.randint(lo, hi + 1)
        invoice_date = today - timedelta(days=int(age_days))
        due_date = invoice_date + timedelta(days=30)
        amount = round(np.random.lognormal(mean=9.2, sigma=0.8), 2)
        collection_prob = float(np.clip(np.random.normal(base_prob, 0.07), 0.03, 0.99))
        expected_days_to_collect = float(np.clip(
            np.random.exponential(scale=10 if bucket == "0-15d" else 20), 1, 90))

        rows.append({
            "invoice_id": inv_id, "customer_id": random.choice(CUSTOMERS),
            "invoice_date": invoice_date, "due_date": due_date, "amount": amount,
            "aging_bucket": bucket, "collection_probability": round(collection_prob, 3),
            "expected_days_to_collect": round(expected_days_to_collect, 1),
        })
    return pd.DataFrame(rows)


def generate_payables(n=N_PAYABLES, today=TODAY, horizon=FORECAST_HORIZON):
    rows = []
    categories = ["payroll", "vendor", "tax", "rent", "loan_emi"]
    cat_weights = [0.20, 0.45, 0.10, 0.15, 0.10]
    for i in range(n):
        pay_id = f"PAY{7000+i}"
        category = random.choices(categories, weights=cat_weights)[0]
        due_offset = np.random.randint(0, horizon + 15)
        due_date = today + timedelta(days=int(due_offset))

        if category == "payroll":
            amount, vendor, jitter_days = round(np.random.normal(140000, 12000), 2), "Nimbus Payroll Services", 0
        elif category == "tax":
            amount, vendor, jitter_days = round(np.random.normal(55000, 18000), 2), "Delta Tax Consultants", 0
        elif category == "rent":
            amount, vendor = round(np.random.normal(35000, 4000), 2), "Granite Facilities"
            jitter_days = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
        elif category == "loan_emi":
            amount, vendor, jitter_days = round(np.random.normal(60000, 5000), 2), "Delta Tax Consultants", 0
        else:
            amount, vendor = round(np.random.lognormal(mean=8.7, sigma=0.7), 2), random.choice(VENDORS)
            jitter_days = np.random.choice([-2, -1, 0, 1, 2, 3], p=[0.05, 0.1, 0.5, 0.15, 0.1, 0.1])

        rows.append({
            "payable_id": pay_id, "vendor": vendor, "category": category,
            "due_date": due_date, "amount": abs(round(amount, 2)),
            "typical_jitter_days": int(jitter_days),
        })
    return pd.DataFrame(rows)


def generate_reconciliation_ledgers(transactions_df, exception_rate=0.14):
    settled = transactions_df[transactions_df["status"] == "success"].copy()
    internal_rows, bank_rows = [], []

    for _, txn in settled.iterrows():
        internal_entry = {
            "source": "internal_ledger", "ref_id": txn["txn_id"],
            "date": txn["settlement_date"].date() if pd.notna(txn["settlement_date"]) else None,
            "amount": txn["amount"],
            "description": f"Settlement for {txn['txn_id']} ({txn['customer_id']})",
            "truth_matchable": True,
        }
        bank_entry = dict(internal_entry)
        bank_entry["source"] = "bank_statement"

        roll = np.random.rand()
        if roll < exception_rate:
            fault = random.choice(["missing_in_bank", "missing_in_internal", "amount_mismatch",
                                    "date_shift", "duplicate_internal"])
            if fault == "missing_in_bank":
                internal_entry["truth_matchable"] = False
                internal_rows.append(internal_entry); continue
            elif fault == "missing_in_internal":
                bank_entry["truth_matchable"] = False
                bank_rows.append(bank_entry); continue
            elif fault == "amount_mismatch":
                bank_entry["amount"] = round(bank_entry["amount"] - random.choice([-1, 1]) * round(np.random.uniform(1, 250), 2), 2)
                internal_entry["truth_matchable"] = False
                bank_entry["truth_matchable"] = False
                internal_rows.append(internal_entry); bank_rows.append(bank_entry); continue
            elif fault == "date_shift":
                bank_entry["date"] = bank_entry["date"] + timedelta(days=random.choice([1, 2, -1]))
                internal_rows.append(internal_entry); bank_rows.append(bank_entry); continue
            elif fault == "duplicate_internal":
                internal_rows.append(internal_entry)
                dup = dict(internal_entry); dup["ref_id"] = internal_entry["ref_id"] + "-DUP"; dup["truth_matchable"] = False
                internal_rows.append(dup); bank_rows.append(bank_entry); continue

        internal_rows.append(internal_entry); bank_rows.append(bank_entry)

    for i in range(6):
        d = START_DATE + timedelta(days=int(np.random.randint(0, HISTORY_DAYS)))
        bank_rows.append({
            "source": "bank_statement", "ref_id": f"BANKFEE{i}", "date": d.date(),
            "amount": -round(np.random.uniform(50, 500), 2),
            "description": random.choice(["Bank charges", "NEFT fee", "Interest credit", "SMS alert charges"]),
            "truth_matchable": False,
        })

    return pd.DataFrame(internal_rows), pd.DataFrame(bank_rows)


def generate_all_data(out_dir=None):
    txns = generate_transactions()
    receivables = generate_receivables()
    payables = generate_payables()
    internal_ledger, bank_statement = generate_reconciliation_ledgers(txns)
    if out_dir:
        txns.to_csv(f"{out_dir}/transactions.csv", index=False)
        receivables.to_csv(f"{out_dir}/receivables.csv", index=False)
        payables.to_csv(f"{out_dir}/payables.csv", index=False)
        internal_ledger.to_csv(f"{out_dir}/internal_ledger.csv", index=False)
        bank_statement.to_csv(f"{out_dir}/bank_statement.csv", index=False)
    return txns, receivables, payables, internal_ledger, bank_statement


                                                                               
                          
                                                                               
AMOUNT_TOLERANCE = 1.0
MAX_DATE_SHIFT = 3


def reconcile(internal_df, bank_df, amount_tolerance=AMOUNT_TOLERANCE, max_date_shift=MAX_DATE_SHIFT):
    internal = internal_df.copy()
    bank = bank_df.copy()
    internal["date"] = pd.to_datetime(internal["date"])
    bank["date"] = pd.to_datetime(bank["date"])
    internal["matched"] = False
    bank["matched"] = False

    matches, exceptions = [], []
    bank_indexed = {}
    for idx, row in bank.iterrows():
        bank_indexed.setdefault(row["ref_id"], []).append(idx)

    for i_idx, i_row in internal.iterrows():
        ref_id_base = i_row["ref_id"].split("-DUP")[0]
        candidates = [c for c in bank_indexed.get(ref_id_base, []) if not bank.loc[c, "matched"]]

        if not candidates:
            exceptions.append({"ref_id": i_row["ref_id"], "side": "internal_ledger",
                                "amount": i_row["amount"], "date": i_row["date"].date(),
                                "reason": "missing_in_bank_statement"})
            continue

        best, best_reason = None, None
        for c in candidates:
            b_row = bank.loc[c]
            amt_diff = abs(b_row["amount"] - i_row["amount"])
            date_diff = abs((b_row["date"] - i_row["date"]).days)
            if amt_diff <= 0.01 and date_diff == 0:
                best, best_reason = c, "exact"; break
            elif amt_diff <= amount_tolerance and date_diff == 0:
                best, best_reason = c, "amount_tolerance"
            elif amt_diff <= 0.01 and date_diff <= max_date_shift and best_reason != "amount_tolerance":
                best, best_reason = c, "date_shift"

        if best is not None:
            internal.loc[i_idx, "matched"] = True
            bank.loc[best, "matched"] = True
            matches.append({"ref_id": i_row["ref_id"], "match_type": best_reason,
                             "internal_amount": i_row["amount"], "bank_amount": bank.loc[best, "amount"],
                             "internal_date": i_row["date"].date(), "bank_date": bank.loc[best, "date"].date(),
                             "truth_matchable": bool(i_row.get("truth_matchable", True))})
        else:
            b_row = bank.loc[candidates[0]]
                                                                           
                                                                         
                                                                       
                                                                             
                                                        
            bank.loc[candidates[0], "matched"] = True
            exceptions.append({"ref_id": i_row["ref_id"], "side": "amount_or_date_mismatch",
                                "amount": i_row["amount"], "date": i_row["date"].date(),
                                "reason": f"bank_amount={b_row['amount']} bank_date={b_row['date'].date()} exceeds tolerance"})

    for b_idx, b_row in bank.iterrows():
        if not b_row["matched"]:
            reason = "bank_only_fee_or_interest" if str(b_row["ref_id"]).startswith("BANKFEE") \
                else "missing_in_internal_ledger"
            exceptions.append({"ref_id": b_row["ref_id"], "side": "bank_statement",
                                "amount": b_row["amount"], "date": b_row["date"].date(), "reason": reason})

    total_internal = len(internal)
    matched_count = len(matches)
    match_rate = matched_count / total_internal if total_internal else 0.0
    exceptions_df = pd.DataFrame(exceptions)
    matches_df = pd.DataFrame(matches)

    matches_df = pd.DataFrame(matches)
    correct_auto = int(matches_df["truth_matchable"].sum()) if len(matches_df) and "truth_matchable" in matches_df else matched_count
    auto_precision = correct_auto / matched_count if matched_count else 0.0
    true_matchable = int(internal["truth_matchable"].sum()) if "truth_matchable" in internal.columns else total_internal
    auto_recall = correct_auto / true_matchable if true_matchable else 0.0

    summary = {
        "total_internal_records": total_internal, "total_bank_records": len(bank),
        "matched_count": matched_count, "match_rate": round(match_rate, 4),
        "auto_match_precision": round(auto_precision, 4),
        "auto_match_recall": round(auto_recall, 4),
        "exception_count": len(exceptions_df),
        "exception_breakdown": (exceptions_df["reason"].value_counts().to_dict() if len(exceptions_df) else {}),
        "match_type_breakdown": (matches_df["match_type"].value_counts().to_dict() if len(matches_df) else {}),
    }
    return {"summary": summary, "matches": matches_df, "exceptions": exceptions_df}


                                                                               
                                
                                                                               
def _fit_daily_inflow_bootstrap_pool(transactions_df):
    settled = transactions_df[transactions_df["status"] == "success"].copy()
    settled["settlement_date"] = pd.to_datetime(settled["settlement_date"])
    daily = settled.groupby(settled["settlement_date"].dt.date)["amount"].sum()
    return daily.values if len(daily) else np.array([0.0])


def _failure_rate(transactions_df):
    total = len(transactions_df)
    failed = (transactions_df["status"] != "success").sum()
    return failed / total if total else 0.0


def _daily_realized_inflows(transactions_df):
    settled = transactions_df[transactions_df["status"] == "success"].copy()
    settled["settlement_date"] = pd.to_datetime(settled["settlement_date"]).dt.normalize()
    return settled.groupby("settlement_date")["amount"].sum().sort_index()


def _daily_realized_outflows(payables_df):
    p = payables_df.copy()
    p["due_date"] = pd.to_datetime(p["due_date"]).dt.normalize()
    return p.groupby("due_date")["amount"].sum().sort_index()


def _historical_starting_balance(transactions_df, payables_df, as_of, initial_balance=350_000):
   
    as_of = pd.Timestamp(as_of).normalize()
    inflow = _daily_realized_inflows(transactions_df)
    outflow = _daily_realized_outflows(payables_df)
    dates = pd.date_range(START_DATE.normalize(), as_of, freq="D")
    net = inflow.reindex(dates, fill_value=0.0) - outflow.reindex(dates, fill_value=0.0)
    return float(initial_balance + net.cumsum().iloc[-1])


def _historical_inflow_pool(transactions_df, as_of):
  
    settled = transactions_df[transactions_df["status"] == "success"].copy()
    settled["settlement_date"] = pd.to_datetime(settled["settlement_date"]).dt.normalize()
    as_of = pd.Timestamp(as_of).normalize()
    observed = settled[settled["settlement_date"] <= as_of]
    if observed.empty:
        return np.array([0.0])
    daily = observed.groupby("settlement_date")["amount"].sum()
    full_dates = pd.date_range(START_DATE.normalize(), as_of, freq="D")
    daily = daily.reindex(full_dates, fill_value=0.0)
    return daily.values


def simulate_cash_paths(starting_balance, receivables_df, payables_df, transactions_df, today,
                         horizon_days=30, n_sims=3000, include_new_business=True, seed=7,
                         historical_mode=False):
    
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today).normalize()
    dates = [today + timedelta(days=d) for d in range(1, horizon_days + 1)]
    paths = np.zeros((n_sims, horizon_days))

    inflow_pool = _historical_inflow_pool(transactions_df, today)
    fail_rate = _failure_rate(transactions_df) if len(transactions_df) else 0.0

    payables_df = payables_df.copy()
    payables_df["due_date"] = pd.to_datetime(payables_df["due_date"]).dt.normalize()
    future_payables = payables_df[payables_df["due_date"] > today]

                                                                                     
                                                                                           
    receivables_df = receivables_df.copy()
    use_receivables = not historical_mode

    for sim in range(n_sims):
        daily_net = np.zeros(horizon_days)

        if use_receivables:
            for _, r in receivables_df.iterrows():
                if rng.random() >= float(r["collection_probability"]):
                    continue
                day_offset = int(np.clip(
                    rng.exponential(scale=max(float(r["expected_days_to_collect"]), 1.0)),
                    0, horizon_days - 1))
                daily_net[day_offset] += float(r["amount"])

        for _, p in future_payables.iterrows():
            jitter = 0
            if p["category"] not in ("payroll", "tax"):
                jitter = int(rng.integers(-2, 4))
            due_offset = (p["due_date"] + timedelta(days=jitter) - today).days
            if 0 <= due_offset < horizon_days:
                daily_net[due_offset] -= float(p["amount"])

                                                                                
                                                                       
        if include_new_business and len(inflow_pool):
            sampled = rng.choice(inflow_pool, size=horizon_days, replace=True)
            if fail_rate > 0:
                success_mask = rng.random(horizon_days) >= fail_rate
                sampled = sampled * success_mask
            daily_net += sampled

        paths[sim] = starting_balance + np.cumsum(daily_net)

    return pd.DataFrame(paths, columns=[d.date().isoformat() for d in dates])


def summarize_paths(paths_df, threshold=0.0):
    p10 = paths_df.quantile(0.10, axis=0)
    p50 = paths_df.quantile(0.50, axis=0)
    p90 = paths_df.quantile(0.90, axis=0)

    breach_any_day = (paths_df.values < threshold).any(axis=1)
    prob_breach = breach_any_day.mean()

    breach_day_idx = np.argmax(paths_df.values < threshold, axis=1)
    breach_day_idx = np.where(breach_any_day, breach_day_idx, -1)
    breaching_days = breach_day_idx[breach_day_idx >= 0]
    worst_day = paths_df.columns[int(np.bincount(breaching_days).argmax())] if len(breaching_days) else None

    return {
        "fan_chart": pd.DataFrame({"P10": p10, "P50": p50, "P90": p90}),
        "prob_breach_threshold": round(float(prob_breach), 4),
        "threshold": threshold,
        "most_likely_breach_day": worst_day,
        "final_day_p10_p50_p90": (round(p10.iloc[-1], 2), round(p50.iloc[-1], 2), round(p90.iloc[-1], 2)),
    }


def backtest_calibration(transactions_df, payables_df, receivables_df, initial_balance,
                          backtest_dates, horizon_days=14, n_sims=1500):
    """Rolling-origin calibration with a calibration/holdout split and no look-ahead."""
    inflow = _daily_realized_inflows(transactions_df)
    outflow = _daily_realized_outflows(payables_df)
    all_dates = pd.date_range(START_DATE.normalize(), TODAY.normalize(), freq="D")
    realized_daily = inflow.reindex(all_dates, fill_value=0.0) - outflow.reindex(all_dates, fill_value=0.0)
    realized_cash = initial_balance + realized_daily.cumsum()

    raw_rows = []
    for as_of in backtest_dates:
        as_of = pd.Timestamp(as_of).normalize()
        if as_of + pd.Timedelta(days=horizon_days) > TODAY.normalize():
            continue
        sb = float(realized_cash.loc[as_of])
        paths = simulate_cash_paths(
            sb, receivables_df, payables_df, transactions_df, as_of,
            horizon_days=horizon_days, n_sims=n_sims,
            seed=int(as_of.strftime("%Y%m%d")), historical_mode=True)
        band = summarize_paths(paths, threshold=-np.inf)["fan_chart"]
        for col in band.index:
            target_date = pd.Timestamp(col).normalize()
            actual = float(realized_cash.loc[target_date])
            lo, hi = float(band.loc[col, "P10"]), float(band.loc[col, "P90"])
            mid, half = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1e-9)
            raw_rows.append({
                "as_of": as_of.date(), "target_date": target_date.date(),
                "actual": actual, "p10": lo, "p90": hi,
                "nonconformity": abs(actual - mid) / half,
                "inside_raw": lo <= actual <= hi,
            })

    detail = pd.DataFrame(raw_rows)
    if detail.empty:
        return {"coverage_rate": None, "raw_coverage_rate": None, "n_checks": 0,
                "calibration_multiplier": 1.0, "detail": detail}

    unique_origins = sorted(detail["as_of"].unique())
    split = max(1, len(unique_origins) // 2)
    cal_origins = set(unique_origins[:split])
    holdout_origins = set(unique_origins[split:])
    cal_scores = detail.loc[detail["as_of"].isin(cal_origins), "nonconformity"]

                                                                             
                                                                         
    if len(cal_scores):
        multiplier = float(cal_scores.quantile(0.80))
        multiplier = max(1.0, multiplier)
    else:
        multiplier = 1.0

    detail["p10_calibrated"] = ((detail["p10"] + detail["p90"]) / 2.0
                                 - multiplier * (detail["p90"] - detail["p10"]) / 2.0)
    detail["p90_calibrated"] = ((detail["p10"] + detail["p90"]) / 2.0
                                 + multiplier * (detail["p90"] - detail["p10"]) / 2.0)
    detail["inside_calibrated"] = (
        (detail["actual"] >= detail["p10_calibrated"]) &
        (detail["actual"] <= detail["p90_calibrated"])
    )

    holdout = detail[detail["as_of"].isin(holdout_origins)]
    evaluated = holdout if len(holdout) else detail
    coverage = float(evaluated["inside_calibrated"].mean())
    raw_coverage = float(evaluated["inside_raw"].mean())
    detail["is_holdout"] = detail["as_of"].isin(holdout_origins)

    return {
        "coverage_rate": coverage,
        "raw_coverage_rate": raw_coverage,
        "n_checks": int(len(evaluated)),
        "calibration_multiplier": multiplier,
        "calibration_origins": sorted(cal_origins),
        "holdout_origins": sorted(holdout_origins),
        "detail": detail,
    }


def apply_uncertainty_calibration(paths_df, multiplier):
    
    if multiplier is None or multiplier <= 1.0:
        return paths_df
    median = paths_df.median(axis=0)
    return median + (paths_df - median) * float(multiplier)


                                                                               
                         
                                                                               
def reconciliation_scorecard(recon_result):
    s = recon_result["summary"]
    exc = recon_result["exceptions"]
    exception_value_exposure = exc["amount"].abs().sum() if len(exc) else 0.0
    rows = [
        ("Total internal records", s["total_internal_records"]),
        ("Total bank records", s["total_bank_records"]),
        ("Matched records", s["matched_count"]),
        ("Match rate", f"{s['match_rate']*100:.1f}%"),
        ("Auto-match precision (synthetic ground truth)", f"{s.get('auto_match_precision', float('nan'))*100:.1f}%"),
        ("Auto-match recall (synthetic ground truth)", f"{s.get('auto_match_recall', float('nan'))*100:.1f}%"),
        ("Unresolved exceptions", s["exception_count"]),
        ("Exception value exposure (Rs.)", f"{exception_value_exposure:,.2f}"),
    ]
    rows.append(("Reconciliation throughput (records/sec)",
                 f"{s.get('records_per_second', float('nan')):,.1f}"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def forecast_scorecard(forecast_summary, calibration=None):
    p10, p50, p90 = forecast_summary["final_day_p10_p50_p90"]
    rows = [
        ("Final-day P10", f"Rs.{p10:,.0f}"),
        ("Final-day P50 (median)", f"Rs.{p50:,.0f}"),
        ("Final-day P90", f"Rs.{p90:,.0f}"),
        ("Band width (P90-P10)", f"Rs.{p90-p10:,.0f}"),
        ("Prob. of breaching threshold", f"{forecast_summary['prob_breach_threshold']*100:.1f}%"),
        ("Most likely breach day", forecast_summary["most_likely_breach_day"] or "-"),
    ]
    if calibration is not None and calibration.get("coverage_rate") is not None:
        rows.append(("Raw backtest P10-P90 coverage", f"{calibration['raw_coverage_rate']*100:.1f}%"))
        rows.append(("Calibrated holdout coverage", f"{calibration['coverage_rate']*100:.1f}%"))
        rows.append(("Uncertainty calibration multiplier", f"{calibration['calibration_multiplier']:.2f}x"))
        rows.append(("Backtest holdout checks", calibration["n_checks"]))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def exception_cost_by_reason(recon_result):
    exc = recon_result["exceptions"]
    if not len(exc):
        return pd.DataFrame(columns=["reason", "count", "total_amount"])
    g = exc.groupby("reason").agg(count=("amount", "size"), total_amount=("amount", lambda x: x.abs().sum()))
    return g.reset_index().sort_values("total_amount", ascending=False)


                                                                               
                                                                
                                                                               
MODEL_ID = "ibm-granite/granite-4.1-8b"

SYSTEM_PROMPT = """You are CashCast, an AI finance controller assistant.

Your job is to EXPLAIN financial facts computed by Python. Python is the source
of truth; you are not the calculator and you must not reconstruct or reinterpret
financial records.

STRICT RULES:
1. Use ONLY facts in STRUCTURED_FACTS. Never invent, infer, or rename a category.
2. Preserve exact counts, IDs, dates, amounts, probabilities, and forecast values.
3. Currency is INR (Rs./₹). NEVER write $, USD, or convert currencies.
4. If a fact is not present, say: "The structured facts do not contain that information."
5. For reconciliation, use the exact exception categories and counts supplied.
6. Never attach a different reason to a transaction ID than the reason in the facts.
7. Distinguish FACT from INFERENCE from ACTION.
8. Do not recommend executing, delaying, cancelling, or changing a payment unless
   an explicitly authorized action is present in the facts. Otherwise say to REVIEW
   the item and do not prescribe a timing change.
9. For cash-shortfall questions, use the supplied forecast probability, threshold,
   breach date, and scheduled payables. Do not calculate a new probability.
10. Be concise, auditable, and faithful to the structured facts.
"""


class GraniteAgent:
    def __init__(self, model_id=MODEL_ID, load_in_4bit=True, device_map="auto", max_new_tokens=350):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quant_kwargs = {}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                pass
        model_kwargs = dict(device_map=device_map, **quant_kwargs)
                                                                    
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **model_kwargs,
            )

    def _generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def _generate_action(self, messages):
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        stopping = None
        if StoppingCriteriaList is not None:
            stopping = StoppingCriteriaList([_FirstJSONObjectStop(self.tokenizer, inputs["input_ids"].shape[1])])
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def ask(self, question, structured_facts):
        facts_json = json.dumps(structured_facts, indent=2, ensure_ascii=False, default=str)
        user_prompt = f"""STRUCTURED_FACTS (authoritative Python output):
{facts_json}

QUESTION:
{question}

Answer using only STRUCTURED_FACTS. Do not add facts that are absent.
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._generate(messages)


class MockAgent:
    def ask(self, question, structured_facts):
        return "[MOCK AGENT -- no LLM loaded]"

    def _generate(self, messages):
        
        last = messages[-1]["content"] if messages else ""
        if last.startswith("TOOL_RESULT"):
            return json.dumps({"final_answer": "[MOCK AGENT] Answer based on the tool result above."})
        return json.dumps({"tool": "get_reconciliation_overview", "args": {}})


ID_PATTERN = re.compile(r"\b(?:TXN\d{5,}(?:-DUP)?|PAY\d{4,}|INV\d{4,}|BANKFEE\d+)\b")
CURRENCY_BAD_PATTERN = re.compile(r"\$|\bUSD\b|\bUS\s*dollars?\b", re.IGNORECASE)


def categorize_exceptions(exc):
    
    categories = {
        "missing_in_bank_statement": 0,
        "missing_in_internal_ledger": 0,
        "amount_or_date_mismatch": 0,
        "bank_only_fee_or_interest": 0,
    }
    if exc is None or not len(exc):
        return categories
    for _, r in exc.iterrows():
        reason = str(r["reason"])
        if reason == "missing_in_bank_statement":
            categories["missing_in_bank_statement"] += 1
        elif reason == "missing_in_internal_ledger":
            categories["missing_in_internal_ledger"] += 1
        elif reason == "bank_only_fee_or_interest":
            categories["bank_only_fee_or_interest"] += 1
        elif "exceeds tolerance" in reason:
            categories["amount_or_date_mismatch"] += 1
    return categories


def build_structured_facts(recon_result, forecast_summary, payables_df, receivables_df,
                           starting_balance, today, calibration):
   
    exc = recon_result["exceptions"]
    s = recon_result["summary"]

    exception_records = []
    for _, r in exc.iterrows():
        exception_records.append({
            "ref_id": str(r["ref_id"]),
            "side": str(r["side"]),
            "amount_inr": round(float(r["amount"]), 2),
            "date": str(r["date"]),
            "reason": str(r["reason"]),
        })

                                                                               
                                                                                    
    categories = categorize_exceptions(exc)

    p10, p50, p90 = forecast_summary["final_day_p10_p50_p90"]
    payable_rows = []
    for _, r in payables_df.sort_values("amount", ascending=False).head(10).iterrows():
        payable_rows.append({
            "payable_id": str(r["payable_id"]),
            "vendor": str(r["vendor"]),
            "category": str(r["category"]),
            "due_date": str(r["due_date"]),
            "amount_inr": round(float(r["amount"]), 2),
            "action_authorized": False,
        })

    receivable_rows = []
    for _, r in receivables_df.sort_values("amount", ascending=False).head(10).iterrows():
        receivable_rows.append({
            "invoice_id": str(r["invoice_id"]),
            "aging_bucket": str(r["aging_bucket"]),
            "collection_probability": float(r["collection_probability"]),
            "amount_inr": round(float(r["amount"]), 2),
        })

    return {
        "currency": "INR",
        "as_of_date": str(today),
        "reconciliation": {
            "total_internal_records": int(s["total_internal_records"]),
            "total_bank_records": int(s["total_bank_records"]),
            "matched_records": int(s["matched_count"]),
            "match_rate_percent": round(float(s["match_rate"] * 100), 2),
            "auto_match_precision_percent": round(float(s["auto_match_precision"] * 100), 2),
            "auto_match_recall_percent": round(float(s["auto_match_recall"] * 100), 2),
            "exception_count": int(s["exception_count"]),
            "exception_categories": categories,
            "exception_records": exception_records,
        },
        "forecast": {
            "starting_cash_inr": round(float(starting_balance), 2),
            "horizon_days": 30,
            "threshold_inr": round(float(forecast_summary["threshold"]), 2),
            "final_day_p10_inr": round(float(p10), 2),
            "final_day_p50_inr": round(float(p50), 2),
            "final_day_p90_inr": round(float(p90), 2),
            "breach_probability_percent": round(float(forecast_summary["prob_breach_threshold"] * 100), 2),
            "most_likely_breach_day": str(forecast_summary["most_likely_breach_day"]),
            "raw_backtest_coverage_percent": round(float(calibration["raw_coverage_rate"] * 100), 2),
            "calibrated_holdout_coverage_percent": round(float(calibration["coverage_rate"] * 100), 2),
            "calibration_multiplier": round(float(calibration["calibration_multiplier"]), 4),
            "backtest_checks": int(calibration["n_checks"]),
        },
        "scheduled_payables": payable_rows,
        "open_receivables": receivable_rows,
        "authorized_actions": [],
    }


def verify_granite_answer(answer_text, facts):
    
    recon = facts["reconciliation"]
    forecast = facts["forecast"]
    all_ids = {r["ref_id"] for r in recon["exception_records"]}
    all_ids |= {r["payable_id"] for r in facts["scheduled_payables"]}
    all_ids |= {r["invoice_id"] for r in facts["open_receivables"]}

    mentioned_ids = set(ID_PATTERN.findall(answer_text))
    unknown_ids = sorted(mentioned_ids - all_ids)

    problems = []
    if unknown_ids:
        problems.append(f"unknown IDs: {unknown_ids}")
    if CURRENCY_BAD_PATTERN.search(answer_text):
        problems.append("wrong currency reference; currency is INR")

                                                                            
                                                                           
    allowed = set(recon["exception_categories"])
    category_aliases = re.findall(
        r"\b([a-z_]+(?:_[a-z_]+)+)\b\s*[:=-]\s*\d+",
        answer_text.lower(),
    )
    invented_categories = sorted({c for c in category_aliases if c not in allowed})
    if invented_categories:
        problems.append(f"unsupported exception categories: {invented_categories}")

                                                                              
                                           
    action_pattern = re.search(
        r"\b(delay|postpone|cancel|move|reschedule|extend)\b[^.\n]*(PAY\d{4,})",
        answer_text,
        re.IGNORECASE,
    )
    if action_pattern and not facts["authorized_actions"]:
        problems.append("unauthorized payment action recommendation")

                                                                       
    expected_tokens = [
        f"{forecast['breach_probability_percent']:.1f}%",
        str(int(round(forecast["threshold_inr"]))),
    ]
                                                                             
                                                                       

    return {
        "is_fully_verified": len(problems) == 0,
        "problems": problems,
        "mentioned_ids": sorted(mentioned_ids),
        "unknown_ids": unknown_ids,
    }


def ask_with_structured_grounding(agent, question, facts):
    answer = agent.ask(question, facts)
    report = verify_granite_answer(answer, facts)

                                                                             
                                                                             
    if not report["is_fully_verified"] and isinstance(agent, GraniteAgent):
        correction = (
            "AUDIT FAILURE. Rewrite your previous answer using ONLY the supplied "
            "STRUCTURED_FACTS. Correct these issues: " + "; ".join(report["problems"]) +
            ". Do not invent categories, IDs, currencies, or payment actions. "
            "If an action is not authorized, say REVIEW rather than prescribing it."
        )
        facts_json = json.dumps(facts, indent=2, ensure_ascii=False, default=str)
        answer = agent._generate([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"STRUCTURED_FACTS:\n{facts_json}\n\nQUESTION:\n{question}"},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": correction},
        ])
        report = verify_granite_answer(answer, facts)

    return answer, report


                                                                               
                                             
                                                                               
def main():
    USE_MOCK_AGENT = False                                                       
    STARTING_BALANCE = 350_000
    HORIZON_DAYS = 30
                                                                                            
                                                                 
    MIN_CASH_THRESHOLD = 200_000

    print("=" * 70)
    print("STEP 1: Generating synthetic data")
    print("=" * 70)
    txns, receivables, payables, internal_ledger, bank_statement = generate_all_data(out_dir=".")
    print(f"transactions: {len(txns)} | receivables: {len(receivables)} | "
          f"payables: {len(payables)} | internal_ledger: {len(internal_ledger)} | "
          f"bank_statement: {len(bank_statement)}")
    print(f"'today' (as-of date) = {TODAY.date()}")

    print("\n" + "=" * 70)
    print("STEP 2: Reconciliation")
    print("=" * 70)
    import time
    recon_start = time.perf_counter()
    recon_result = reconcile(internal_ledger, bank_statement)
    recon_seconds = time.perf_counter() - recon_start
    throughput = len(internal_ledger) / max(recon_seconds, 1e-9)
    recon_result["summary"]["processing_seconds"] = recon_seconds
    recon_result["summary"]["records_per_second"] = throughput
    print(f"Match rate: {recon_result['summary']['match_rate']*100:.1f}%")
    print(f"Auto-match precision (synthetic ground truth): {recon_result['summary']['auto_match_precision']*100:.1f}%")
    print(f"Auto-match recall (synthetic ground truth): {recon_result['summary']['auto_match_recall']*100:.1f}%")
    print(f"Reconciliation throughput: {throughput:,.1f} internal records/sec")
    print(f"Exceptions: {recon_result['summary']['exception_count']}")
    print(recon_result["exceptions"].head(10).to_string())

    print("\n" + "=" * 70)
    print("STEP 3: Monte Carlo cash forecast")
    print("=" * 70)

                                                                        
                                                                            
    raw_paths = simulate_cash_paths(
        STARTING_BALANCE,
        receivables,
        payables,
        txns,
        TODAY,
        horizon_days=HORIZON_DAYS,
        n_sims=3000,
        include_new_business=True,
        seed=7,
        historical_mode=False,
    )

    print(f"Generated {len(raw_paths):,} raw Monte Carlo paths.")
    print("Final forecast will use the uncertainty-calibrated paths from Step 4.")

    print("\n" + "=" * 70)
    print("STEP 4: Calibration backtest + final calibrated forecast")
    print("=" * 70)

                                                                               
                                                                  
    backtest_dates = [START_DATE + pd.Timedelta(days=d) for d in [35, 42, 49, 56, 63, 70]]
    calibration = backtest_calibration(
        txns,
        payables,
        receivables,
        STARTING_BALANCE,
        backtest_dates,
        horizon_days=14,
        n_sims=1500,
    )

                                                                   
                                                                               
    calibrated_paths = apply_uncertainty_calibration(
        raw_paths,
        calibration.get("calibration_multiplier", 1.0),
    )

                                                                   
    forecast_summary = summarize_paths(
        calibrated_paths,
        threshold=MIN_CASH_THRESHOLD,
    )

    p10, p50, p90 = forecast_summary["final_day_p10_p50_p90"]

    print(f"Raw backtest P10-P90 coverage: {calibration['raw_coverage_rate']*100:.1f}%")
    print(f"Calibrated holdout coverage: {calibration['coverage_rate']*100:.1f}%  (n={calibration['n_checks']})")
    print(f"Uncertainty calibration multiplier: {calibration['calibration_multiplier']:.2f}x")
    print("Backtest uses rolling historical origins with no look-ahead; future transaction inflows are withheld from each forecast.")

                                                                            
                                               
    print(f"Probability of breaching Rs.{MIN_CASH_THRESHOLD:,} within {HORIZON_DAYS} days: "
          f"{forecast_summary['prob_breach_threshold']*100:.1f}%")
    print(f"Most likely breach day: {forecast_summary['most_likely_breach_day']}")
    print(f"Day {HORIZON_DAYS} forecast: P10=Rs.{p10:,.0f}  P50=Rs.{p50:,.0f}  P90=Rs.{p90:,.0f}")

                                                                          
                                              
    try:
        import matplotlib.pyplot as plt
        fan = forecast_summary["fan_chart"]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = range(len(fan))
        ax.fill_between(x, fan["P10"], fan["P90"], alpha=0.25, label="Calibrated P10-P90 band")
        ax.plot(x, fan["P50"], linewidth=2, label="P50 (median forecast)")
        ax.axhline(MIN_CASH_THRESHOLD, color="red", linestyle="--", linewidth=1, label="Minimum cash threshold")
        ax.set_xticks(list(x)[::3])
        ax.set_xticklabels([fan.index[i] for i in x][::3], rotation=45, ha="right")
        ax.set_ylabel("Cash balance (Rs.)")
        ax.set_title(f"CashCast -- calibrated {HORIZON_DAYS}-day Monte Carlo Cash Forecast ({len(calibrated_paths):,} paths)")
        ax.legend()
        plt.tight_layout()
        plt.savefig("cashcast_fan_chart.png", dpi=140)
        plt.close(fig)
        print("Fan chart saved to cashcast_fan_chart.png")
    except Exception as e:
        print(f"(Skipping plot: {e})")

    print("\n" + "=" * 70)
    print("STEP 5: Scorecards")
    print("=" * 70)
    print("\n--- Reconciliation scorecard ---")
    print(reconciliation_scorecard(recon_result).to_string(index=False))
    print("\n--- Forecast scorecard ---")
    print(forecast_scorecard(forecast_summary, calibration).to_string(index=False))
    print("\n--- Exception cost by reason ---")
    print(exception_cost_by_reason(recon_result).to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 6: Granite agent -- structured-facts Q&A")
    print("=" * 70)
    if USE_MOCK_AGENT:
        agent = MockAgent()
        print("(Using MockAgent -- real Granite disabled)")
    else:
        agent = GraniteAgent(load_in_4bit=True)

    structured_facts = build_structured_facts(
        recon_result=recon_result,
        forecast_summary=forecast_summary,
        payables_df=payables,
        receivables_df=receivables,
        starting_balance=STARTING_BALANCE,
        today=TODAY.date(),
        calibration=calibration,
    )

    print("Granite grounding source: STRUCTURED_FACTS (Python authoritative output)")
    print(f"Grounded exception records: {len(structured_facts['reconciliation']['exception_records'])}")
    print(f"Grounded payables: {len(structured_facts['scheduled_payables'])}")
    print(f"Grounded receivables: {len(structured_facts['open_receivables'])}")
    print(f"Currency: {structured_facts['currency']}")

    questions = [
        "Why is the reconciliation match rate not 100%? Summarize the main causes.",
        "What is the single biggest driver of cash-shortfall risk in the next 30 days, "
        "and what should I review to reduce the risk without assuming that any payment "
        "can legally or contractually be delayed?",
        "Which reconciliation exceptions represent real money at risk versus routine bank fees?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer, report = ask_with_structured_grounding(agent, q, structured_facts)
        print(f"A: {answer}")
        if report["is_fully_verified"]:
            print("[grounding check: OK -- structured facts verified]")
        else:
            print("[grounding check: FLAGGED]")
            for problem in report["problems"]:
                print(f"  - {problem}")

    print("\n" + "=" * 70)
    print("DONE. CSVs + cashcast_fan_chart.png written to current directory.")
    print("=" * 70)



                                                                               
                                                          
                                                                               

STARTING_BALANCE = 350_000
HORIZON_DAYS = 30
MIN_CASH_THRESHOLD = 200_000

def build_world_state():
    txns, receivables, payables, internal_ledger, bank_statement = generate_all_data(out_dir=".")

    import time
    recon_start = time.perf_counter()
    recon_result = reconcile(internal_ledger, bank_statement)
    recon_seconds = time.perf_counter() - recon_start
    recon_result["summary"]["processing_seconds"] = recon_seconds
    recon_result["summary"]["records_per_second"] = len(internal_ledger) / max(recon_seconds, 1e-9)

    raw_paths = simulate_cash_paths(
        STARTING_BALANCE, receivables, payables, txns, TODAY,
        horizon_days=HORIZON_DAYS, n_sims=3000, include_new_business=True,
        seed=7, historical_mode=False,
    )

    backtest_dates = [START_DATE + pd.Timedelta(days=d) for d in [35, 42, 49, 56, 63, 70]]
    calibration = backtest_calibration(
        txns, payables, receivables, STARTING_BALANCE, backtest_dates,
        horizon_days=14, n_sims=1500,
    )
    calibrated_paths = apply_uncertainty_calibration(raw_paths, calibration.get("calibration_multiplier", 1.0))
    forecast_summary = summarize_paths(calibrated_paths, threshold=MIN_CASH_THRESHOLD)

    return {
        "txns": txns, "receivables": receivables, "payables": payables,
        "internal_ledger": internal_ledger, "bank_statement": bank_statement,
        "recon_result": recon_result, "calibration": calibration,
        "forecast_summary": forecast_summary, "calibrated_paths": calibrated_paths,
        "pending_actions": [],
    }


def tool_get_reconciliation_overview(WORLD, **_):
    s = WORLD["recon_result"]["summary"]
    exc = WORLD["recon_result"]["exceptions"]
    return {
        "total_internal_records": s["total_internal_records"],
        "total_bank_records": s["total_bank_records"],
        "matched_records": s["matched_count"],
        "match_rate_percent": round(s["match_rate"] * 100, 2),
        "auto_match_precision_percent": round(s.get("auto_match_precision", 0) * 100, 2),
        "auto_match_recall_percent": round(s.get("auto_match_recall", 0) * 100, 2),
        "exception_count": s["exception_count"],
                                                                             
                                                                             
                                                                    
        "exception_category_counts": categorize_exceptions(exc),
        "throughput_records_per_sec": round(s.get("records_per_second", 0), 1),
    }


def tool_list_exceptions(WORLD, category=None, limit=10, **_):
    exc = WORLD["recon_result"]["exceptions"]
    if not len(exc):
        return {"records": [], "total_matching": 0}
    df = exc.copy()
    if category:
        if category == "amount_or_date_mismatch":
            df = df[df["reason"].str.contains("exceeds tolerance", na=False)]
        else:
            df = df[df["reason"] == category]
    limit = max(1, min(int(limit), 50))
    out = df.head(limit)[["ref_id", "side", "amount", "date", "reason"]].copy()
    out["amount"] = out["amount"].round(2)
    out["date"] = out["date"].astype(str)
    return {"records": out.to_dict(orient="records"), "total_matching": len(df)}


def tool_get_exception_detail(WORLD, ref_id=None, **_):
    exc = WORLD["recon_result"]["exceptions"]
    if not ref_id or not len(exc):
        return {"found": False}
    row = exc[exc["ref_id"] == ref_id]
    if row.empty:
        return {"found": False, "note": f"No exception record exists for ref_id={ref_id!r}."}
    r = row.iloc[0]
    return {"found": True, "ref_id": r["ref_id"], "side": r["side"],
            "amount_inr": round(float(r["amount"]), 2), "date": str(r["date"]), "reason": r["reason"]}


def tool_get_forecast_overview(WORLD, **_):
    fs = WORLD["forecast_summary"]
    cal = WORLD["calibration"]
    p10, p50, p90 = fs["final_day_p10_p50_p90"]
    return {
        "starting_cash_inr": STARTING_BALANCE,
        "horizon_days": HORIZON_DAYS,
        "threshold_inr": fs["threshold"],
        "final_day_p10_inr": p10, "final_day_p50_inr": p50, "final_day_p90_inr": p90,
        "breach_probability_percent": round(fs["prob_breach_threshold"] * 100, 2),
        "most_likely_breach_day": str(fs["most_likely_breach_day"]),
        "raw_backtest_coverage_percent": round((cal.get("raw_coverage_rate") or 0) * 100, 2),
        "calibrated_holdout_coverage_percent": round((cal.get("coverage_rate") or 0) * 100, 2),
        "calibration_multiplier": round(cal.get("calibration_multiplier", 1.0), 4),
    }


def tool_get_forecast_on_date(WORLD, date=None, **_):
    fan = WORLD["forecast_summary"]["fan_chart"]
    if date not in fan.index:
        return {"found": False, "available_dates": list(fan.index)}
    row = fan.loc[date]
    return {"found": True, "date": date, "p10_inr": round(float(row["P10"]), 2),
            "p50_inr": round(float(row["P50"]), 2), "p90_inr": round(float(row["P90"]), 2)}


def tool_list_payables(WORLD, category=None, limit=10, **_):
    df = WORLD["payables"].copy()
    if category:
        df = df[df["category"] == category]
    limit = max(1, min(int(limit), 50))
    out = df.sort_values("amount", ascending=False).head(limit)
    return {"records": out[["payable_id", "vendor", "category", "due_date", "amount"]]
            .assign(amount=lambda d: d["amount"].round(2), due_date=lambda d: d["due_date"].astype(str))
            .to_dict(orient="records"), "total_matching": len(df)}


def tool_list_receivables(WORLD, aging_bucket=None, limit=10, **_):
    df = WORLD["receivables"].copy()
    if aging_bucket:
        df = df[df["aging_bucket"] == aging_bucket]
    limit = max(1, min(int(limit), 50))
    out = df.sort_values("amount", ascending=False).head(limit)
    return {"records": out[["invoice_id", "aging_bucket", "collection_probability", "amount"]]
            .assign(amount=lambda d: d["amount"].round(2)).to_dict(orient="records"),
            "total_matching": len(df)}


def tool_propose_action(WORLD, action_type=None, target_id=None, rationale=None, **_):
    entry = {"action_type": action_type, "target_id": target_id, "rationale": rationale,
             "status": "pending_human_approval"}
    WORLD["pending_actions"].append(entry)
    return {"logged": True, "status": "pending_human_approval",
            "note": "This action has been queued for human review. It has NOT been executed."}


TOOL_REGISTRY = {
    "get_reconciliation_overview": {
        "fn": tool_get_reconciliation_overview,
        "description": "Get aggregate reconciliation stats: match rate, precision/recall, and exception counts already grouped into exactly 4 categories that sum to exception_count. Do not re-derive or re-sum these -- read them directly.",
        "args": {},
    },
    "list_exceptions": {
        "fn": tool_list_exceptions,
        "description": "List reconciliation exception records, optionally filtered by category.",
        "args": {"category": "optional string: missing_in_bank_statement | missing_in_internal_ledger | bank_only_fee_or_interest | amount_or_date_mismatch", "limit": "optional int, default 10, max 50"},
    },
    "get_exception_detail": {
        "fn": tool_get_exception_detail,
        "description": "Look up the full detail of ONE exception by its exact ref_id. Use this before making any specific claim about a single transaction ID.",
        "args": {"ref_id": "required string, e.g. TXN100001"},
    },
    "get_forecast_overview": {
        "fn": tool_get_forecast_overview,
        "description": "Get the 30-day cash forecast summary: P10/P50/P90, breach probability, calibration quality.",
        "args": {},
    },
    "get_forecast_on_date": {
        "fn": tool_get_forecast_on_date,
        "description": "Get the P10/P50/P90 cash forecast for one specific date in the horizon.",
        "args": {"date": "required string YYYY-MM-DD"},
    },
    "list_payables": {
        "fn": tool_list_payables,
        "description": "List scheduled outflows (payables), optionally filtered by category, sorted by amount descending.",
        "args": {"category": "optional string: payroll | vendor | tax | rent | loan_emi", "limit": "optional int, default 10, max 50"},
    },
    "list_receivables": {
        "fn": tool_list_receivables,
        "description": "List open receivables (unpaid invoices), optionally filtered by aging bucket, sorted by amount descending.",
        "args": {"aging_bucket": "optional string: 0-15d | 16-30d | 31-60d | 60d+", "limit": "optional int, default 10, max 50"},
    },
    "propose_action": {
        "fn": tool_propose_action,
        "description": "The ONLY way to recommend a concrete action (e.g. delaying a payable, flagging an exception for finance to review). Logs it for human approval -- never executes it. Use this instead of just describing an action in prose.",
        "args": {"action_type": "required string", "target_id": "required string, an id from a tool result", "rationale": "required string, 1-2 sentences"},
    },
}


def _tools_catalog_text():
    lines = ["Available tools:"]
    for name, spec in TOOL_REGISTRY.items():
        lines.append(f"- {name}({', '.join(spec['args'].keys())}): {spec['description']}")
        for arg, desc in spec["args"].items():
            lines.append(f"    {arg}: {desc}")
    return "\n".join(lines)


AGENT_SYSTEM_PROMPT = f"""You are CashCast, an AI finance controller assistant.

You do NOT have direct access to the merchant's data. You can only see it by
calling tools. Python owns every number; you only explain and reason.

{_tools_catalog_text()}

PROTOCOL -- follow this exactly, every turn:
- To call a tool, respond with ONLY a single JSON object and nothing else:
  {{"tool": "<tool_name>", "args": {{...}}}}
- When you have enough information to answer the user, respond with ONLY:
  {{"final_answer": "<your complete answer to the user>"}}
- Output EXACTLY ONE JSON object, then STOP. Do not write anything before
  or after it, and do not write a second JSON object in the same reply --
  you will get another turn once the tool result comes back.
- Never answer from memory or guess a number. If a tool returns
  "found": false, say so plainly -- do not invent a substitute.
- When writing final_answer, you MUST cite the specific numbers, IDs, and
  categories that the tool result actually returned. A final_answer that
  only restates field names or describes what the data "would show"
  without citing the real values is not acceptable.
- Call get_exception_detail before stating specifics about any single
  transaction ID you were not already given verbatim by a tool result.
- Currency is INR (Rs./₹). Never use $ or USD.
- To recommend any concrete action (delaying a payment, flagging something for
  review), you MUST call propose_action. Do not just describe the action in prose
  -- log it, then reference that you logged it.
- Keep the loop short: gather only what you need, then answer.
"""

ACTION_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)                                                        

def _parse_agent_action(raw_text):
    return _extract_first_json_object(raw_text)

def _dispatch_tool(WORLD, tool_name, args):
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return {"error": f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"}
    try:
        return spec["fn"](WORLD, **(args or {}))
    except TypeError as e:
        return {"error": f"Bad arguments for '{tool_name}': {e}"}

def run_agent_turn(agent, WORLD, user_message, chat_history, max_tool_calls=5):
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

                                                                            
                                                                                      
    generate_fn = getattr(agent, "_generate_action", None) or agent._generate

    trace = []
    for step in range(max_tool_calls + 1):
        raw = generate_fn(messages)
        action = _parse_agent_action(raw)
        if action is None:
            return raw.strip(), trace
        if "final_answer" in action:
            return str(action["final_answer"]).strip(), trace
        if "tool" in action:
            tool_name = action["tool"]
            args = action.get("args", {})
            result = _dispatch_tool(WORLD, tool_name, args)
            trace.append({"tool": tool_name, "args": args, "result": result})
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"TOOL_RESULT for {tool_name}({args}):\n{json.dumps(result, default=str)}\n\nContinue: respond with ONLY one JSON object -- another tool-call, or a final_answer that cites the specific values above. Do not write anything after the closing brace."})
            continue
        return raw.strip(), trace
    return ("I couldn't finish gathering the data I needed within the tool-call budget. Try asking a narrower question."), trace


                                                                           
                                                                             
                                                                      
def verify_against_trace(answer_text, trace):
    
    seen_ids = set()
    for call in trace:
        seen_ids |= set(ID_PATTERN.findall(json.dumps(call["result"], default=str)))

    mentioned = set(ID_PATTERN.findall(answer_text))
    unknown = sorted(mentioned - seen_ids)

    problems = []
    if unknown:
        problems.append(
            f"IDs cited that never appeared in any tool result this turn: {unknown}"
        )
    if CURRENCY_BAD_PATTERN.search(answer_text):
        problems.append("wrong currency reference; currency is INR")
    if not trace and mentioned:
        problems.append("cited specific IDs without calling any tool")

    return {
        "is_fully_verified": len(problems) == 0,
        "problems": problems
    }
def launch_ui(use_mock_agent=False):
    print("Building world state (data gen -> reconciliation -> Monte Carlo -> calibration)...")
    WORLD = build_world_state()
    print("World state ready.")
    print(reconciliation_scorecard(WORLD["recon_result"]).to_string(index=False))
    print(forecast_scorecard(WORLD["forecast_summary"], WORLD["calibration"]).to_string(index=False))

    agent = MockAgent() if use_mock_agent else GraniteAgent(load_in_4bit=True)
    import gradio as gr

    def respond(user_message, history):
        normalized_history = []
        for item in history or []:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant") and content is not None:
                    normalized_history.append({"role": role, "content": str(content)})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if item[0]:
                    normalized_history.append({"role": "user", "content": str(item[0])})
                if item[1]:
                    normalized_history.append({"role": "assistant", "content": str(item[1])})

        answer, trace = run_agent_turn(agent, WORLD, user_message, normalized_history)
        if isinstance(answer, list):
           parts = []
           for item in answer:
             if isinstance(item, dict):
               if "text" in item:
                parts.append(str(item["text"]))
               elif "content" in item:
                parts.append(str(item["content"]))
             else:
                parts.append(str(item))
           answer = "\n".join(parts)
        elif isinstance(answer, dict):
           answer = str(answer.get("text", answer.get("content", answer)))

        answer = str(answer)
        report = verify_against_trace(answer, trace)
        tool_lines = [f"🔧 `{c['tool']}({c['args']})`" for c in trace]
        tool_block = ("\n\n---\n**Tools used this turn:**\n" + "\n".join(tool_lines)) if tool_lines else ""
        flag_block = ""
        if not report["is_fully_verified"]:
            flag_block = "\n\n **Grounding check flagged:** " + "; ".join(report["problems"])
        return answer + tool_block + flag_block

    demo = gr.ChatInterface(
        fn=respond,
        title="CashCast — AI Finance Controller",
        description=("Ask about reconciliation exceptions, the cash forecast, payables, or receivables. "
                     "Every answer is backed by real tool calls -- expand 'Tools used' to see the audit trail."),
        examples=[
            "Why isn't the reconciliation match rate 100%?",
            "What's the cash forecast for 2026-08-27?",
            "Which payables are due in the next two weeks?",
            "Should I be worried about a cash shortfall? What should I review?",
        ],
    )
    demo.launch(share=True,debug=True)


if __name__ == "__main__":
    launch_ui(use_mock_agent=False)