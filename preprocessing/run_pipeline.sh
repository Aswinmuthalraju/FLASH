#!/usr/bin/env bash
# Reproducible real-data pipeline. Stages 1-3 = data; 4 = smoke train; 5-6 = FULL training (needs sign-off).
# Usage:  bash preprocessing/run_pipeline.sh [smoke|verify-mock|data|verify|audit|train-light|train|evaluate]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
START=${START:-2024-06-01}       # news coverage window (confirm against reports/data_prep/news_adapt_log.json)
END=${END:-2025-01-31}

case "${1:-smoke}" in
  smoke)   # offline: mock HF-shaped data, stub encoder - checks plumbing only
    $PY preprocessing/make_mock_hf.py
    $PY preprocessing/prepare_market.py --start 2024-01-01 --end 2024-02-28 --source data/raw/hf_mock/minute.parquet --out data/raw/market_mock --log-dir reports/data_prep/smoke_mock
    $PY preprocessing/prepare_news.py --source data/raw/hf_mock/money_control_news.parquet --out data/raw/news_mock.csv --undated-out data/interim/news_undated_mock.csv --log-dir reports/data_prep/smoke_mock
    $PY preprocessing/verify_dataset.py --market data/raw/market_mock --news data/raw/news_mock.csv --stub-encoder --config configs/smoke.yaml --out-dir reports/data_prep/smoke_mock
    $PY training/run_with_config.py configs/smoke.yaml scripts/train.py --market data/raw/market_mock --news data/raw/news_mock.csv --stub-encoder ;;
  data)    # downloads from huggingface.co (news first, so its date range can set START/END)
    $PY preprocessing/prepare_news.py
    $PY preprocessing/prepare_market.py --start "$START" --end "$END" ;;
  verify)  # REAL data + real FinBERT; exits non-zero on any critical issue. Never falls back to mock data.
    if [ -z "$(ls data/raw/market/*.parquet 2>/dev/null)" ] || [ ! -f data/raw/news.csv ]; then
      echo "verify: real data not found (data/raw/market/*.parquet, data/raw/news.csv). Run the 'data' stage first," >&2
      echo "        or use 'verify-mock' to check plumbing on mock data." >&2; exit 3; fi
    $PY preprocessing/verify_dataset.py --config configs/real.yaml ;;
  verify-mock)  # plumbing only: mock HF-shaped data + stub encoder (run 'smoke' first). NOT a data-quality result.
    $PY preprocessing/verify_dataset.py --market data/raw/market_mock --news data/raw/news_mock.csv --stub-encoder --config configs/smoke.yaml --out-dir reports/data_prep/smoke_mock ;;
  audit)   # REAL data: leakage audit (stub encoder - timestamps/labels/splits only) + baselines on VALIDATION
    if [ -z "$(ls data/raw/market/*.parquet 2>/dev/null)" ]; then echo "audit: real data not found" >&2; exit 3; fi
    $PY preprocessing/leakage_audit.py --config configs/real.yaml
    $PY evaluation/run_baselines.py --config configs/real.yaml --market data/raw/market --split val --out reports/baselines/real_val_baselines.json ;;
  audit-mock)  # plumbing only (run 'smoke' first)
    $PY preprocessing/leakage_audit.py --market data/raw/market_mock --news data/raw/news_mock.csv --config configs/smoke.yaml --out reports/data_prep/smoke_mock/leakage_audit.json ;;
  train-light)  # initial experiment: first ~6 weeks only
    $PY preprocessing/prepare_market.py --start "$START" --end "${LIGHT_END:-2024-07-15}" --source data/raw/hf/minute_subset.parquet --out data/raw/market_light
    $PY training/run_with_config.py configs/real.yaml scripts/train.py --market data/raw/market_light --news data/raw/news.csv ;;
  train)   # FULL run - only after the team reviews reports/data_prep/
    $PY training/run_with_config.py configs/real.yaml scripts/train.py --market data/raw/market --news data/raw/news.csv ;;
  evaluate)
    $PY training/run_with_config.py configs/real.yaml scripts/evaluate.py --market data/raw/market --news data/raw/news.csv
    $PY training/run_with_config.py configs/real.yaml scripts/benchmark.py ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
