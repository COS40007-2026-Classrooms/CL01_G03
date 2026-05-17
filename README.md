# COS40007 Group Task 3 - MLOps Pipeline with Retraining & Monitoring

**Group:** CL01_G03  
**Unit:** COS40007 Artificial Intelligence Engineering

## Overview
Automated ML retraining pipeline with drift detection and monitoring using GitHub Actions and DVC.

## Structure
- `.github/workflows/` — GitHub Actions workflow
- `src/` — Training, evaluation, monitoring scripts
- `data/` — New data uploads (triggers retraining)
- `train/` / `test/` — Training and test datasets
- `artifacts/` — Model, scaler, metrics storage
- `monitoring/` — Drift reports and logs
- `reports/` — Performance reports
- `dvc.yaml` — DVC pipeline definition

## How to trigger retraining
1. Add new data to `data/new_data.csv`
2. `git add data/new_data.csv`
3. `git commit -m "feat: add new data for retraining"`
4. `git push` — GitHub Actions handles the rest
