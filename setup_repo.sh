#!/bin/bash
# COS40007 Group Task 3 - Repo Setup Script
# Run this inside your cloned group repo folder

echo "Setting up Group Task 3 repo structure..."

# ── GitHub Actions ──
mkdir -p .github/workflows
cat > .github/workflows/retrain-on-push.yml << 'EOF'
# Placeholder - Ishank to complete
name: Retrain on Data Push

on:
  push:
    branches: [ main ]
    paths:
      - 'data/**'
  schedule:
    - cron: '0 0 * * 0'   # weekly on Sunday midnight
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      # TODO: Ishank to add full pipeline steps here
EOF

# ── Source scripts ──
mkdir -p src
touch src/model.py src/evaluate.py src/monitor.py src/preprocess_new_data.py

cat > src/model.py << 'EOF'
# model.py - Training script
# TODO: Ratul to implement
EOF

cat > src/evaluate.py << 'EOF'
# evaluate.py - Evaluation script
# TODO: Ratul to implement
EOF

cat > src/monitor.py << 'EOF'
# monitor.py - Drift & performance monitoring
# TODO: Ratul to implement
EOF

cat > src/preprocess_new_data.py << 'EOF'
# preprocess_new_data.py - Preprocess new data
# TODO: Ratul to implement
EOF

# ── Data folders ──
mkdir -p data train test
touch data/.gitkeep train/.gitkeep test/.gitkeep

# ── Monitoring ──
mkdir -p monitoring/reports monitoring/logs monitoring/alerts
touch monitoring/reports/.gitkeep monitoring/logs/.gitkeep monitoring/alerts/.gitkeep

# ── Artifacts ──
mkdir -p artifacts/models artifacts/data artifacts/preprocessing artifacts/metrics artifacts/metadata
touch artifacts/models/.gitkeep artifacts/data/.gitkeep artifacts/preprocessing/.gitkeep
touch artifacts/metrics/.gitkeep artifacts/metadata/.gitkeep

# ── Logs & Reports ──
mkdir -p logs reports
touch logs/.gitkeep reports/.gitkeep

# ── DVC ──
cat > dvc.yaml << 'EOF'
# dvc.yaml - DVC pipeline
# TODO: Ishank to complete
stages:
  preprocess:
    cmd: python src/preprocess_new_data.py
    deps:
      - src/preprocess_new_data.py
      - data/
    outs:
      - artifacts/data/

  train:
    cmd: python src/model.py
    deps:
      - src/model.py
      - artifacts/data/
    outs:
      - artifacts/models/

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - artifacts/models/
    outs:
      - artifacts/metrics/

  monitor:
    cmd: python src/monitor.py
    deps:
      - src/monitor.py
      - artifacts/metrics/
    outs:
      - reports/
EOF

# ── Requirements ──
cat > requirements.txt << 'EOF'
tensorflow
numpy
pandas
scikit-learn
matplotlib
dvc
dvc-gs
joblib
scipy
EOF

# ── .gitignore ──
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*.egg-info/
.env
.venv
venv/
*.h5
*.keras
*.pkl
*.npy
/artifacts/models/
/artifacts/data/
.dvc/cache
.dvc/tmp
EOF

# ── .dvcignore ──
cat > .dvcignore << 'EOF'
.git
__pycache__
*.pyc
EOF

# ── README ──
cat > README.md << 'EOF'
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
EOF

echo ""
echo "✅ Structure created. Now run:"
echo ""
echo "  git add ."
echo "  git commit -m 'feat: initial repo structure for MLOps pipeline'"
echo "  git push"
