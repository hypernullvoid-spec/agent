# sandbox/Dockerfile
#
# Optional: build a custom sandbox image that pre-installs common data
# science packages so the agent doesn't need to `pip install` them at
# runtime. Especially useful for Phases 6–10 (the ML pipeline).
#
# Build:   docker build -t neo-sandbox ./sandbox
# Then in agent/sandbox.py, change SANDBOX_IMAGE = "neo-sandbox"
#
# If you skip this, the agent falls back to python:3.11-slim and installs
# packages on demand with install_package() — which works fine but is
# slower on first use.

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Common data science + ML stack
# Versions are unpinned intentionally — the agent may install newer ones.
RUN pip install --no-cache-dir \
        numpy pandas polars \
        matplotlib seaborn plotly \
        scikit-learn xgboost lightgbm \
        requests httpx \
        jupytext \
    && pip cache purge

WORKDIR /workspace
