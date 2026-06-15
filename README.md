# Autonomous ML Platform 🚀

An implementation-complete, self-healing MLOps platform designed to monitor production models, detect data drift, leverage LLMs to formulate retraining strategies, evaluate candidate models under strict policy guardrails, and promote them automatically.

---

## 🏛️ System Architecture

The platform runs a continuous self-healing loop:

```
[ Inference Traffic ] ──► [ Inference Logs ]
                                  │
                                  ▼
[ Target Registry ]       [ Monitoring Scheduler ]
        │                         │
        ▼ (computes drift)        ▼
[ LLM Strategy Engine ] ◄── [ Drift Alarm Signal ]
        │
        ▼ (formulates retraining configuration)
[ Policy Guardrails ] ──► [ Automation Executor ]
        │                         │
        ▼ (if approved)           ▼
[ Training Runner ]       [ Model Evaluator ] (compares candidates)
        │                         │
        └─────────────────────────┴──► [ Auto-Promotion / Versioning ]
```

### Core Components
1. **Dataset Registry**: Stores dataset metadata, profiling metrics, and baseline statistics.
2. **Model Registry**: Manages model endpoints, versions, lineage (`parent_version`), and operational status (`active`, `staging`).
3. **Inference Layer**: Serves model endpoints with in-memory TTL caching and logs input/prediction features to PostgreSQL.
4. **Monitoring Engine**: Triggers every 5 minutes to calculate drift scores (PSI and feature shift) and alert on threshold violations.
5. **LLM Decision Engine**: Interfaces with Ollama (`llama3`) to analyze drift and formulate structured retraining strategies.
6. **Policy & Guardrails**: Evaluates decisions against a 7-checkpoint policy (confidences, cost caps, GPU allocations, time freeze windows, retraining rate limits, production blocks, and severity filters).
7. **Training Runner & Orchestrator**: In-process or Docker-based training pipelines supporting HPO (Optuna), data rebalancing, feature selection, threshold optimization, and ensembling. Fully supports **13 algorithms** out-of-the-box:
   * *Classification*: RandomForest, GradientBoosting, AdaBoost, ExtraTrees, LogisticRegression, DecisionTree, SVC, KNN, XGBoost, and LightGBM.
   * *Regression*: LinearRegression, Ridge, Lasso, and regression-only variants of the above tree/ensemble models.
8. **Model Evaluator**: Compares newly trained candidate model metrics against the current active baseline to determine promotion/rollback safety.
9. **Observability**: Exposes real-time Prometheus metrics at `/metrics` (decisions, violations, latency histograms, and counts).
10. **Frontend**: React-based dashboard visualizing models, training runs, monitoring metrics, drift trends, and policy approvals.

---

## 🛠️ Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop)
* [Node.js v20+](https://nodejs.org/) (only if running frontend locally without Docker)
* [Python 3.12+](https://www.python.org/) (only if running backend locally without Docker)

---

## ⚡ Quickstart (Docker Compose)

The easiest way to run the entire system is via Docker Compose. This starts PostgreSQL, MinIO, Ollama (with `llama3` automatically pre-pulled), the FastAPI Backend, and the React Frontend.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/autonomous-ml-platform.git
   cd autonomous-ml-platform
   ```

2. **Launch the Services**:
   ```bash
   docker compose up --build
   ```

3. **Access the Applications**:
   * **React Dashboard**: [http://localhost:3000](http://localhost:3000)
   * **FastAPI Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   * **MinIO Object Console**: [http://localhost:9001](http://localhost:9001) (User/Password: `minioadmin` / `minioadmin`)
   * **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 🔧 Local Development Setup

If you wish to run the backend and frontend separately for development:

### 1. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment variables template and configure it:
   ```bash
   cp .env.example .env
   ```
5. Run the FastAPI application:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

### 2. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../automl-frontend
   ```
2. Install Node modules:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🚶 Walkthrough Guide

To test the self-healing retraining loop:

### Step 1: Upload a Dataset
1. Go to the **Dataset** tab in the dashboard.
2. Upload the sample dataset provided in `data/loan_approval_test.csv`.
3. The platform will automatically profile the dataset, calculate baseline distributions, and save it in MinIO.

### Step 2: Register a Model
1. Go to the **Models** tab and register a new model (e.g., name `loan_predictor`).
2. Select your desired algorithm (e.g., `LightGBM` or `RandomForestClassifier`) from the unified algorithm list.
3. Upon registration, the backend automatically triggers **baseline training (v1)** on the uploaded dataset, computes actual performance metrics, and sets the model status to **Active**.

### Step 3: Simulate Traffic and Introduce Drift
1. Run the traffic simulator script to send normal requests to the active model endpoint:
   ```bash
   python scripts/simulate_traffic.py --duration 120 --rps 5.0 --drift-start 0.3
   ```
   *(This starts healthy, then injects out-of-distribution high-risk candidates to trigger feature drift).*

### Step 4: Observe Self-Healing
1. The **Monitoring Engine** scheduler (runs every 5 mins) will pick up the drifted logs.
2. The drift score will exceed the threshold (`0.2`).
3. The **LLM Decision Engine** generates a retraining configuration.
4. The **Policy Guardrails** checks the confidence and daily costs. If approved, the **Automation Executor** starts retraining.
5. Once a candidate wins (e.g., v2 outperforms the degraded baseline), it is promoted to **Active** production status automatically!

---

## 🧪 Automated Verification Suite

A comprehensive verification suite is included to validate the core ML training and tuning pipeline in isolation (bypassing the database & MinIO):

* **Run Verification**:
  ```powershell
  $env:DATABASE_URL="postgresql+asyncpg://dummy:dummy@localhost:5432/dummy"; $env:MINIO_ENDPOINT="localhost:9000"; $env:MINIO_ACCESS_KEY="dummy"; $env:MINIO_SECRET_KEY="dummy"; $env:MINIO_BUCKET="dummy"; $env:PYTHONIOENCODING="utf-8"; .\venv\Scripts\python.exe verify_algorithms.py
  ```
* **Coverage**: Runs 75 test cases verifying case-insensitive alias mapping, training of all 10 classifiers and 12 regressors, model selector leaderboards, Optuna trial objectives, and parameter isolation during fallback states.
