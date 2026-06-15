# Autonomous ML Lifecycle Platform: Demo Execution Plan

This document serves as your definitive guide for a 7–10 minute jury presentation, optimized for technical storytelling, flawless execution, and handling high-stakes Q&A.

---

## 1️⃣ Demo Environment Preparation

### Required Services
Ensure the following are running before the demo starts:
- **PostgreSQL**: Stores metadata, logs, registries.
- **MinIO**: S3-compatible object storage for model artifacts.
- **Ollama**: Local LLM engine for policy decisions.
- **FastAPI Backend**: Core orchestration engine.
- **React Frontend**: The main control panel.

### Environment Variables (`backend/.env`)
```env
DATABASE_URL=postgresql+asyncpg://ml_user:ml_password@localhost:5432/ml_platform
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=ml-artifacts
OLLAMA_BASE_URL=http://localhost:11434
```

### Seed Dataset
We will use a binary classification dataset (e.g., Telecom Customer Churn or Student Dropout Risk) with at least 1000+ rows. 

### Startup Commands
Execute these in separate terminal windows 10 minutes before the pitch:
1. **Start Core Services & Backend** (from project root):
   ```bash
   docker-compose up -d
   ```
2. **Start Frontend** (from `automl-frontend`):
   ```bash
   npm run dev
   ```

### Pre-Flight Checklist
- **Backend Health**: `curl http://localhost:8000/api/v1/system/health` → Should return `{"status":"healthy"}`
- **Ollama**: `curl http://localhost:11434/api/tags` → Should return a JSON with downloaded LLM models (e.g., `llama3`).
- **MinIO**: Navigate to `http://localhost:9001` with `minioadmin`/`minioadmin` to ensure bucket `ml-artifacts` exists.
- **Frontend**: Open `http://localhost:5173`. Ensure no errors in the browser console.

---

## 2️⃣ Demo Flow – Step-by-Step

### STEP 1 — Dataset Upload & Baseline
* **What to show**: The Dataset view. Upload the historical CSV.
* **What to say**: "Our journey begins with data. We upload our initial baseline dataset. Immediately, the system profiles this data, locking in statistical distributions that act as our golden standard for future drift detection."

### STEP 2 — Training Phase A (Initial Model)
* **What happens internally**: Backend kicks off Scikit-learn training, computes baseline metrics (Accuracy, F1), and pushes artifacts to MinIO.
* **What to highlight**: Point to the Training Progress indicator.
* **What to say**: "We trigger our Phase A pipeline to train a baseline XGBoost model. The system automatically engineers features, trains, evaluates, and stores the model securely in our local S3-compatible vault."

### STEP 3 — Model Registry & KPI
* **What to click**: Navigate to "Model Registry". Expand the newly trained model version (v1).
* **What metric to point out**: Highlight the precision/recall or accuracy metric. 
* **What to say**: "Here is our registry. Version 1 is marked 'active'. It achieved 92% accuracy on the test set. Notice the complete lineage—we know exactly which dataset and hyperparameters produced this artifact."

### STEP 4 — Deployment & Hot Reload
* **What to click**: Navigate to the Inference page. 
* **What to say**: "Serving ML models usually requires downtime. We use a dynamic router that allows instant hot-reloading. The endpoint is live immediately at `/inference/predict` without dropping a single packet."

### STEP 5 — Inference Sandbox
* **What to click**: Use the Inference sandbox UI to send 5 requests.
* **What to show**: Show the live request logs streaming in at the bottom of the page.
* **What to say**: "Let's send some live traffic. As predictions flow through, our edge monitoring captures every input feature and output probability asynchronously."

### STEP 6 — Inject Drift
* **What to click**: Open your terminal. Run the drift simulation script (see section 3).
* **What to say**: "In the real world, data changes. Let's simulate a sudden drop in input data quality—what we call 'Feature Drift'—by injecting high-variance, out-of-distribution traffic into the inference endpoint."

### STEP 7 — Monitoring Reaction
* **What to click**: Switch to the "Monitoring" tab. 
* **What metrics change**: Point out the "Global Drift" score spiking above 0.20 and the warning UI turning red.
* **What to say**: "Instantly, our statistical monitors detect the anomaly. The feature distribution has diverged significantly from the Phase A baseline. The drift score crosses our critical 0.20 threshold."

### STEP 8 — LLM Decision Engine
* **What to click**: Navigate to the "Automation Engine" log. Click on the latest "DRIFT_ALERT" trace.
* **What to show**: The JSON output from the LLM, showing `"reasoning"` and `"confidence"`.
* **What to say**: "This is where we go from reactive to autonomous. The system feeds the statistical anomaly to our local deterministic LLM. The LLM acts as an AI SRE, analyzes the context, and concludes with 92% confidence that a retrain is required to heal the system."

### STEP 9 — Policy Engine Guardrails
* **What to click**: Move to "Policy Engine" or show the "Approval" status.
* **What to say**: "We don't just let LLMs execute arbitrary code. The LLM's decision is intercept-checked by our deterministic Policy Engine, which verifies we haven't exceeded budget or compute constraints before approving."

### STEP 10 — Autonomous Retraining (Phase C)
* **What to click**: Switch back to Training/Jobs to show the autonomous job running.
* **What to say**: "The system auto-triggers Phase C: a self-healing retraining loop using the newly collected, drifted data combined with historical baselines to recover performance."

### STEP 11 — Model Promotion & Comparison
* **What to click**: Show Registry (v2 vs v1).
* **What to say**: "Version 2 is ready. Our shadow evaluation proves it recovers accuracy from the degraded state. Because it outperforms v1 on the new distribution, it is marked for promotion."

### STEP 12 — Hot Redeploy
* **What to click**: Switch the active version to v2 in the UI, then send an inference request.
* **What to say**: "With a single click—or fully automated if policies allow—we hot-swap to Version 2. The system has successfully healed itself from data drift with zero downtime."

---

## 3️⃣ Drift Simulation Scripts

### Option A: Feature Drift Python Script (`simulate_feature_drift.py`)
See the accompanying generated Python file. This script injects continuous high-variance data into your endpoint.

### Option B: Quick cURL API Injection
```bash
# Baseline request (Normal distribution)
curl -X POST http://localhost:8000/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 1.2, "feature_2": 3.4, "feature_3": 0.5}'

# Drift request (Anomalous distribution)
curl -X POST http://localhost:8000/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 99.9, "feature_2": -45.0, "feature_3": 8.0}'
```

---

## 4️⃣ Failure-Proof Backup Plan

1. **If training takes too long**:
   * *Fallback*: Prepare a pre-trained `v2` model artifact already stored in MinIO. 
   * *Script*: "In the interest of time, I’ll fast-forward. The system has already completed this exact job under ID 4829..."
2. **If Ollama (LLM) response delays**:
   * *Fallback*: Have the LLM's JSON response hardcoded as a fallback in your `AutomationService` if the local LLM times out after 3 seconds.
3. **If Drift Detection doesn't trigger visually**:
   * *Fallback*: Have a second browser tab pre-opened to a state where drift *did* trigger (a cached view or an earlier session).
4. **Pre-Seeded Log Strategy**:
   * Always run a full mock-cycle 10 minutes before the jury enters, so the database has rich history charts, even if the live demo hiccups.

---

## 5️⃣ Demo Narrative Script (Speech)

*(Confident, poised, moderate pace.)*

"Good morning. Today, machine learning systems fail not because models are bad, but because the world changes. Data drifts. Models degrade. And engineering teams spend endless hours manually debugging pipelines. We built the **Autonomous ML Lifecycle Platform** to stop this.

What you are looking at is not just an MLOps tool; it is a closed-loop, self-healing infrastructure.

*(Upload Dataset & Train v1)*
We start by establishing a baseline. The engine automatically trains an initial model and locks down its statistical fingerprints.

*(Switch to Inference & Inject Drift)*
Our endpoint is live, serving predictions with zero downtime. But watch what happens when we simulate a sudden market shock—injecting highly anomalous data. 

*(Switch to Monitoring)*
Instantly, edge monitors detect the anomaly. The distribution has diverged. Most systems just send a dumb Slack alert and wait for an engineer. We don't.

*(Show Automation Log)*
This statistical signal is passed to a localized, deterministic LLM acting as an AI Site Reliability Engineer. The LLM analyzes the context and definitively recommends a self-healing retrain. 

*(Show Policy & Registry)*
Before execution, our deterministic guardrails ensure this action is safe, within budget, and compliant. The system autonomously trains Version 2 on the new data, shadow-evaluates it, proves it outperforms the degraded model, and seamlessly hot-swaps the endpoint. 

In less than 3 minutes, our platform detected a failure, diagnosed it with AI, and healed itself with zero human intervention. This is the enterprise-scalable future of ML."

---

## 6️⃣ Visual Highlights for the Jury

- **Most Impressive UI**: The **Automation Log / Decision Engine View**. Showing the LLM's thought process (Reasoning + Confidence Score) mapped over a real MLOps problem always wins crowds.
- **When to pause**: Pause on the JSON output of the LLM. Let them read `"action": "RETRAIN"`.
- **When to scroll slowly**: On the "Model Registry", comparing the metrics side-by-side.
- **What to circle/point at**:
   1. The `Drift Score > 0.20` threshold turning red.
   2. The `0ms` downtime latency indicator during a model hot-swap.

---

## 7️⃣ Time-Optimized Versions

### 3-Minute Version (Lightning Pitch)
- Show Inference UI.
- Inject drift via script.
- Show Monitoring turn red.
- Jump straight to Automation Log showing LLM decision.
- Show v2 automatically appearing in Registry. "It healed itself."

### 5-Minute Version (Standard Pitch)
- Upload data -> Train v1.
- Show Inference.
- Inject drift.
- Review LLM reasoning JSON.
- Manually trigger the hot-swap to emphasize control.

### 10-Minute Version
- *Includes the full Step 1-12 flow above.*

---

## 8️⃣ High-Risk Jury Questions & Bulletproof Answers

**1. Q: What happens if the LLM hallucinates an unnecessary retrain?**
**A:** "We use a two-tiered architecture. The LLM is strictly an analytical engine; it cannot execute tasks. Its JSON output must pass through a deterministic Policy Engine written in standard code. If the LLM proposes a retrain but the drift score is mathematically below the critical threshold, the policy engine blocks the request." 

**2. Q: How do you handle zero-downtime deployments with database schema changes?**
**A:** "Model artifacts and metadata schemas are decoupled. The inferencing router dynamically loads the Pickle/ONNX artifact from MinIO entirely in-memory using an asynchronous pointer swap, totally independent of the Postgres schema. No server restarts occur."

**3. Q: Why use a local LLM instead of GPT-4?**
**A:** "Data privacy and latency. Enterprise ML data often contains PII or proprietary features. By utilizing a quantized local Llama3 via Ollama, we ensure zero data egress and maintain sub-second context-processing latency."

**4. Q: How exactly is drift calculated?**
**A:** "We utilize statistical measures depending on the data type—specifically measuring the coefficient of variation against the locked baseline mean for continuous variables, and categorical distribution shifts for discrete ones. It calculates a p-value equivalent proxy."

**5. Q: What if retraining on the drifted data introduces bias?**
**A:** "Phase C retraining does not discard old data; it blends the new anomalous distribution with historical uniform samples to prevent catastrophic forgetting. Furthermore, the new model must explicitly pass a shadow-evaluation threshold before it is permitted to be hot-swapped."

**6. Q: How scalable is this architecture?**
**A:** "Highly. It relies on async FastAPI for non-blocking I/O, PostgreSQL for ACID compliance on metadata, and MinIO for highly concurrent artifact retrieval. The actual training nodes can be dynamically spun up as temporary containers in a K8s integration."

**7. Q: Is the system vulnerable to adversarial data injection?**
**A:** "The Policy Engine implements rate-limiting and budget caps. An adversarial attack flooding the endpoint might trigger drift, but it cannot trigger infinite retraining loops because the engine limits actions to e.g., one retrain per 24 hours."

**8. Q: How do you manage concurrent model versions?**
**A:** "The Model Registry acts as a state machine. Every model is immutable once trained. Transitions from 'archived' to 'active' require satisfying specific relational constraints in the backend."

**9. Q: Why not just automate based on standard if/else thresholds? Why the LLM?**
**A:** "Simple thresholds suffer from alert fatigue. If CPU spikes and drift spikes simultaneously, hardcoded logic fails to see context. The LLM acts as an event correlator—it looks at latency, drift, and time-of-day to deduce whether it's an anomaly or a natural traffic surge, fundamentally reducing false positives."

**10. Q: What happens if MinIO goes down?**
**A:** "The inference endpoint caches the active model in RAM. If MinIO drops, current predictions continue uninterrupted. Automation will pause gracefully, logging an infrastructure-layer exception rather than crashing the system."
