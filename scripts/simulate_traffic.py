import random
import time
import requests
import json
import argparse
import sys
import uuid

# Configuration
API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/inference/predict"

# Baseline profile (healthy loan applications)
def generate_healthy_payload():
    return {
        "age": random.randint(25, 65),
        "income": round(random.uniform(50000.0, 120000.0), 2),
        "credit_score": random.randint(600, 850),
        "loan_amount": round(random.uniform(10000.0, 50000.0), 2),
        "employment_years": random.randint(2, 20),
        "debt_ratio": round(random.uniform(0.1, 0.4), 4),
        "num_accounts": random.randint(2, 8),
        "late_payments": random.randint(0, 2),
        "credit_utilization": round(random.uniform(0.1, 0.5), 4)
    }

# Drift profile (degraded/higher risk loan applications - set extremely to guarantee drift > 0.5)
def generate_drift_payload():
    return {
        "age": 18,
        "income": 1000.0,
        "credit_score": 300,
        "loan_amount": 99000.0,
        "employment_years": 0,
        "debt_ratio": 0.99,
        "num_accounts": 20,
        "late_payments": 15,
        "credit_utilization": 0.99
    }

def run_simulation(duration_sec: int, rps: float, drift_start_pct: float):
    print(f"Starting traffic simulation for {duration_sec}s at ~{rps} req/sec...")
    drift_start_time = time.time() + (duration_sec * drift_start_pct)
    end_time = time.time() + duration_sec
    
    req_count = 0
    err_count = 0
    drift_active = False

    while time.time() < end_time:
        start_ms = time.time()
        
        # Determine payload type
        is_drift = time.time() >= drift_start_time
        if is_drift and not drift_active:
            print("\n\n*** INJECTING CHAOS: CONCEPT DRIFT STARTED ***")
            print("*** Model accuracy will drop. SRE alarms should trigger soon. ***\n")
            drift_active = True
            
        payload = generate_drift_payload() if is_drift else generate_healthy_payload()
        
        # Add random noise to simulate latency spikes during drift
        if is_drift and random.random() < 0.3:
            time.sleep(random.uniform(0.1, 0.5))
            
        try:
            res = requests.post(PREDICT_ENDPOINT, json=payload, timeout=2.0)
            if res.status_code == 200:
                req_count += 1
            else:
                err_count += 1
        except Exception:
            err_count += 1
            
        # Pace the loop
        elapsed = time.time() - start_ms
        target_sleep = (1.0 / rps) - elapsed
        if target_sleep > 0:
            time.sleep(target_sleep)
            
        if req_count % 10 == 0:
            sys.stdout.write(f"\rRequests: {req_count} | Errors: {err_count}")
            sys.stdout.flush()

    print(f"\n\nSimulation complete. Sent {req_count} requests with {err_count} errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Inference Traffic and Concept Drift")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--rps", type=float, default=5.0, help="Requests per second")
    parser.add_argument("--drift-start", type=float, default=0.3, help="When to start drift (0.0 to 1.0)")
    args = parser.parse_args()
    
    run_simulation(args.duration, args.rps, args.drift_start)
