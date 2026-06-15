import requests
import json
import random
import time
import argparse

# Configuration
API_URL = "http://localhost:8000/inference/predict" # Adjust if your endpoint differs (e.g. /inference/predict)
NUM_REQUESTS = 50
DELAY_MS = 200

def get_baseline_sample():
    """Generates a sample from the expected 'normal' distribution."""
    return {
        "feature_1": random.gauss(100.0, 20.0),
        "feature_2": random.gauss(-50.0, 10.0),
        "feature_3": random.gauss(250.0, 30.0),
        "feature_4": random.uniform(0, 1),
        "feature_5": random.gauss(200.0, 15.0)
    }

def get_drifted_sample():
    """Generates a sample with significant statistical variance (Feature Drift)."""
    return {
        # Drastic mean shift and variance explosion
        "feature_1": random.gauss(250.0, 50.0), 
        "feature_2": random.gauss(50.0, 80.0),
        "feature_3": random.gauss(10.0, 100.0),
        "feature_4": random.uniform(5, 10),
        "feature_5": random.gauss(0.0, 50.0)
    }

def simulate_traffic(drift=False):
    print(f"--- Simulating {'DRIFTED' if drift else 'NORMAL'} Traffic ---")
    successes, failures = 0, 0
    
    for i in range(NUM_REQUESTS):
        payload = get_drifted_sample() if drift else get_baseline_sample()
        
        try:
            # We assume your payload goes directly into the predict endpoint
            # It may need to be wrapped in {"features": ... } depending on schema
            response = requests.post(API_URL, json=payload, timeout=2)
            
            if response.status_code == 200:
                successes += 1
                # print(f"[{i+1}/{NUM_REQUESTS}] Success: {response.json()}")
            else:
                failures += 1
                print(f"[{i+1}/{NUM_REQUESTS}] Failed ({response.status_code}): {response.text}")
                
        except Exception as e:
            failures += 1
            # print(f"Request failed: {str(e)}")
            
        time.sleep(DELAY_MS / 1000.0)
        
    print(f"Completed! Success: {successes} | Failures: {failures}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Inference Traffic")
    parser.add_argument("--drift", action="store_true", help="Inject drifted data")
    args = parser.parse_args()
    
    simulate_traffic(drift=args.drift)
