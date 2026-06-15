import requests
import json
import random
import time

# Configuration
API_URL = "http://localhost:8000/inference/predict" 
NUM_REQUESTS = 30
DELAY = 0.5 

def get_concept_drift_sample():
    """
    Concept drift simulation.
    The features look normal (Feature 1 and 2), but the unseen underlying relationship has shifted.
    We simulate this by pushing inputs that *used* to mean Class 0, but in reality,
    market conditions mean they should now be Class 1.
    While the monitoring API tracks feature drift, concept drift often requires actual labels (ground truth)
    to detect a drop in accuracy. 
    """
    return {
        # Normal feature space
        "feature_1": random.gauss(100.0, 20.0),
        "feature_2": random.gauss(-50.0, 10.0),
        "feature_3": random.gauss(250.0, 30.0),
        "feature_4": random.uniform(0, 1),
        "feature_5": random.gauss(200.0, 15.0),
        "__simulation_note": "In a real concept drift scenario, ground_truth shifted."
    }

print("--- Injecting Concept Drift Pattern ---")
for i in range(NUM_REQUESTS):
    payload = get_concept_drift_sample()
    try:
        res = requests.post(API_URL, json=payload, timeout=2)
        print(f"REQ {i}: {res.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(DELAY)
