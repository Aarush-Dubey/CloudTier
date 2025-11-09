#!/usr/bin/env python3
"""
Real-Time Synthetic Data Producer for ICSM

This script simulates dataset behavior in real-time, sending
hourly metric summaries to a Kafka topic.
"""

import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from kafka import KafkaProducer

# Fixed seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    SIM_CYCLE_DAYS = 30 
    SIM_SPEED_SECONDS = 0.5 # How many real-world seconds to wait between simulating one hour
    NUM_DATASETS = 1000
    BIRTH_RATE_HOURLY = 0.01
    BIRTH_COUNT_RANGE = (2, 5)

    # --- ENDPOINTS ---
    KAFKA_SERVER = os.getenv('KAFKA_SERVER', 'localhost:9092')
    KAFKA_TOPIC = 'access_events' 

    GLOBAL_EVENTS = [
        {"name": "FlashSale", "start_day": 5, "duration_days": 3, "affected_personas": ["business_hours", "viral_content"], "read_multiplier": 5.0},
        {"name": "MonthlyETL", "start_day": 15, "duration_days": 2, "affected_personas": ["batch_processing"], "write_multiplier": 8.0},
        {"name": "ComplianceAudit", "start_day": 20, "duration_days": 1, "affected_personas": ["cold_archive"], "read_multiplier": 30.0},
        {"name": "EndOfMonth", "start_day": 28, "duration_days": 2, "affected_personas": ["batch_processing", "business_hours"], "read_multiplier": 4.0}
    ]

# ============================================================================
# PERSONA TEMPLATES
# ============================================================================

PERSONA_TEMPLATES = {
    "business_hours": {"size_gb_range": (10, 500), "initial_backend": "on-prem", "base_reads_per_hour": 5000, "base_writes_per_hour": 500, "weight": 0.30},
    "batch_processing": {"size_gb_range": (500, 5000), "initial_backend": "private-cloud", "base_reads_per_hour": 800, "base_writes_per_hour": 400, "weight": 0.25},
    "cold_archive": {"size_gb_range": (1000, 50000), "initial_backend": "public-cold", "base_reads_per_hour": 1, "base_writes_per_hour": 0.1, "weight": 0.25},
    "viral_content": {"size_gb_range": (50, 1000), "initial_backend": "public-hot", "base_reads_per_hour": 8000, "base_writes_per_hour": 100, "weight": 0.20}
}

# ============================================================================
# DATASET INITIALIZATION
# ============================================================================

def create_dataset(dataset_id: int, persona: str, creation_hour: int) -> Dict:
    template = PERSONA_TEMPLATES[persona]
    return {"dataset_id": f"ds_{dataset_id:06d}", "persona": persona, "size_gb": np.random.uniform(*template["size_gb_range"]), "creation_timestamp": creation_hour, "current_backend": template["initial_backend"], "base_reads_per_hour": template["base_reads_per_hour"], "base_writes_per_hour": template["base_writes_per_hour"]}

def initialize_datasets(num_datasets: int) -> List[Dict]:
    datasets = []
    personas = list(PERSONA_TEMPLATES.keys())
    weights = [PERSONA_TEMPLATES[p]["weight"] for p in personas]
    for i in range(num_datasets):
        persona = np.random.choice(personas, p=weights)
        datasets.append(create_dataset(i, persona, creation_hour=0))
    return datasets

# ============================================================================
# BEHAVIOR, NOISE, CHURN MODELS
# ============================================================================

def get_persona_multiplier(persona: str, hour_of_day: int, day: int, creation_hour: int) -> float:
    if persona == "business_hours":
        day_of_week = day % 7
        if day_of_week < 5 and 9 <= hour_of_day < 17: return 1.0
        else: return 0.15
    elif persona == "batch_processing":
        return np.exp(-((hour_of_day - 2) ** 2) / (2 * 2 ** 2))
    elif persona == "cold_archive":
        return 1.0 if np.random.random() < 0.002 else 0.01
    elif persona == "viral_content":
        current_hour = day * 24 + hour_of_day
        hours_since_creation = current_hour - creation_hour
        half_life = np.random.uniform(36, 72)
        return max(0.01, np.exp(-0.693 * hours_since_creation / half_life))
    return 1.0

def apply_noise(base_value: float) -> float:
    value = base_value * np.random.uniform(0.75, 1.25)
    value += np.random.normal(0, base_value * 0.08)
    if np.random.random() < 0.02: return 0
    if np.random.random() < 0.03: value *= np.random.uniform(2, 5)
    return max(0, value)

def apply_drift(base_value: float, day: int) -> float:
    return base_value * (1 + (0.0015 * day))

def get_event_multiplier(persona: str, day: int) -> Tuple[float, float]:
    read_mult, write_mult = 1.0, 1.0
    for event in Config.GLOBAL_EVENTS:
        start, end = event["start_day"], event["start_day"] + event["duration_days"]
        if (start <= day < end) and (persona in event.get("affected_personas", [])):
            read_mult *= event.get("read_multiplier", 1.0)
            write_mult *= event.get("write_multiplier", 1.0)
    return read_mult, write_mult

def simulate_churn(datasets: List[Dict], current_hour: int, next_dataset_id: int) -> Tuple[List[Dict], int]:
    if np.random.random() < Config.BIRTH_RATE_HOURLY:
        num_new = np.random.randint(*Config.BIRTH_COUNT_RANGE)
        personas, weights = list(PERSONA_TEMPLATES.keys()), [0.30, 0.15, 0.10, 0.45]
        for _ in range(num_new):
            persona = np.random.choice(personas, p=weights)
            datasets.append(create_dataset(next_dataset_id, persona, current_hour))
            next_dataset_id += 1
    return datasets, next_dataset_id

# ============================================================================
# KAFKA PRODUCER MAIN LOOP
# ============================================================================

def create_kafka_producer() -> KafkaProducer:
    """Connects to Kafka, retrying until successful."""
    while True:
        try:
            print(f"PRODUCER: Attempting to connect to Kafka at {Config.KAFKA_SERVER}...")
            producer = KafkaProducer(
                bootstrap_servers=Config.KAFKA_SERVER,
                key_serializer=lambda k: k.encode('utf-8'), # Encodes the partition key
                value_serializer=lambda v: json.dumps(v).encode('utf-8') # Encodes the message
            )
            print("PRODUCER: Kafka Producer connected successfully.")
            return producer
        except Exception as e:
            print(f"PRODUCER: Failed to connect to Kafka: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def run_producer():
    print("=" * 70)
    print("ICSM Real-Time Data Producer")
    print("=" * 70)
    
    producer = create_kafka_producer()
    
    print(f"PRODUCER: Initializing {Config.NUM_DATASETS} datasets...")
    current_datasets = initialize_datasets(Config.NUM_DATASETS)
    next_dataset_id = Config.NUM_DATASETS
    print("PRODUCER: Starting real-time simulation loop...")
    
    current_hour_idx = 0
    start_time = datetime(2025, 1, 1)

    while True:
        day = (current_hour_idx // 24) % Config.SIM_CYCLE_DAYS
        hour_of_day = current_hour_idx % 24
        sim_timestamp = start_time + timedelta(hours=current_hour_idx)
        
        current_datasets, next_dataset_id = simulate_churn(
            current_datasets, current_hour_idx, next_dataset_id
        )
        
        hour_start_time = time.time()
        
        for ds in current_datasets:
            persona_mult = get_persona_multiplier(ds["persona"], hour_of_day, day, ds["creation_timestamp"])
            read_event_mult, write_event_mult = get_event_multiplier(ds["persona"], day)
            base_reads = apply_drift(ds["base_reads_per_hour"], day)
            base_writes = apply_drift(ds["base_writes_per_hour"], day)
            reads_1h = apply_noise(base_reads * persona_mult * read_event_mult)
            writes_1h = apply_noise(base_writes * persona_mult * write_event_mult)
            bytes_read_1h = reads_1h * np.random.lognormal(10, 2) * 1024
            
            event_payload = {
                "timestamp": int(sim_timestamp.timestamp()),
                "dataset_id": ds["dataset_id"],
                "reads_1h": int(reads_1h),
                "writes_1h": int(writes_1h),
                "bytes_read_1h": int(bytes_read_1h),
                "hour_of_day": hour_of_day,
                "day_of_week": day % 7,
                "current_backend": ds["current_backend"], 
                "size_gb": round(ds["size_gb"], 2)
            }
            
            producer.send(
                Config.KAFKA_TOPIC,
                key=event_payload["dataset_id"],
                value=event_payload
            )

        producer.flush()
        hour_exec_time = time.time() - hour_start_time
        
        print(f"PRODUCER: Simulated Hour {current_hour_idx}. Sent {len(current_datasets)} metrics.")
        
        current_hour_idx += 1
        sleep_time = Config.SIM_SPEED_SECONDS - hour_exec_time
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        run_producer()
    except KeyboardInterrupt:
        print("\nPRODUCER: Stopped.")