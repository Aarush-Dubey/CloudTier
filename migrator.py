import time
import random
import os
from pymongo import MongoClient, ReturnDocument

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- MongoDB Endpoints (Input & Output) ---
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'icms_db'
METADATA_COLLECTION = 'metadata' # To update the *final* location
JOBS_COLLECTION = 'migration_jobs' # To read and update jobs

# --- Worker Configuration ---
POLL_INTERVAL_SEC = 5 # How long to wait if no jobs are found

# --- Backend Simulation ---
MOCK_BACKEND_LATENCY = {
    "on-prem": 1,
    "private-cloud": 2,
    "public-hot": 3,
    "public-cold": 8,
    "default": 4
}

# ============================================================================
# SERVICE CONNECTIONS
# ============================================================================

def connect_to_mongo():
    """Connects to MongoDB, retrying until successful."""
    while True:
        try:
            print(f"MIGRATOR: Attempting to connect to MongoDB at {MONGO_URI}...")
            db_client = MongoClient(MONGO_URI)
            db = db_client[DB_NAME]
            db.command('ping')
            print("MIGRATOR: MongoDB connected successfully.")
            return db[METADATA_COLLECTION], db[JOBS_COLLECTION]
        except Exception as e:
            print(f"MIGRATOR: Failed to connect to MongoDB: {e}. Retrying in 5 seconds...")
            time.sleep(5)

# ============================================================================
# MAIN WORKER LOGIC
# ============================================================================

def process_one_job(metadata_coll, jobs_coll) -> bool:
    """
    Finds and processes a single PENDING job.
    Returns True if a job was found, False otherwise.
    """
    job = None # Initialize job to None
    try:
        # --- 1. Atomically find and "lock" a job ---
        # We also use ReturnDocument.AFTER to get the *updated* doc back
        # which includes the 'started_at' timestamp we just set.
        job_start_time = time.time()
        job = jobs_coll.find_one_and_update(
            { 'status': 'PENDING' },
            { 
                '$set': { 
                    'status': 'RUNNING',
                    'started_at': job_start_time
                }
            },
            return_document=ReturnDocument.AFTER
        )

        if not job:
            return False # No pending jobs found

        # --- 2. We have a job! Let's process it. ---
        dataset_id = job['dataset_id']
        from_backend = job['from_backend']
        to_backend = job['to_backend']

        print(f"\nMIGRATOR: [JOB {job['_id']}] STARTING: Move {dataset_id} from {from_backend} -> {to_backend} (Reason: {job.get('reason', 'N/A')})")

        # --- 3. Simulate the data migration (the "work") ---
        read_latency = MOCK_BACKEND_LATENCY.get(from_backend, MOCK_BACKEND_LATENCY['default'])
        write_latency = MOCK_BACKEND_LATENCY.get(to_backend, MOCK_BACKEND_LATENCY['default'])
        
        simulated_work_time = read_latency + write_latency + random.uniform(0.5, 2.0)
        
        time.sleep(simulated_work_time)

        # --- 4. Update the "source of truth" ---
        metadata_coll.update_one(
            { 'dataset_id': dataset_id },
            { '$set': { 'current_backend': to_backend } }
        )

        # --- 5. Log the successful migration ---
        job_end_time = time.time()
        duration = job_end_time - job_start_time
        
        jobs_coll.update_one(
            { '_id': job['_id'] },
            { 
                '$set': { 
                    'status': 'COMPLETE',
                    'finished_at': job_end_time,
                    'duration_sec': round(duration, 2)
                }
            }
        )
        
        print(f"MIGRATOR: [JOB {job['_id']}] COMPLETE: {dataset_id} is now in {to_backend}. (Took {duration:.2f}s)")
        return True

    except Exception as e:
        print(f"MIGRATOR: [JOB {job.get('_id', 'UNKNOWN')}] FAILED: {e}")
        # --- 6. Log the failed migration ---
        if job:
            job_end_time = time.time()
            duration = job_end_time - job.get('started_at', job_end_time) # Calculate duration if possible
            
            jobs_coll.update_one(
                { '_id': job['_id'] },
                { 
                    '$set': { 
                        'status': 'FAILED', 
                        'error': str(e),
                        'finished_at': job_end_time,
                        'duration_sec': round(duration, 2)
                    } 
                }
            )
        return True # We still "processed" a job, even if it failed

# ============================================================================
# MAIN WORKER LOOP
# ============================================================================

def run_migrator():
    print("=" * 70)
    print("ICSM Migration Worker (The Muscle)")
    print("=" * 70)
    
    metadata_coll, jobs_coll = connect_to_mongo()
    
    while True:
        if not process_one_job(metadata_coll, jobs_coll):
            # No jobs found, sleep
            print(f"MIGRATOR: No pending jobs. Sleeping for {POLL_INTERVAL_SEC}s...")
            time.sleep(POLL_INTERVAL_SEC)
        
        # If a job *was* processed, immediately check for another.

if __name__ == "__main__":
    run_migrator()