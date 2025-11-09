import time
import numpy as np
import torch
import torch.nn as nn
from pymongo import MongoClient, ReturnDocument

# =---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# --- MongoDB Endpoints ---
MONGO_URI = 'mongodb://localhost:20017/' # Corrected port
DB_NAME = 'icms_db'
METADATA_COLLECTION = 'metadata'
JOBS_COLLECTION = 'migration_jobs'
ANALYSIS_JOBS_COLLECTION = 'analysis_jobs' # <-- NEW: We listen to this

# --- Optimizer Schedule ---
FULL_SCAN_INTERVAL_HOURS = 3
# How long to wait if no jobs are found
POLL_INTERVAL_SEC = 10

# ... (ML Model Config is unchanged) ...
# --- ML Model Config ---
# Model files must be in the same directory
READ_MODEL_PATH = 'reads_model_h8.pth'
WRITE_MODEL_PATH = 'writes_model_h8.pth'
# Shape: (sequence, features)
INPUT_FEATURES = 12
SEQUENCE_LENGTH = 24
HIDDEN_DIM = 64 # Assumed, must match your trained model

# ... (Cost Calculation Config is unchanged) ...
# --- Cost Calculation Config ---
# All prices are in $/GB or $/operation
# P_store is $/GB/Month, so we adjust
HOURS_IN_MONTH = 720.0
PREDICTION_WINDOW_HOURS = 12.0 # Model predicts for 12 hours

# --- NEW: Define the "business cost" of user wait time ---
# This is our Service Level Agreement (SLA) cost.
# How much money do we "lose" for every second a user waits for a read?
# Let's say it's $0.0001 per second of waiting.
SLA_PENALTY_PER_SECOND_OF_WAIT = 0.0001 

PRICING_CONFIG = {
    'on-prem': {
        'P_store': 0.40, # $/GB/Month
        'P_read':  0.00, # $/op
        'P_write': 0.00,
        'P_egress': 0.0,
        'P_ingress': 0.0,
        'P_latency': 0.01 # 10ms read latency
    },
    'private-cloud': {
        'P_store': 0.25,
        'P_read':  0.05,
        'P_write': 0.05,
        'P_egress': 0.01,
        'P_ingress': 0.0,
        'P_latency': 0.03 
    },
    'public-hot': {
        'P_store': 0.20,
        'P_read':  0.04,
        'P_write': 0.04,
        'P_egress': 0.02,
        'P_ingress': 0.0,
        'P_latency': 0.07 
    },
    'public-cold': {
        'P_store': 0.04,
        'P_read':  0.001,
        'P_write': 0.001,
        'P_egress': 0.02,
        'P_ingress': 0.0,
        'P_latency': 4.0  # 4-second read latency
    }
}
ALL_BACKENDS = list(PRICING_CONFIG.keys())
COOLDOWN_READ_THRESHOLD = 10
COOLDOWN_TARGET = 'public-cold'

# ============================================================================
# PYTORCH ML MODEL API
# (This section is unchanged)
# ============================================================================

class ReadWritePredictor(nn.Module):
    """
    Plausible LSTM architecture for your models.
    This *must* match the architecture used to train your .pth files.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(ReadWritePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len=24, features=12)
        lstm_out, _ = self.lstm(x)
        # We only care about the last hidden state
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out

def load_ml_models():
    """Loads the ML models from disk."""
    print("OPTIMIZER: Loading ML models...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Read Model
        read_model = ReadWritePredictor(INPUT_FEATURES, HIDDEN_DIM)
        read_model.load_state_dict(torch.load(READ_MODEL_PATH, map_location=device))
        read_model.to(device)
        read_model.eval()
        
        # Load Write Model
        write_model = ReadWritePredictor(INPUT_FEATURES, HIDDEN_DIM)
        write_model.load_state_dict(torch.load(WRITE_MODEL_PATH, map_location=device))
        write_model.to(device)
        write_model.eval()
        
        print(f"OPTIMIZER: Models loaded successfully on {device}.")
        return read_model, write_model, device
        
    except FileNotFoundError as e:
        print(f"OPTIMIZER: FATAL ERROR: Model file not found: {e}")
        print("OPTIMIZER: Cannot run without models. Exiting.")
        exit(1)
    except Exception as e:
        print(f"OPTIMIZER: FATAL ERROR loading models: {e}")
        exit(1)

# ============================================================================
# DATA PRE-PROCESSING (The 12 Features)
# (This section is unchanged)
# ============================================================================

def classify_temperature_encoded(reads_per_hour: float) -> int:
    """Encodes temperature: cold=0, warm=1, hot=2"""
    if reads_per_hour > 1000: return 2 # hot
    if reads_per_hour > 50: return 1 # warm
    return 0 # cold

def calculate_trend(data, window=6):
    """Calculates linear trend over a window."""
    if len(data) < 2:
        return 0
    x = np.arange(len(data))
    y = np.array(data)
    # Fit a 1st degree polynomial (a line)
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0] # Return the slope

def preprocess_history(history: list, size_gb: float, mode: str = 'reads') -> np.ndarray:
    """
    The "magic" function.
    Converts the simple history from MongoDB into the (24, 12) feature
    matrix required by the ML model.
    """
    
    # Pad history if it's shorter than 24
    if len(history) < SEQUENCE_LENGTH:
        padding = [history[0]] * (SEQUENCE_LENGTH - len(history))
        history = padding + history
    
    # Get all reads/writes as lists
    reads_history = [h['reads_1h'] for h in history]
    writes_history = [h['writes_1h'] for h in history]
    
    feature_matrix = []
    
    for i in range(SEQUENCE_LENGTH): # 0 to 23
        # Get the slice of history up to this point
        hist_slice = history[:i+1]
        reads_slice = reads_history[:i+1]
        writes_slice = writes_history[:i+1]
        
        # Get the current hour's data
        current_hour = hist_slice[-1]
        
        # 1. reads_1h / writes_1h
        if mode == 'reads':
            col0 = current_hour['reads_1h']
        else:
            col0 = current_hour['writes_1h']
        
        # 2. bytes_read_1h
        col1 = current_hour['bytes_read_1h']
        # 3. hour_of_day
        col2 = current_hour['hour_of_day']
        # 4. day_of_week
        col3 = current_hour['day_of_week']
        
        # 5-8. Rolling Aggregations
        if mode == 'reads':
            col4 = np.sum(reads_slice[-3:])
            col5 = np.sum(reads_slice[-6:])
            col6 = np.sum(reads_slice[-12:])
            col7 = np.sum(reads_slice) # Full 24h
        else:
            col4 = np.sum(writes_slice[-3:])
            col5 = np.sum(writes_slice[-6:])
            col6 = np.sum(writes_slice[-12:])
            col7 = np.sum(writes_slice) # Full 24h
        
        # 9. trend_6h
        if mode == 'reads':
            col8 = calculate_trend(reads_slice[-6:])
        else:
            col8 = calculate_trend(writes_slice[-6:])
            
        # 10. size_gb
        col9 = size_gb
        
        # 11. data_temperature_encoded
        col10 = classify_temperature_encoded(current_hour['reads_1h'])
        
        # 12. access_freq_24h
        col11 = np.mean(reads_slice) # Mean reads over 24h
        
        # Add all 12 features for this timestamp
        feature_matrix.append([
            col0, col1, col2, col3, col4, col5,
            col6, col7, col8, col9, col10, col11
        ])
        
    return np.array(feature_matrix, dtype=np.float32)


def get_model_prediction(model, history_tensor, device) -> int:
    """Feeds the tensor to the model and returns an integer prediction."""
    with torch.no_grad():
        # Add batch dimension: (24, 12) -> (1, 24, 12)
        history_tensor = history_tensor.unsqueeze(0).to(device)
        
        prediction = model(history_tensor)
        
        # Ensure positive prediction
        return int(max(0, prediction.item()))

# ============================================================================
# COST ANALYSIS
# (This section is unchanged)
# ============================================================================

def calculate_total_cost(dataset, pred_reads, pred_writes, target_backend):
    """Implements the cost formulas."""
    
    current_backend = dataset['current_backend']
    size_gb = dataset['size_gb']
    
    # Get pricing for the *target* backend
    prices = PRICING_CONFIG[target_backend]
    
    # 1. Host Cost
    # We scale the *predicted* window (12h) to a monthly cost
    # Formula: size_gb * P_store(b) * (12 / 720)
    C_host = size_gb * prices['P_store'] * (PREDICTION_WINDOW_HOURS / HOURS_IN_MONTH)
    
    # 2. Access Cost
    # Formula: pred_reads * P_read(b) + pred_writes * P_write(b)
    C_access = (pred_reads * prices['P_read']) + (pred_writes * prices['P_write'])
    
    # 3. Migration Cost
    C_migrate = 0.0
    if target_backend != current_backend:
        # Get prices for the *current* backend
        current_prices = PRICING_CONFIG[current_backend]
        
        # Formula: size_gb * (P_egress(a) + P_ingress(b)) + 0.5
        P_egress_a = current_prices['P_egress']
        P_ingress_b = prices['P_ingress']
        
        C_migrate = (size_gb * (P_egress_a + P_ingress_b)) + 0.5
        
    # --- 4. NEW: Latency Cost Penalty ---
    # We calculate the total seconds users will wait...
    total_wait_time_sec = pred_reads * prices['P_latency']
    # ...and convert that time into a dollar penalty.
    C_latency_penalty = total_wait_time_sec * SLA_PENALTY_PER_SECOND_OF_WAIT
        
    return C_host + C_access + C_migrate + C_latency_penalty

# ============================================================================
# MIGRATION JOB CREATION
# (This section is unchanged)
# ============================================================================

def create_migration_job(jobs_coll, dataset, new_backend, reason, cost):
    """
    Creates a new migration job in the database,
    using upsert to avoid duplicate PENDING jobs.
    """
    
    dataset_id = dataset['dataset_id']
    current_backend = dataset['current_backend']
    
    # Debounce: Only create a job if one for this dataset isn't already pending
    jobs_coll.update_one(
        {
            'dataset_id': dataset_id,
            'status': 'PENDING'
        },
        { 
            '$set': { 
                'from_backend': current_backend,
                'to_backend': new_backend,
                'created_at': time.time(),
                'reason': reason,
                'predicted_cost_saving': cost
            }
        },
        upsert=True 
    )
    print(f"OPTIMIZER: *** MIGRATION JOB CREATED ({reason}) ***: Move {dataset_id} from {current_backend} to {new_backend}")

# ============================================================================
# REFACTORED OPTIMIZER LOGIC
# ============================================================================

def run_analysis_for_dataset(ds, db, read_model, write_model, device):
    """
    This is the core logic, refactored to run for a SINGLE dataset.
    """
    metadata_coll = db[METADATA_COLLECTION]
    jobs_coll = db[JOBS_COLLECTION]
    
    dataset_id = ds['dataset_id']
    current_backend = ds['current_backend']
    history = ds.get('history', [])
    
    if not history:
        print(f"OPTIMIZER: Skipping {dataset_id} (no history).")
        return
        
    latest_reads = history[-1]['reads_1h']
    
    # --- 1. Handle "Sudden Decrease" (Cooldown) ---
    if latest_reads < COOLDOWN_READ_THRESHOLD and current_backend == 'public-hot':
        create_migration_job(jobs_coll, ds, COOLDOWN_TARGET, "Optimizer (Cooldown)", 0.0)
        return # Done with this dataset
        
    # --- 2. Check for ML-based Optimization ---
    if len(history) < SEQUENCE_LENGTH:
        print(f"OPTIMIZER: Skipping {dataset_id} (not enough data for ML - {len(history)}/24).")
        return
        
    try:
        # --- 3. Pre-process Data ---
        read_tensor_data = preprocess_history(history, ds['size_gb'], mode='reads')
        read_tensor = torch.from_numpy(read_tensor_data)
        
        write_tensor_data = preprocess_history(history, ds['size_gb'], mode='writes')
        write_tensor = torch.from_numpy(write_tensor_data)
        
        # --- 4. Get ML Predictions ---
        pred_reads_12h = get_model_prediction(read_model, read_tensor, device)
        pred_writes_12h = get_model_prediction(write_model, write_tensor, device)

        # --- 5. Run "What-If" Cost Analysis ---
        costs = {}
        for backend in ALL_BACKENDS:
            costs[backend] = calculate_total_cost(ds, pred_reads_12h, pred_writes_12h, backend)
            
        # --- 6. Find Best Backend ---
        best_backend = min(costs, key=costs.get)
        min_cost = costs[best_backend]

        # --- 7. Make Decision & Create Job ---
        if best_backend != current_backend:
            print(f"OPTIMIZER: Analysis for {dataset_id}:")
            print(f"  Current: {current_backend} | Predicted Best: {best_backend} (Cost: {min_cost:.4f})")
            print(f"  PredReads: {pred_reads_12h}, PredWrites: {pred_writes_12h}")
            
            create_migration_job(jobs_coll, ds, best_backend, "Optimizer (Cost)", min_cost)
        
    except Exception as e:
        print(f"OPTIMIZER: FAILED analysis for {dataset_id}: {e}")

# ============================================================================
# NEW 24/7 MAIN LOOP
# ============================================================================

def connect_to_mongo_optimizer():
    """Connects to MongoDB, retrying until successful."""
    while True:
        try:
            print("OPTIMIZER: Attempting to connect to MongoDB...")
            db_client = MongoClient(MONGO_URI)
            db = db_client[DB_NAME]
            db.command('ping')
            print("OPTIMIZER: MongoDB connected.")
            return db
        except Exception as e:
            print(f"OPTIMIZER: Failed to connect to MongoDB: {e}. Retrying in 10s...")
            time.sleep(10)

def process_one_analysis_job(db, read_model, write_model, device) -> bool:
    """
    Finds and processes a single PENDING analysis job.
    """
    analysis_jobs_coll = db[ANALYSIS_JOBS_COLLECTION]
    metadata_coll = db[METADATA_COLLECTION]
    
    # 1. Atomically find and lock a job
    job = analysis_jobs_coll.find_one_and_update(
        { 'status': 'PENDING' },
        { 
            '$set': { 
                'status': 'RUNNING',
                'started_at': time.time()
            }
        },
        return_document=ReturnDocument.AFTER
    )
    
    if not job:
        return False # No jobs to do

    print(f"\nOPTIMIZER: --- Received Real-Time Analysis Job for {job['dataset_id']} (Reason: {job['reason']}) ---")
    
    try:
        # 2. Get the full dataset metadata
        ds = metadata_coll.find_one({'dataset_id': job['dataset_id']})
        
        if not ds:
            raise Exception(f"Dataset {job['dataset_id']} not found in metadata.")
        
        # 3. Run the core analysis logic
        run_analysis_for_dataset(ds, db, read_model, write_model, device)
        
        # 4. Mark job as complete
        analysis_jobs_coll.update_one(
            { '_id': job['_id'] },
            { '$set': { 'status': 'COMPLETE', 'finished_at': time.time() } }
        )
        
    except Exception as e:
        print(f"OPTIMIZER: FAILED processing analysis job {job['_id']}: {e}")
        analysis_jobs_coll.update_one(
            { '_id': job['_id'] },
            { '$set': { 'status': 'FAILED', 'error': str(e) } }
        )
        
    return True # We processed a job

def run_full_scan_job(db, read_model, write_model, device):
    """
    This is the main job that runs every 3 hours.
    """
    print(f"\nOPTIMIZER: --- Running {FULL_SCAN_INTERVAL_HOURS}-Hour Full Scan at {time.ctime()} ---")
    metadata_coll = db[METADATA_COLLECTION]
    
    all_datasets = list(metadata_coll.find())
    print(f"OPTIMIZER: Found {len(all_datasets)} datasets for full scan.")
    
    for ds in all_datasets:
        run_analysis_for_dataset(ds, db, read_model, write_model, device)
            
    print("OPTIMIZER: --- Full Scan Complete ---")

def main():
    print("=" * 70)
    print("ICSM Optimizer (The Brain) - 24/7 Service")
    print("=" * 70)
    
    db = connect_to_mongo_optimizer()
    read_model, write_model, device = load_ml_models()
    
    print(f"OPTIMIZER: Starting 24/7 loop. Full scan every {FULL_SCAN_INTERVAL_HOURS} hours.")
    print(f"OPTIMIZER: Polling for real-time analysis jobs every {POLL_INTERVAL_SEC} seconds.")
    
    last_full_scan_time = 0 # Run immediately
    
    while True:
        # 1. Process one high-priority, real-time job (if one exists)
        # We loop this to clear the queue quickly
        while process_one_analysis_job(db, read_model, write_model, device):
            pass # Keep processing jobs until the queue is empty
        
        # 2. Check if it's time for the 3-hour full scan
        now = time.time()
        if (now - last_full_scan_time) > (FULL_SCAN_INTERVAL_HOURS * 3600):
            try:
                run_full_scan_job(db, read_model, write_model, device)
                last_full_scan_time = now
            except Exception as e:
                print(f"OPTIMIZER: FATAL ERROR during full scan: {e}")
        
        # 3. Wait before polling again
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    main()