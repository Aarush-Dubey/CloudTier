from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import time
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Configuration
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'icms_db'

# Pricing config (must match optimizer.py)
PRICING_CONFIG = {
    'on-prem': {'P_store': 0.40, 'P_read': 0.00, 'P_write': 0.00, 'P_latency': 0.01},
    'private-cloud': {'P_store': 0.25, 'P_read': 0.05, 'P_write': 0.05, 'P_latency': 0.03},
    'public-hot': {'P_store': 0.20, 'P_read': 0.04, 'P_write': 0.04, 'P_latency': 0.07},
    'public-cold': {'P_store': 0.04, 'P_read': 0.001, 'P_write': 0.001, 'P_latency': 4.0}
}

HOURS_IN_MONTH = 720.0
SLA_PENALTY_PER_SECOND = 0.0001

# MongoDB connection
def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# ============================================================================
# COST CALCULATION FUNCTIONS
# ============================================================================

def calculate_baseline_cost(datasets):
    """Calculate cost if all datasets stayed in their initial backend"""
    total_cost = 0
    for ds in datasets:
        backend = ds.get('current_backend', 'on-prem')
        size_gb = ds.get('size_gb', 0)
        history = ds.get('history', [])
        
        if not history:
            continue
            
        # Get last 24 hours of activity
        recent_history = history[-24:]
        total_reads = sum(h.get('reads_1h', 0) for h in recent_history)
        total_writes = sum(h.get('writes_1h', 0) for h in recent_history)
        
        prices = PRICING_CONFIG[backend]
        
        # Storage cost (24 hours)
        storage_cost = size_gb * prices['P_store'] * (24 / HOURS_IN_MONTH)
        
        # Access cost
        access_cost = (total_reads * prices['P_read']) + (total_writes * prices['P_write'])
        
        # Latency penalty
        latency_cost = total_reads * prices['P_latency'] * SLA_PENALTY_PER_SECOND
        
        total_cost += storage_cost + access_cost + latency_cost
    
    return total_cost

def calculate_optimized_cost(datasets, migrations):
    """Calculate actual cost with optimizations"""
    total_cost = 0
    migration_costs = 0
    
    for ds in datasets:
        current_backend = ds.get('current_backend', 'on-prem')
        size_gb = ds.get('size_gb', 0)
        history = ds.get('history', [])
        
        if not history:
            continue
            
        recent_history = history[-24:]
        total_reads = sum(h.get('reads_1h', 0) for h in recent_history)
        total_writes = sum(h.get('writes_1h', 0) for h in recent_history)
        
        prices = PRICING_CONFIG[current_backend]
        
        # Storage cost
        storage_cost = size_gb * prices['P_store'] * (24 / HOURS_IN_MONTH)
        
        # Access cost
        access_cost = (total_reads * prices['P_read']) + (total_writes * prices['P_write'])
        
        # Latency penalty
        latency_cost = total_reads * prices['P_latency'] * SLA_PENALTY_PER_SECOND
        
        total_cost += storage_cost + access_cost + latency_cost
    
    # Add migration costs
    for mig in migrations:
        if mig.get('status') == 'COMPLETE':
            migration_costs += 0.5  # Base migration cost
    
    return total_cost + migration_costs, migration_costs

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/overview')
def get_overview():
    db = get_db()
    metadata = db['metadata']
    jobs = db['migration_jobs']
    
    total_datasets = metadata.count_documents({})
    
    # Backend distribution
    pipeline = [
        {'$group': {'_id': '$current_backend', 'count': {'$sum': 1}}}
    ]
    backend_dist = list(metadata.aggregate(pipeline))
    
    # Migration stats
    total_migrations = jobs.count_documents({})
    pending_migrations = jobs.count_documents({'status': 'PENDING'})
    running_migrations = jobs.count_documents({'status': 'RUNNING'})
    completed_migrations = jobs.count_documents({'status': 'COMPLETE'})
    failed_migrations = jobs.count_documents({'status': 'FAILED'})
    
    # Cost analysis
    all_datasets = list(metadata.find())
    all_migrations = list(jobs.find())
    
    baseline_cost = calculate_baseline_cost(all_datasets)
    optimized_cost, migration_cost = calculate_optimized_cost(all_datasets, all_migrations)
    savings = baseline_cost - optimized_cost
    savings_percent = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    return jsonify({
        'total_datasets': total_datasets,
        'backend_distribution': backend_dist,
        'migrations': {
            'total': total_migrations,
            'pending': pending_migrations,
            'running': running_migrations,
            'completed': completed_migrations,
            'failed': failed_migrations
        },
        'costs': {
            'baseline': round(baseline_cost, 2),
            'optimized': round(optimized_cost, 2),
            'migration_cost': round(migration_cost, 2),
            'savings': round(savings, 2),
            'savings_percent': round(savings_percent, 2)
        }
    })

@app.route('/api/dataset/<dataset_id>')
def get_dataset_detail(dataset_id):
    db = get_db()
    metadata = db['metadata']
    
    dataset = metadata.find_one({'dataset_id': dataset_id})
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    # Convert ObjectId to string
    dataset['_id'] = str(dataset['_id'])
    
    # Calculate metrics
    history = dataset.get('history', [])
    if history:
        recent = history[-24:]
        dataset['metrics'] = {
            'avg_reads_24h': np.mean([h['reads_1h'] for h in recent]),
            'avg_writes_24h': np.mean([h['writes_1h'] for h in recent]),
            'total_reads_24h': sum(h['reads_1h'] for h in recent),
            'total_writes_24h': sum(h['writes_1h'] for h in recent),
            'peak_reads': max(h['reads_1h'] for h in recent),
            'peak_writes': max(h['writes_1h'] for h in recent)
        }
    
    return jsonify(dataset)

@app.route('/api/random_dataset')
def get_random_dataset():
    db = get_db()
    metadata = db['metadata']
    
    # Get a random dataset with substantial history
    pipeline = [
        {'$match': {'history': {'$exists': True, '$ne': []}}},
        {'$addFields': {'history_length': {'$size': '$history'}}},
        {'$match': {'history_length': {'$gte': 24}}},
        {'$sample': {'size': 1}}
    ]
    
    datasets = list(metadata.aggregate(pipeline))
    
    if not datasets:
        return jsonify({'error': 'No datasets with sufficient history'}), 404
    
    dataset = datasets[0]
    dataset['_id'] = str(dataset['_id'])
    
    # Extract time series data
    history = dataset.get('history', [])[-48:]  # Last 48 hours
    
    time_series = {
        'timestamps': [h['timestamp'] for h in history],
        'reads': [h['reads_1h'] for h in history],
        'writes': [h['writes_1h'] for h in history],
        'bytes_read': [h['bytes_read_1h'] / (1024**3) for h in history]  # Convert to GB
    }
    
    return jsonify({
        'dataset_id': dataset['dataset_id'],
        'current_backend': dataset.get('current_backend'),
        'size_gb': dataset.get('size_gb'),
        'time_series': time_series
    })

@app.route('/api/migrations/recent')
def get_recent_migrations():
    db = get_db()
    jobs = db['migration_jobs']
    
    # Get last 50 migrations
    recent = list(jobs.find().sort('created_at', -1).limit(50))
    
    for job in recent:
        job['_id'] = str(job['_id'])
    
    return jsonify(recent)

@app.route('/api/activity/realtime')
def get_realtime_activity():
    db = get_db()
    metadata = db['metadata']
    
    # Get datasets with recent activity (last update within 5 minutes)
    five_mins_ago = time.time() - 300
    
    pipeline = [
        {'$match': {'last_timestamp': {'$gte': five_mins_ago}}},
        {'$sort': {'reads_1h': -1}},
        {'$limit': 10}
    ]
    
    active_datasets = list(metadata.aggregate(pipeline))
    
    for ds in active_datasets:
        ds['_id'] = str(ds['_id'])
    
    return jsonify(active_datasets)

@app.route('/api/backend_costs')
def get_backend_costs():
    db = get_db()
    metadata = db['metadata']
    
    # Calculate cost breakdown by backend
    all_datasets = list(metadata.find())
    
    backend_costs = defaultdict(lambda: {'storage': 0, 'access': 0, 'latency': 0, 'total': 0})
    
    for ds in all_datasets:
        backend = ds.get('current_backend', 'on-prem')
        size_gb = ds.get('size_gb', 0)
        history = ds.get('history', [])
        
        if not history:
            continue
            
        recent = history[-24:]
        total_reads = sum(h.get('reads_1h', 0) for h in recent)
        total_writes = sum(h.get('writes_1h', 0) for h in recent)
        
        prices = PRICING_CONFIG[backend]
        
        storage_cost = size_gb * prices['P_store'] * (24 / HOURS_IN_MONTH)
        access_cost = (total_reads * prices['P_read']) + (total_writes * prices['P_write'])
        latency_cost = total_reads * prices['P_latency'] * SLA_PENALTY_PER_SECOND
        
        backend_costs[backend]['storage'] += storage_cost
        backend_costs[backend]['access'] += access_cost
        backend_costs[backend]['latency'] += latency_cost
        backend_costs[backend]['total'] += storage_cost + access_cost + latency_cost
    
    return jsonify(dict(backend_costs))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)