#!/usr/bin/env python3
"""
Advanced Analytics and Reporting Module for ICMS

This script generates comprehensive reports on:
- Cost savings analysis
- Migration efficiency
- ML model performance
- System health metrics
"""

import json
import time
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np
from collections import defaultdict

# Configuration
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'icms_db'

PRICING_CONFIG = {
    'on-prem': {'P_store': 0.40, 'P_read': 0.00, 'P_write': 0.00, 'P_latency': 0.01},
    'private-cloud': {'P_store': 0.25, 'P_read': 0.05, 'P_write': 0.05, 'P_latency': 0.03},
    'public-hot': {'P_store': 0.20, 'P_read': 0.04, 'P_write': 0.04, 'P_latency': 0.07},
    'public-cold': {'P_store': 0.04, 'P_read': 0.001, 'P_write': 0.001, 'P_latency': 4.0}
}

HOURS_IN_MONTH = 720.0
SLA_PENALTY = 0.0001

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def calculate_baseline_cost_detailed(datasets):
    """Calculate what it would cost if datasets never moved"""
    total = 0
    breakdown = {'storage': 0, 'access': 0, 'latency': 0}
    
    for ds in datasets:
        backend = ds.get('current_backend', 'on-prem')
        size_gb = ds.get('size_gb', 0)
        history = ds.get('history', [])
        
        if not history:
            continue
            
        recent = history[-24:]
        reads = sum(h.get('reads_1h', 0) for h in recent)
        writes = sum(h.get('writes_1h', 0) for h in recent)
        
        prices = PRICING_CONFIG[backend]
        
        storage = size_gb * prices['P_store'] * (24 / HOURS_IN_MONTH)
        access = (reads * prices['P_read']) + (writes * prices['P_write'])
        latency = reads * prices['P_latency'] * SLA_PENALTY
        
        breakdown['storage'] += storage
        breakdown['access'] += access
        breakdown['latency'] += latency
        total += storage + access + latency
    
    return total, breakdown

def calculate_optimized_cost_detailed(datasets, migrations):
    """Calculate actual cost with all optimizations"""
    total = 0
    breakdown = {'storage': 0, 'access': 0, 'latency': 0, 'migration': 0}
    
    for ds in datasets:
        backend = ds.get('current_backend', 'on-prem')
        size_gb = ds.get('size_gb', 0)
        history = ds.get('history', [])
        
        if not history:
            continue
            
        recent = history[-24:]
        reads = sum(h.get('reads_1h', 0) for h in recent)
        writes = sum(h.get('writes_1h', 0) for h in recent)
        
        prices = PRICING_CONFIG[backend]
        
        storage = size_gb * prices['P_store'] * (24 / HOURS_IN_MONTH)
        access = (reads * prices['P_read']) + (writes * prices['P_write'])
        latency = reads * prices['P_latency'] * SLA_PENALTY
        
        breakdown['storage'] += storage
        breakdown['access'] += access
        breakdown['latency'] += latency
        total += storage + access + latency
    
    # Migration costs
    for mig in migrations:
        if mig.get('status') == 'COMPLETE':
            breakdown['migration'] += 0.5
            total += 0.5
    
    return total, breakdown

def analyze_migration_efficiency(migrations):
    """Analyze migration job performance"""
    stats = {
        'total': len(migrations),
        'by_status': defaultdict(int),
        'by_reason': defaultdict(int),
        'avg_duration': 0,
        'success_rate': 0
    }
    
    durations = []
    
    for mig in migrations:
        status = mig.get('status', 'UNKNOWN')
        reason = mig.get('reason', 'Unknown')
        
        stats['by_status'][status] += 1
        stats['by_reason'][reason] += 1
        
        if mig.get('duration_sec'):
            durations.append(mig['duration_sec'])
    
    if durations:
        stats['avg_duration'] = np.mean(durations)
        stats['median_duration'] = np.median(durations)
        stats['min_duration'] = np.min(durations)
        stats['max_duration'] = np.max(durations)
    
    completed = stats['by_status']['COMPLETE']
    failed = stats['by_status']['FAILED']
    
    if completed + failed > 0:
        stats['success_rate'] = (completed / (completed + failed)) * 100
    
    return stats

def analyze_backend_distribution(datasets):
    """Analyze how datasets are distributed across backends"""
    distribution = defaultdict(lambda: {'count': 0, 'total_gb': 0})
    
    for ds in datasets:
        backend = ds.get('current_backend', 'unknown')
        size = ds.get('size_gb', 0)
        
        distribution[backend]['count'] += 1
        distribution[backend]['total_gb'] += size
    
    return dict(distribution)

def analyze_access_patterns(datasets):
    """Analyze overall access patterns"""
    total_reads = 0
    total_writes = 0
    hot_datasets = 0
    warm_datasets = 0
    cold_datasets = 0
    
    for ds in datasets:
        history = ds.get('history', [])
        if not history:
            continue
            
        recent = history[-24:]
        reads = sum(h.get('reads_1h', 0) for h in recent)
        writes = sum(h.get('writes_1h', 0) for h in recent)
        
        total_reads += reads
        total_writes += writes
        
        avg_reads = reads / 24
        
        if avg_reads > 100:
            hot_datasets += 1
        elif avg_reads > 10:
            warm_datasets += 1
        else:
            cold_datasets += 1
    
    return {
        'total_reads_24h': total_reads,
        'total_writes_24h': total_writes,
        'hot_datasets': hot_datasets,
        'warm_datasets': warm_datasets,
        'cold_datasets': cold_datasets
    }

def generate_report():
    """Generate comprehensive analytics report"""
    print("=" * 70)
    print("ICMS ANALYTICS REPORT")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print()
    
    db = get_db()
    metadata = db['metadata']
    jobs = db['migration_jobs']
    
    # Fetch data
    all_datasets = list(metadata.find())
    all_migrations = list(jobs.find())
    
    print("📊 SYSTEM OVERVIEW")
    print("-" * 70)
    print(f"Total Datasets:       {len(all_datasets)}")
    print(f"Total Migrations:     {len(all_migrations)}")
    print()
    
    # Cost Analysis
    print("💰 COST ANALYSIS (Last 24 Hours)")
    print("-" * 70)
    
    baseline_cost, baseline_breakdown = calculate_baseline_cost_detailed(all_datasets)
    optimized_cost, optimized_breakdown = calculate_optimized_cost_detailed(all_datasets, all_migrations)
    
    savings = baseline_cost - optimized_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    print(f"Baseline Cost:        ${baseline_cost:.2f}")
    print(f"  - Storage:          ${baseline_breakdown['storage']:.2f}")
    print(f"  - Access:           ${baseline_breakdown['access']:.2f}")
    print(f"  - Latency Penalty:  ${baseline_breakdown['latency']:.2f}")
    print()
    print(f"Optimized Cost:       ${optimized_cost:.2f}")
    print(f"  - Storage:          ${optimized_breakdown['storage']:.2f}")
    print(f"  - Access:           ${optimized_breakdown['access']:.2f}")
    print(f"  - Latency Penalty:  ${optimized_breakdown['latency']:.2f}")
    print(f"  - Migration:        ${optimized_breakdown['migration']:.2f}")
    print()
    print(f"💚 TOTAL SAVINGS:     ${savings:.2f} ({savings_pct:.1f}%)")
    print()
    
    # Extrapolated Savings
    monthly_savings = savings * 30
    annual_savings = savings * 365
    
    print("📈 PROJECTED SAVINGS")
    print("-" * 70)
    print(f"Monthly (30 days):    ${monthly_savings:.2f}")
    print(f"Annual (365 days):    ${annual_savings:.2f}")
    print()
    
    # Migration Efficiency
    print("🔄 MIGRATION EFFICIENCY")
    print("-" * 70)
    
    mig_stats = analyze_migration_efficiency(all_migrations)
    
    print(f"Total Migrations:     {mig_stats['total']}")
    print(f"Success Rate:         {mig_stats['success_rate']:.1f}%")
    print()
    print("By Status:")
    for status, count in mig_stats['by_status'].items():
        print(f"  - {status:12s}: {count}")
    print()
    print("By Reason:")
    for reason, count in mig_stats['by_reason'].items():
        print(f"  - {reason:20s}: {count}")
    print()
    
    if mig_stats['avg_duration']:
        print("Duration Stats:")
        print(f"  - Average:      {mig_stats['avg_duration']:.2f}s")
        print(f"  - Median:       {mig_stats['median_duration']:.2f}s")
        print(f"  - Min:          {mig_stats['min_duration']:.2f}s")
        print(f"  - Max:          {mig_stats['max_duration']:.2f}s")
    print()
    
    # Backend Distribution
    print("🗄️  BACKEND DISTRIBUTION")
    print("-" * 70)
    
    backend_dist = analyze_backend_distribution(all_datasets)
    
    for backend, stats in sorted(backend_dist.items()):
        count = stats['count']
        total_gb = stats['total_gb']
        pct = (count / len(all_datasets) * 100) if all_datasets else 0
        print(f"{backend:15s}: {count:4d} datasets ({pct:5.1f}%) | {total_gb:8.1f} GB")
    print()
    
    # Access Patterns
    print("📊 ACCESS PATTERNS (Last 24 Hours)")
    print("-" * 70)
    
    access_stats = analyze_access_patterns(all_datasets)
    
    print(f"Total Reads:          {access_stats['total_reads_24h']:,}")
    print(f"Total Writes:         {access_stats['total_writes_24h']:,}")
    print()
    print("Dataset Temperature:")
    print(f"  - Hot (>100/h):     {access_stats['hot_datasets']}")
    print(f"  - Warm (10-100/h):  {access_stats['warm_datasets']}")
    print(f"  - Cold (<10/h):     {access_stats['cold_datasets']}")
    print()
    
    print("=" * 70)
    print("Report Complete!")
    print("=" * 70)

def generate_json_report():
    """Generate JSON report for API consumption"""
    db = get_db()
    metadata = db['metadata']
    jobs = db['migration_jobs']
    
    all_datasets = list(metadata.find())
    all_migrations = list(jobs.find())
    
    # Calculate metrics
    baseline_cost, baseline_breakdown = calculate_baseline_cost_detailed(all_datasets)
    optimized_cost, optimized_breakdown = calculate_optimized_cost_detailed(all_datasets, all_migrations)
    savings = baseline_cost - optimized_cost
    
    mig_stats = analyze_migration_efficiency(all_migrations)
    backend_dist = analyze_backend_distribution(all_datasets)
    access_stats = analyze_access_patterns(all_datasets)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'system': {
            'total_datasets': len(all_datasets),
            'total_migrations': len(all_migrations)
        },
        'costs': {
            'baseline': {
                'total': baseline_cost,
                'breakdown': baseline_breakdown
            },
            'optimized': {
                'total': optimized_cost,
                'breakdown': optimized_breakdown
            },
            'savings': {
                'amount': savings,
                'percentage': (savings / baseline_cost * 100) if baseline_cost > 0 else 0,
                'monthly': savings * 30,
                'annual': savings * 365
            }
        },
        'migrations': {
            'efficiency': mig_stats
        },
        'backends': backend_dist,
        'access_patterns': access_stats
    }
    
    return report

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        report = generate_json_report()
        print(json.dumps(report, indent=2, default=str))
    else:
        generate_report()