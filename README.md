# ICMS - Intelligent Cloud Management System

A comprehensive cloud data management platform that uses machine learning to optimize data storage across multiple backends (on-premises, private cloud, and public cloud tiers). ICMS automatically analyzes data access patterns and migrates datasets to the most cost-effective storage tier while maintaining performance requirements.

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** (or Docker Engine + Docker Compose)
  - macOS: [Download Docker Desktop](https://docs.docker.com/desktop/install/mac-install/)
  - Windows: [Download Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
  - Linux: [Install Docker Engine](https://docs.docker.com/engine/install/)
- **4GB+ RAM** allocated to Docker
- **10GB+ free disk space**

### Installation & Startup

1. **Clone or download this repository**
   ```bash
   cd NetApp
   ```

2. **Make the start script executable** (Linux/macOS)
   ```bash
   chmod +x start.sh stop.sh
   ```

3. **Start ICMS**
   ```bash
   ./start.sh
   ```

That's it! The script will:
- ✅ Check Docker installation and start it if needed
- ✅ Verify system resources (disk space, memory)
- ✅ Build all Docker images
- ✅ Start all services
- ✅ Display the dashboard URL

**First run takes 5-10 minutes** to download images and build containers.

### Access the Dashboard

Once started, open your browser to:
```
http://localhost:8080
```

Wait 30-60 seconds after startup for services to initialize and data to appear.

## 📋 What ICMS Does

ICMS is a complete data lifecycle management system that:

1. **Generates Synthetic Data** - Simulates realistic dataset behavior with various access patterns
2. **Monitors in Real-Time** - Tracks reads, writes, and access patterns via Kafka
3. **Analyzes with ML** - Uses LSTM neural networks to predict future access patterns
4. **Optimizes Automatically** - Recommends and executes migrations to cost-effective storage tiers
5. **Visualizes Everything** - Beautiful dashboard showing costs, migrations, and analytics

## 🏗️ Architecture

### Services

| Service | Description | Port |
|---------|-------------|------|
| **Dashboard** | Web UI for monitoring and analytics | 8080 |
| **Producer** | Generates synthetic dataset access events | - |
| **Consumer** | Processes events and updates MongoDB | - |
| **Optimizer** | ML-powered optimization engine | - |
| **Migrator** | Executes data migrations between backends | - |
| **MongoDB** | Primary database for metadata and jobs | 27017 |
| **Kafka** | Message broker for event streaming | 9092 |
| **Zookeeper** | Coordinates Kafka cluster | - |

### Data Flow

```
Producer → Kafka → Consumer → MongoDB
                              ↓
                    Optimizer → Migration Jobs → Migrator
                              ↓
                          Dashboard (reads from MongoDB)
```

## 💰 Storage Tiers

ICMS manages data across 4 storage tiers:

| Tier | Storage Cost | Read Cost | Write Cost | Latency | Use Case |
|------|-------------|-----------|------------|---------|----------|
| **On-Prem** | $0.40/GB/mo | $0.00 | $0.00 | 0.01s | High-frequency access |
| **Private Cloud** | $0.25/GB/mo | $0.05 | $0.05 | 0.03s | Business hours data |
| **Public Hot** | $0.20/GB/mo | $0.04 | $0.04 | 0.07s | Frequently accessed |
| **Public Cold** | $0.04/GB/mo | $0.001 | $0.001 | 4.0s | Archive, rarely accessed |

## 📊 Dashboard Features

- **Real-Time Monitoring** - Live updates every 10 seconds
- **Cost Analytics** - Compare baseline vs optimized costs
- **Migration Tracking** - View all migrations and their status
- **Backend Distribution** - See where your data is stored
- **Activity Charts** - Visualize dataset access patterns
- **Service Control** - Monitor and manage all services

## 🛠️ Management Commands

### Using the Scripts

**Start ICMS:**
```bash
./start.sh
```

**Stop ICMS:**
```bash
./stop.sh
```

### Using Docker Compose Directly

If you prefer using Docker Compose commands directly:

**View logs:**
```bash
docker-compose logs -f
# or
docker compose logs -f
```

**View specific service logs:**
```bash
docker-compose logs -f producer
docker-compose logs -f consumer
docker-compose logs -f optimizer
```

**Restart a service:**
```bash
docker-compose restart dashboard
```

**View container status:**
```bash
docker-compose ps
```

**Stop all services:**
```bash
docker-compose down
```

**Stop and remove all data:**
```bash
docker-compose down -v
```

## 🔧 Configuration

### Environment Variables

Services are configured via environment variables in `docker-compose.yml`:

- `KAFKA_SERVER` - Kafka broker address (default: `kafka:29092`)
- `MONGO_URI` - MongoDB connection string (default: `mongodb://mongodb:27017/`)

### ML Models

ICMS uses pre-trained LSTM models for predictions:
- `model/reads_model_h8.pth` - Predicts read patterns
- `model/writes_model_h8.pth` - Predicts write patterns

If these files don't exist, dummy models will be created automatically.

### Producer Configuration

Edit `producer.py` to adjust:
- `NUM_DATASETS` - Number of datasets to simulate (default: 1000)
- `SIM_SPEED_SECONDS` - Simulation speed (default: 0.5s per hour)
- `SIM_CYCLE_DAYS` - Days to simulate (default: 30)

## 🐛 Troubleshooting

### Services Won't Start

1. **Check Docker is running:**
   ```bash
   docker info
   ```

2. **Check available resources:**
   - Docker Desktop: Settings → Resources → Memory (should be 4GB+)
   - Disk space: `df -h` (should have 10GB+ free)

3. **View error logs:**
   ```bash
   docker-compose logs
   ```

### Dashboard Shows No Data

1. **Wait 30-60 seconds** after startup for data generation
2. **Check producer is running:**
   ```bash
   docker-compose logs producer
   ```
3. **Check consumer is processing:**
   ```bash
   docker-compose logs consumer
   ```
4. **Verify MongoDB has data:**
   ```bash
   docker exec icms-mongodb mongosh --eval "db.getSiblingDB('icms_db').metadata.countDocuments({})"
   ```

### Port Already in Use

If port 8080 is already in use:
1. Stop the conflicting service
2. Or modify `docker-compose.yml` to use a different port:
   ```yaml
   ports:
     - "8081:8080"  # Change 8081 to your preferred port
   ```

### Build Failures

1. **Clear Docker cache:**
   ```bash
   docker system prune -a
   ```

2. **Rebuild from scratch:**
   ```bash
   docker-compose build --no-cache
   ```

## 📁 Project Structure

```
NetApp/
├── start.sh              # Main entry point - start everything
├── stop.sh               # Stop all services
├── README.md             # This file
├── docker-compose.yml    # Service orchestration
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
│
├── producer.py          # Data generation service
├── consumer.py          # Event processing service
├── optimizer.py         # ML optimization engine
├── migrator.py          # Migration execution
├── dashboard.py         # Web dashboard API
├── analytics.py         # Analytics utilities
│
├── model/               # ML models
│   ├── model.py
│   ├── train.py
│   ├── reads_model_h8.pth
│   └── writes_model_h8.pth
│
├── templates/           # Dashboard HTML templates
│   └── dashboard.html
│
└── static/              # Static assets (CSS, JS)
```

## 🔬 How It Works

### 1. Data Generation
The **Producer** simulates realistic dataset behavior:
- 4 persona types (business hours, batch processing, cold archive, viral content)
- Hourly access patterns with daily/weekly cycles
- Global events (flash sales, ETL jobs, audits)
- Natural churn (datasets created and deleted over time)

### 2. Event Processing
The **Consumer** processes Kafka events:
- Updates dataset metadata in MongoDB
- Maintains hourly history for each dataset
- Detects sudden access spikes
- Creates migration jobs for hot datasets

### 3. ML Optimization
The **Optimizer** analyzes patterns:
- Uses LSTM models to predict future reads/writes
- Calculates cost for each backend
- Recommends migrations based on cost savings
- Considers latency penalties and migration costs

### 4. Migration Execution
The **Migrator** executes migrations:
- Processes pending migration jobs
- Simulates data transfer between backends
- Updates dataset locations
- Tracks migration status and duration

### 5. Visualization
The **Dashboard** provides insights:
- Real-time cost comparisons
- Migration history and status
- Backend distribution charts
- Dataset activity visualizations

## 🎯 Use Cases

- **Cost Optimization** - Automatically move data to cheaper storage tiers
- **Performance Monitoring** - Track access patterns and optimize for latency
- **Capacity Planning** - Understand data growth and access trends
- **Multi-Cloud Management** - Manage data across on-prem and cloud backends
- **Compliance** - Archive old data to cold storage automatically

## 📈 Performance

- **Throughput**: Processes 1000+ datasets with hourly updates
- **Latency**: Real-time processing with <1 second event handling
- **Scalability**: Kafka-based architecture scales horizontally
- **Accuracy**: ML models trained on historical patterns

## 🤝 Contributing

This is a demonstration project. To extend it:

1. Modify `producer.py` to generate different data patterns
2. Adjust `optimizer.py` to change optimization algorithms
3. Customize `dashboard.html` for different visualizations
4. Add new storage backends in `optimizer.py` pricing config

## 📝 License

This project is provided as-is for demonstration purposes.

## 🙏 Acknowledgments

Built with:
- **Docker** - Containerization
- **Kafka** - Event streaming
- **MongoDB** - Data storage
- **Flask** - Web framework
- **PyTorch** - Machine learning
- **Chart.js** - Data visualization

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review Docker logs: `docker-compose logs`
3. Verify all services are running: `docker-compose ps`

---

**Made with ❤️ for intelligent cloud data management**

