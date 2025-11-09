import json
import time
from kafka import KafkaConsumer
from pymongo import MongoClient, ReturnDocument

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Kafka Endpoints (Input) ---
KAFKA_TOPIC = 'access_events'
KAFKA_SERVER = 'localhost:9092'
KAFKA_GROUP_ID = 'icms-reactors' # All consumers share this group

# --- MongoDB Endpoints (Output) ---
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'icms_db'
METADATA_COLLECTION = 'metadata' # For dataset stats
JOBS_COLLECTION = 'migration_jobs' # For creating jobs

# --- Real-Time Rules ---
HOT_READ_THRESHOLD = 100 # "Sudden Increase"
DEFAULT_HOT_TIER = 'public-hot'
COLD_TIERS = ['public-cold'] # Standard names

# ============================================================================
# SERVICE CONNECTIONS
# ============================================================================

def connect_to_mongo():
    """Connects to MongoDB, retrying until successful."""
    while True:
        try:
            print(f"CONSUMER: Attempting to connect to MongoDB at {MONGO_URI}...")
            db_client = MongoClient(MONGO_URI)
            db = db_client[DB_NAME]
            db.command('ping')
            print("CONSUMER: MongoDB connected successfully.")
            return db[METADATA_COLLECTION], db[JOBS_COLLECTION]
        except Exception as e:
            print(f"CONSUMER: Failed to connect to MongoDB: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def connect_to_kafka():
    """Connects to Kafka as a consumer, retrying until successful."""
    while True:
        try:
            print(f"CONSUMER: Attempting to connect to Kafka at {KAFKA_SERVER} (Group: {KAFKA_GROUP_ID})...")
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVER,
                group_id=KAFKA_GROUP_ID,
                auto_offset_reset='earliest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8')
            )
            print("CONSUMER: Kafka Consumer connected. Waiting for messages...")
            return consumer
        except Exception as e:
            print(f"CONSUMER: Failed to connect to Kafka: {e}. Retrying in 5 seconds...")
            time.sleep(5)

# ============================================================================
# MAIN REACTOR LOOP
# ============================================================================

def run_consumer():
    print("=" * 70)
    print("ICSM Real-Time Reactor (Consumer)")
    print("=" * 70)
    
    metadata_collection, jobs_collection = connect_to_mongo()
    consumer = connect_to_kafka()

    for message in consumer:
        try:
            event = message.value
            dataset_id = message.key
            
            if not dataset_id or not event:
                print(f"CONSUMER: Skipping malformed message: {message}")
                continue

            # === 1. Update Stats (Write to MongoDB 'metadata') ===
            
            # We create a simple history object for the optimizer
            new_history_entry = {
                "timestamp": event['timestamp'],
                "reads_1h": event['reads_1h'],
                "writes_1h": event['writes_1h'],
                "bytes_read_1h": event['bytes_read_1h'],
                "hour_of_day": event['hour_of_day'],
                "day_of_week": event['day_of_week']
            }
            
            stats = metadata_collection.find_one_and_update(
                { 'dataset_id': dataset_id },
                { 
                    '$set': {
                        # Set the *latest* stats for real-time checks
                        'last_timestamp': event['timestamp'],
                        'reads_1h': event['reads_1h'],
                        'writes_1h': event['writes_1h'],
                        'bytes_read_1h': event['bytes_read_1h'],
                        'size_gb': event['size_gb']
                    },
                    '$setOnInsert': {
                        'current_backend': event['current_backend'] 
                    },
                    '$push': {
                        # Add the new entry to our history array
                        'history': {
                            '$each': [new_history_entry],
                            '$slice': -24 # Keep only the last 24 entries
                        }
                    }
                },
                upsert=True, 
                return_document=ReturnDocument.AFTER 
            )

            # === 2. Run Real-Time "Sudden Increase" Logic ===
            current_backend = stats.get('current_backend', DEFAULT_HOT_TIER)
            is_hot = event['reads_1h'] > HOT_READ_THRESHOLD
            is_in_cold_storage = current_backend in COLD_TIERS

            # === 3. The Decision: Act on Emergency ===
            if is_hot and is_in_cold_storage:
                
                # === 4. The "Debounced" Job Creation (Write to MongoDB 'migration_jobs') ===
                jobs_collection.update_one(
                    {
                        'dataset_id': dataset_id,
                        'status': 'PENDING',
                        'reason': 'Real-Time Emergency' # This job is from the consumer
                    },
                    { 
                        '$set': { 
                            'from_backend': current_backend,
                            'to_backend': DEFAULT_HOT_TIER,
                            'created_at': time.time()
                        }
                    },
                    upsert=True 
                )
                print(f"CONSUMER: *** JOB CREATED (EMERGENCY) ***: Move {dataset_id} from {current_backend} to {DEFAULT_HOT_TIER}")

        except Exception as e:
            print(f"CONSUMER: Error processing message: {message}. Error: {e}")

if __name__ == "__main__":
    run_consumer()