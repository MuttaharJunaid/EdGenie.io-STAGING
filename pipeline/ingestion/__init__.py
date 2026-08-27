"""
Stage 4 — MongoDB Ingestion & Firebase Asset Storage
"""
from pipeline.ingestion.firebase_storage import FirebaseStorageManager
from pipeline.ingestion.mongo_client import MongoDBClient
from pipeline.ingestion.ingestor import MongoIngestor

__all__ = ["FirebaseStorageManager", "MongoDBClient", "MongoIngestor"]
