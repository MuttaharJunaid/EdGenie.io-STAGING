import logging
import os
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "edgenie")


class MongoDBClient:
    """
    MongoDB client manager for EdGenie database collections:
    - subjects
    - papers
    - questions
    - answers
    - visuals
    - pipeline_runs
    """

    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self._client: Optional[MongoClient] = None
        self._db = None

    def get_db(self):
        """Initializes and returns MongoDB database connection."""
        if self._db is not None:
            return self._db

        try:
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            # Test connection
            self._client.admin.command("ping")
            self._db = self._client[self.db_name]
            logger.info(f"MongoDB connected: {self.uri}/{self.db_name}")
            self.init_collections()
        except Exception as e:
            logger.warning(f"MongoDB connection timeout/offline ({e}). Operating in memory/dry-run mode.")
            self._db = None

        return self._db

    def init_collections(self):
        """Ensures all 6 MongoDB collections exist with required indexes."""
        if self._db is None:
            return

        db = self._db
        collections = ["subjects", "papers", "questions", "answers", "visuals", "pipeline_runs"]

        for coll in collections:
            if coll not in db.list_collection_names():
                db.create_collection(coll)

        # Create indexes
        db.subjects.create_index([("subject_code", ASCENDING)], unique=True)

        db.papers.create_index([("paper_code", ASCENDING)], unique=True)
        db.papers.create_index([("subject_code", ASCENDING), ("year", DESCENDING), ("session", ASCENDING)])

        db.questions.create_index([("question_key", ASCENDING)], unique=True)
        db.questions.create_index([("paper_code", ASCENDING)])
        db.questions.create_index([("subject_code", ASCENDING), ("topic_number", ASCENDING)])
        db.questions.create_index([("difficulty_label", ASCENDING), ("difficulty_score", ASCENDING)])

        db.answers.create_index([("question_id", ASCENDING)], unique=True)

        db.visuals.create_index([("paper_code", ASCENDING), ("question_number", ASCENDING)])

        db.pipeline_runs.create_index([("subject_code", ASCENDING), ("year", DESCENDING)])

        logger.info("MongoDB collections and indexes initialized successfully.")

    def upsert_document(self, collection_name: str, query: Dict[str, Any], document: Dict[str, Any]) -> str:
        """Helper to upsert a document into a collection."""
        db = self.get_db()
        if db is None:
            return str(query)

        collection = db[collection_name]
        collection.update_one(query, {"$set": document}, upsert=True)
        return str(query)

    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> int:
        """Helper to insert multiple documents."""
        db = self.get_db()
        if db is None or not documents:
            return 0

        collection = db[collection_name]
        res = collection.insert_many(documents)
        return len(res.inserted_ids)
