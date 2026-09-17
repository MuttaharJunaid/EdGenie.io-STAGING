import json
import logging
import os
import re
from typing import Any

import psycopg
from psycopg.rows import dict_row

from pipeline.scraper.subject_registry import SubjectRegistry

logger = logging.getLogger(__name__)

# Default connection: local postgres running edgenie DB
POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    os.getenv("DATABASE_URL", "postgresql://localhost:5432/edgenie")
)


SUBJECT_NAME_MAP = {
    "5054": ("Physics", "O-Level"),
    "5070": ("Chemistry", "O-Level"),
    "5090": ("Biology", "O-Level"),
    "4024": ("Mathematics", "O-Level"),
    "2210": ("Computer Science", "O-Level"),
    "2281": ("Economics", "O-Level"),
    "7115": ("Business Studies", "O-Level"),
    "7707": ("Accounting", "O-Level"),
    "1123": ("English Language", "O-Level"),
    "9702": ("Physics", "A-Level"),
    "9701": ("Chemistry", "A-Level"),
    "9700": ("Biology", "A-Level"),
    "9709": ("Mathematics", "A-Level"),
    "9618": ("Computer Science", "A-Level"),
    "9708": ("Economics", "A-Level"),
    "9609": ("Business", "A-Level"),
}


SCHEMA_SQL = """
-- 1. Subjects Table
CREATE TABLE IF NOT EXISTS subjects (
    subject_code VARCHAR(10) PRIMARY KEY,
    subject_name VARCHAR(100) NOT NULL,
    level VARCHAR(20) DEFAULT 'O-Level',
    syllabus_rules_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Papers Table
CREATE TABLE IF NOT EXISTS papers (
    paper_code VARCHAR(50) PRIMARY KEY,
    subject_code VARCHAR(10) NOT NULL REFERENCES subjects(subject_code) ON DELETE CASCADE,
    year INT NOT NULL,
    session VARCHAR(10) NOT NULL,
    variant VARCHAR(10),
    paper_number INT,
    paper_type VARCHAR(20) NOT NULL,
    category VARCHAR(20) NOT NULL,
    pdf_filename VARCHAR(255),
    pdf_cdn_url TEXT,
    question_count INT DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_subject_year_session ON papers(subject_code, year DESC, session);
CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(category);

-- 3. Questions Table (Relational + JSONB)
CREATE TABLE IF NOT EXISTS questions (
    question_key VARCHAR(100) PRIMARY KEY,
    paper_code VARCHAR(50) NOT NULL REFERENCES papers(paper_code) ON DELETE CASCADE,
    subject_code VARCHAR(10) NOT NULL REFERENCES subjects(subject_code),
    year INT NOT NULL,
    session VARCHAR(10) NOT NULL,
    variant VARCHAR(10),
    question_number INT NOT NULL,
    question_number_formatted VARCHAR(100),
    question_text TEXT NOT NULL,
    marks INT DEFAULT 1,
    unit VARCHAR(50) DEFAULT '',
    topic VARCHAR(50) DEFAULT '',
    topic_name VARCHAR(255) DEFAULT '',
    sub_topics JSONB DEFAULT '[]'::jsonb,
    difficulty_score INT DEFAULT 0,
    difficulty_label VARCHAR(20) DEFAULT '',
    command_words JSONB DEFAULT '[]'::jsonb,
    options JSONB,
    parts JSONB,
    image_filename VARCHAR(255),
    image_cdn_url TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_questions_paper_code ON questions(paper_code);
CREATE INDEX IF NOT EXISTS idx_questions_subject_topic ON questions(subject_code, topic);
CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions(difficulty_label, difficulty_score);
CREATE INDEX IF NOT EXISTS idx_questions_sub_topics ON questions USING GIN (sub_topics);
CREATE INDEX IF NOT EXISTS idx_questions_command_words ON questions USING GIN (command_words);
CREATE INDEX IF NOT EXISTS idx_questions_options ON questions USING GIN (options);
CREATE INDEX IF NOT EXISTS idx_questions_parts ON questions USING GIN (parts);

-- 4. Answers Table
CREATE TABLE IF NOT EXISTS answers (
    id VARCHAR(100) PRIMARY KEY,
    question_key VARCHAR(100),
    paper_code VARCHAR(50) NOT NULL REFERENCES papers(paper_code) ON DELETE CASCADE,
    subject_code VARCHAR(10) NOT NULL REFERENCES subjects(subject_code),
    year INT NOT NULL,
    session VARCHAR(10) NOT NULL,
    variant VARCHAR(10),
    question_number INT NOT NULL,
    answer_text TEXT,
    parts JSONB,
    image_filename VARCHAR(255),
    image_cdn_url TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_answers_question_key ON answers(question_key);
CREATE INDEX IF NOT EXISTS idx_answers_paper_code ON answers(paper_code);
CREATE INDEX IF NOT EXISTS idx_answers_parts ON answers USING GIN (parts);

-- 5. Visuals Table
CREATE TABLE IF NOT EXISTS visuals (
    id VARCHAR(255) PRIMARY KEY,
    paper_code VARCHAR(50) NOT NULL REFERENCES papers(paper_code) ON DELETE CASCADE,
    subject_code VARCHAR(10) NOT NULL REFERENCES subjects(subject_code),
    year INT NOT NULL,
    session VARCHAR(10) NOT NULL,
    question_number INT,
    caption VARCHAR(100),
    page INT,
    bbox JSONB,
    filename VARCHAR(255) NOT NULL,
    relative_path VARCHAR(255) NOT NULL,
    cdn_url TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_visuals_paper_code ON visuals(paper_code);
CREATE INDEX IF NOT EXISTS idx_visuals_question_number ON visuals(paper_code, question_number);

-- 6. Pipeline Runs Table
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    stage VARCHAR(50) NOT NULL,
    subject_code VARCHAR(10),
    year INT,
    status VARCHAR(20) NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);
"""


class PostgresClient:
    """
    PostgreSQL Client Manager for EdGenie (Relational + JSONB).
    Handles schema initialization, connection management, and atomic upserts.
    """

    def __init__(self, uri: str = POSTGRES_URI):
        self.uri = uri

    def get_connection(self) -> psycopg.Connection:
        """Opens and returns a new connection to PostgreSQL."""
        try:
            conn = psycopg.connect(self.uri, autocommit=True, row_factory=dict_row)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL at {self.uri}: {e}")
            raise

    def init_schema(self) -> None:
        """Executes DDL to initialize all 6 tables and GIN/B-Tree indexes."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
        logger.info("PostgreSQL schema (Relational + JSONB) initialized successfully.")

    def ensure_subject_exists(self, subject_code: str) -> None:
        """Ensures the subject exists in the subjects table to satisfy foreign keys."""
        entry = SubjectRegistry.lookup_by_code(subject_code)
        if entry:
            raw_title = entry.get("title") or entry.get("subject_slug") or f"Subject {subject_code}"
            name = re.sub(r"\s*[-–(]?\s*\d{4}\s*\)?\s*$", "", raw_title).strip()
            level = "A-Level" if "a-level" in entry.get("level", "").lower() else "O-Level"
        else:
            name, level = SUBJECT_NAME_MAP.get(subject_code, (f"Subject {subject_code}", "O-Level"))
        sql = """
        INSERT INTO subjects (subject_code, subject_name, level, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (subject_code) DO NOTHING;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (subject_code, name, level))

    def upsert_paper(self, paper: dict[str, Any]) -> None:
        """Upserts a paper record into the papers table."""
        self.ensure_subject_exists(paper["subject_code"])
        sql = """
        INSERT INTO papers (
            paper_code, subject_code, year, session, variant, paper_number,
            paper_type, category, pdf_filename, pdf_cdn_url, question_count, metadata, updated_at
        ) VALUES (
            %(paper_code)s, %(subject_code)s, %(year)s, %(session)s, %(variant)s, %(paper_number)s,
            %(paper_type)s, %(category)s, %(pdf_filename)s, %(pdf_cdn_url)s, %(question_count)s,
            %(metadata)s, CURRENT_TIMESTAMP
        )
        ON CONFLICT (paper_code) DO UPDATE SET
            subject_code = EXCLUDED.subject_code,
            year = EXCLUDED.year,
            session = EXCLUDED.session,
            variant = EXCLUDED.variant,
            paper_number = EXCLUDED.paper_number,
            paper_type = EXCLUDED.paper_type,
            category = EXCLUDED.category,
            pdf_filename = EXCLUDED.pdf_filename,
            pdf_cdn_url = COALESCE(EXCLUDED.pdf_cdn_url, papers.pdf_cdn_url),
            question_count = EXCLUDED.question_count,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    {
                        "paper_code": paper["paper_code"],
                        "subject_code": paper["subject_code"],
                        "year": paper["year"],
                        "session": paper["session"],
                        "variant": paper.get("variant"),
                        "paper_number": paper.get("paper_number"),
                        "paper_type": paper.get("paper_type", "qp"),
                        "category": paper.get("category", "essay_qp"),
                        "pdf_filename": paper.get("pdf_filename"),
                        "pdf_cdn_url": paper.get("pdf_cdn_url"),
                        "question_count": paper.get("question_count", 0),
                        "metadata": json.dumps(paper.get("metadata", {})),
                    },
                )

    def upsert_questions(self, questions: list[dict[str, Any]]) -> int:
        """Batch upserts question records (MCQ & Essay) into the questions table."""
        if not questions:
            return 0

        sql = """
        INSERT INTO questions (
            question_key, paper_code, subject_code, year, session, variant,
            question_number, question_number_formatted, question_text, marks,
            unit, topic, topic_name, sub_topics, difficulty_score, difficulty_label,
            command_words, options, parts, image_filename, image_cdn_url, metadata, updated_at
        ) VALUES (
            %(question_key)s, %(paper_code)s, %(subject_code)s, %(year)s, %(session)s, %(variant)s,
            %(question_number)s, %(question_number_formatted)s, %(question_text)s, %(marks)s,
            %(unit)s, %(topic)s, %(topic_name)s, %(sub_topics)s, %(difficulty_score)s, %(difficulty_label)s,
            %(command_words)s, %(options)s, %(parts)s, %(image_filename)s, %(image_cdn_url)s, %(metadata)s, CURRENT_TIMESTAMP
        )
        ON CONFLICT (question_key) DO UPDATE SET
            paper_code = EXCLUDED.paper_code,
            subject_code = EXCLUDED.subject_code,
            year = EXCLUDED.year,
            session = EXCLUDED.session,
            variant = EXCLUDED.variant,
            question_number = EXCLUDED.question_number,
            question_number_formatted = EXCLUDED.question_number_formatted,
            question_text = EXCLUDED.question_text,
            marks = EXCLUDED.marks,
            unit = EXCLUDED.unit,
            topic = EXCLUDED.topic,
            topic_name = EXCLUDED.topic_name,
            sub_topics = EXCLUDED.sub_topics,
            difficulty_score = EXCLUDED.difficulty_score,
            difficulty_label = EXCLUDED.difficulty_label,
            command_words = EXCLUDED.command_words,
            options = EXCLUDED.options,
            parts = EXCLUDED.parts,
            image_filename = EXCLUDED.image_filename,
            image_cdn_url = COALESCE(EXCLUDED.image_cdn_url, questions.image_cdn_url),
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP;
        """

        params_list = []
        for q in questions:
            params_list.append(
                {
                    "question_key": q["question_key"],
                    "paper_code": q["paper_code"],
                    "subject_code": q["subject_code"],
                    "year": q["year"],
                    "session": q["session"],
                    "variant": q.get("variant"),
                    "question_number": q["question_number"],
                    "question_number_formatted": q.get("question_number_formatted", ""),
                    "question_text": q.get("question_text", ""),
                    "marks": q.get("marks", 1),
                    "unit": q.get("unit", ""),
                    "topic": q.get("topic", ""),
                    "topic_name": q.get("topic_name", ""),
                    "sub_topics": json.dumps(q.get("sub_topics", [])),
                    "difficulty_score": q.get("difficulty_score", 0),
                    "difficulty_label": q.get("difficulty_label", ""),
                    "command_words": json.dumps(q.get("command_words", [])),
                    "options": json.dumps(q.get("options")) if q.get("options") is not None else None,
                    "parts": json.dumps(q.get("parts")) if q.get("parts") is not None else None,
                    "image_filename": q.get("image_filename"),
                    "image_cdn_url": q.get("image_cdn_url"),
                    "metadata": json.dumps(q.get("metadata", {})),
                }
            )

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)

        return len(questions)

    def upsert_answers(self, answers: list[dict[str, Any]]) -> int:
        """Batch upserts answer records into the answers table."""
        if not answers:
            return 0

        sql = """
        INSERT INTO answers (
            id, question_key, paper_code, subject_code, year, session, variant,
            question_number, answer_text, parts, image_filename, image_cdn_url, metadata, updated_at
        ) VALUES (
            %(id)s, %(question_key)s, %(paper_code)s, %(subject_code)s, %(year)s, %(session)s, %(variant)s,
            %(question_number)s, %(answer_text)s, %(parts)s, %(image_filename)s, %(image_cdn_url)s, %(metadata)s, CURRENT_TIMESTAMP
        )
        ON CONFLICT (id) DO UPDATE SET
            question_key = EXCLUDED.question_key,
            paper_code = EXCLUDED.paper_code,
            subject_code = EXCLUDED.subject_code,
            year = EXCLUDED.year,
            session = EXCLUDED.session,
            variant = EXCLUDED.variant,
            question_number = EXCLUDED.question_number,
            answer_text = EXCLUDED.answer_text,
            parts = EXCLUDED.parts,
            image_filename = EXCLUDED.image_filename,
            image_cdn_url = COALESCE(EXCLUDED.image_cdn_url, answers.image_cdn_url),
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP;
        """

        params_list = []
        for a in answers:
            params_list.append(
                {
                    "id": a["id"],
                    "question_key": a.get("question_key"),
                    "paper_code": a["paper_code"],
                    "subject_code": a["subject_code"],
                    "year": a["year"],
                    "session": a["session"],
                    "variant": a.get("variant"),
                    "question_number": a["question_number"],
                    "answer_text": a.get("answer_text", ""),
                    "parts": json.dumps(a.get("parts")) if a.get("parts") is not None else None,
                    "image_filename": a.get("image_filename"),
                    "image_cdn_url": a.get("image_cdn_url"),
                    "metadata": json.dumps(a.get("metadata", {})),
                }
            )

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)

        return len(answers)

    def upsert_visuals(self, visuals: list[dict[str, Any]]) -> int:
        """Batch upserts visual figure metadata records into the visuals table."""
        if not visuals:
            return 0

        sql = """
        INSERT INTO visuals (
            id, paper_code, subject_code, year, session, question_number,
            caption, page, bbox, filename, relative_path, cdn_url, created_at
        ) VALUES (
            %(id)s, %(paper_code)s, %(subject_code)s, %(year)s, %(session)s, %(question_number)s,
            %(caption)s, %(page)s, %(bbox)s, %(filename)s, %(relative_path)s, %(cdn_url)s, CURRENT_TIMESTAMP
        )
        ON CONFLICT (id) DO UPDATE SET
            paper_code = EXCLUDED.paper_code,
            subject_code = EXCLUDED.subject_code,
            year = EXCLUDED.year,
            session = EXCLUDED.session,
            question_number = EXCLUDED.question_number,
            caption = EXCLUDED.caption,
            page = EXCLUDED.page,
            bbox = EXCLUDED.bbox,
            filename = EXCLUDED.filename,
            relative_path = EXCLUDED.relative_path,
            cdn_url = COALESCE(EXCLUDED.cdn_url, visuals.cdn_url);
        """

        params_list = []
        for v in visuals:
            params_list.append(
                {
                    "id": v["id"],
                    "paper_code": v["paper_code"],
                    "subject_code": v["subject_code"],
                    "year": v["year"],
                    "session": v["session"],
                    "question_number": v.get("question_number"),
                    "caption": v.get("caption"),
                    "page": v.get("page"),
                    "bbox": json.dumps(v.get("bbox", [])),
                    "filename": v["filename"],
                    "relative_path": v.get("relative_path", ""),
                    "cdn_url": v.get("cdn_url"),
                }
            )

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)

        return len(visuals)

    def get_stats(self) -> dict[str, int]:
        """Returns row counts for all tables."""
        tables = ["subjects", "papers", "questions", "answers", "visuals", "pipeline_runs"]
        stats = {}
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for tbl in tables:
                    cur.execute(f"SELECT COUNT(*) as count FROM {tbl};")
                    row = cur.fetchone()
                    stats[tbl] = row["count"] if row else 0
        return stats
