from pydantic import BaseModel, Field
from typing import List, Optional

# ── MCQ Schemas ──

class MCQOptions(BaseModel):
    """Fixed A/B/C/D options — avoids Dict[str, str] which generates
    additionalProperties in JSON Schema (unsupported by Gemini Developer API)."""
    A: str = ""
    B: str = ""
    C: str = ""
    D: str = ""

class MCQQuestion(BaseModel):
    QuestionNumber: str         # e.g., "May/June/12/2023/Q1"
    Question_text: str
    Unit: Optional[str] = None
    Topic: Optional[str] = None
    Topic_name: Optional[str] = None
    Sub_topic: Optional[List[str]] = None
    Difficulty_score: int = 0
    Difficulty_label: str = ""
    Command_words: List[str] = Field(default_factory=list)
    Options: MCQOptions         # fixed A/B/C/D — no additionalProperties
    Marks: int = 1
    image: Optional[str] = None

class MCQAnswer(BaseModel):
    AnswerNumber: str           # e.g., "May/June/12/2023/Q1"
    Answer: str                 # e.g., "A"
    image: Optional[str] = None

# ── Essay Schemas ──

class EssayQuestionSubSubPart(BaseModel):
    sub_part_label: str         # e.g., "(1)", "(a)"
    sub_part_text: str
    marks: int = 1
    image: Optional[str] = None

class EssayQuestionSubPart(BaseModel):
    sub_part_label: str         # e.g., "(i)", "(ii)"
    sub_part_text: str
    marks: int = 1
    image: Optional[str] = None
    sub_parts: List[EssayQuestionSubSubPart] = Field(default_factory=list)

class EssayQuestionPart(BaseModel):
    part_label: str             # e.g., "a", "b"
    part_text: str = ""
    marks: int = 1
    image: Optional[str] = None
    sub_parts: List[EssayQuestionSubPart] = Field(default_factory=list)

class EssayQuestion(BaseModel):
    QuestionNumber: str         # e.g., "Oct/Nov/41/2023/Q3"
    Question_text: str = ""
    Unit: Optional[str] = None
    Topic: Optional[str] = None
    Topic_name: Optional[str] = None
    Sub_topic: Optional[List[str]] = None
    Difficulty_score: int = 0
    Difficulty_label: str = ""
    Command_words: List[str] = Field(default_factory=list)
    Marks: int = 1
    image: Optional[str] = None
    Parts: List[EssayQuestionPart] = Field(default_factory=list)

class EssayPartAnswer(BaseModel):
    part_label: str
    part_answer: str
    image: Optional[str] = None

class EssayAnswer(BaseModel):
    AnswerNumber: str           # e.g., "Oct/Nov/41/2023/Q3"
    Answer_text: str = ""
    image: Optional[str] = None
    Parts: List[EssayPartAnswer] = Field(default_factory=list)

# ── Syllabus Schema ──

class SyllabusUnit(BaseModel):
    unit_number: str
    unit_name: str
    topic_number: str
    topic_name: str
    sub_topic: str

# ── Container schemas for structured LLM parsing ──

class MCQQuestionsContainer(BaseModel):
    questions: List[MCQQuestion]

class MCQAnswersContainer(BaseModel):
    answers: List[MCQAnswer]

class EssayQuestionsContainer(BaseModel):
    questions: List[EssayQuestion]

class EssayAnswersContainer(BaseModel):
    answers: List[EssayAnswer]
