import os
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from pipeline.extractor.schemas import SyllabusUnit
from pipeline.config import settings

logger = logging.getLogger(__name__)

class TopicClassificationResult(BaseModel):
    unit: str = Field(description="The matching syllabus Unit number (e.g. '1', '2')")
    topic: str = Field(description="The matching syllabus Topic number (e.g. '1.2', '2.1')")
    topic_name: str = Field(description="The matching syllabus Topic name exactly as it appears in the syllabus")
    sub_topics: List[str] = Field(description="List of specific sub-topics tested, matching keywords in sub_topic field or inferred")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of why this question maps to this topic")

class TopicClassifierAgent:
    """Agentic module that classifies exam questions to syllabus units/topics using LLMs."""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        # Determine provider & model
        self.provider = provider or os.getenv("EDGENIE_LLM_PROVIDER", "google")
        
        # Check API key existence and select appropriate client
        if self.provider == "google" and os.getenv("GOOGLE_API_KEY"):
            self.model_name = model_name or "gemini-1.5-flash"
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.1)
        elif self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            self.model_name = model_name or "gpt-4o-mini"
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.1)
        elif self.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            self.model_name = model_name or "claude-3-5-haiku-20241022"
            self.llm = ChatAnthropic(model=self.model_name, temperature=0.1)
        else:
            # Fallback checks if first choice is not configured
            if os.getenv("GOOGLE_API_KEY"):
                self.provider = "google"
                self.model_name = "gemini-1.5-flash"
                self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.1)
            elif os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
                self.model_name = "gpt-4o-mini"
                self.llm = ChatOpenAI(model=self.model_name, temperature=0.1)
            else:
                raise ValueError("No valid LLM API keys found in environment. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

        logger.info(f"Initialized TopicClassifierAgent with provider={self.provider}, model={self.model_name}")
        
        # Build structured output chain
        self.structured_llm = self.llm.with_structured_output(TopicClassificationResult)

    def classify_question(self, question_text: str, syllabus_text: str) -> TopicClassificationResult:
        """Classifies a question's text against the syllabus block."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational examiner for O/A-Levels.
Your task is to classify an exam question to the correct Unit and Topic based on the provided syllabus.

Syllabus Reference:
{syllabus_text}

Rules:
1. You MUST match to a Unit and Topic that actually exists in the Syllabus Reference.
2. Under no circumstances should you invent a topic number or topic name.
3. Extract sub-topics tested by the question (they can be exact matching keywords from the sub_topic field or highly related terms).
4. Provide a confidence score (0.0 to 1.0) and brief reasoning.
"""),
            ("human", "Syllabus Question Text:\n{question_text}\n\nPerform classification:")
        ])
        
        chain = prompt | self.structured_llm
        
        try:
            result = chain.invoke({
                "syllabus_text": syllabus_text,
                "question_text": question_text
            })
            return result
        except Exception as e:
            logger.error(f"Error during question topic classification: {e}")
            # Fallback result to avoid crashing pipeline
            return TopicClassificationResult(
                unit="1",
                topic="1.1",
                topic_name="Unknown Topic (Error)",
                sub_topics=[],
                confidence=0.0,
                reasoning=f"Error occurred during LLM classification: {str(e)}"
            )
