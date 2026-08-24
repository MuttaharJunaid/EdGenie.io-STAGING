import os
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

class DifficultyAnalysisResult(BaseModel):
    difficulty_score: int = Field(description="Difficulty score on a 1-10 scale (1 = trivial, 10 = expert/extremely challenging)")
    difficulty_label: str = Field(description="Difficulty category: 'easy' (1-3), 'medium' (4-6), 'hard' (7-8), 'expert' (9-10)")
    command_words: List[str] = Field(description="Command words found in the question (e.g. 'calculate', 'explain', 'evaluate')")
    blooms_level: int = Field(description="Cognitive complexity level matching Bloom's taxonomy (1 = Recall, 2 = Understand, 3 = Apply, 4 = Analyze, 5 = Evaluate/Synthesize)")
    reasoning: str = Field(description="Detailed explanation of the estimated difficulty and command complexity")

class DifficultyAnalyzerAgent:
    """Agentic module that analyzes question complexity and command words using LLMs."""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider or os.getenv("EDGENIE_LLM_PROVIDER", "google")
        
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

        logger.info(f"Initialized DifficultyAnalyzerAgent with provider={self.provider}, model={self.model_name}")
        self.structured_llm = self.llm.with_structured_output(DifficultyAnalysisResult)

    def analyze_question(self, question_text: str, marks: int) -> DifficultyAnalysisResult:
        """Analyzes a question to determine command words, cognitive load, and difficulty score."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert CAIE exam designer and examiner.
Analyze the question's text and marks allocation to estimate its cognitive complexity and command word load.

Command Word complexity guidelines (Bloom's Taxonomy):
- Level 1 (Recall): state, define, list, name, identify
- Level 2 (Understand): describe, explain, outline, distinguish
- Level 3 (Apply): calculate, suggest, apply, construct, prepare
- Level 4 (Analyze): analyse, compare, contrast, examine
- Level 5 (Evaluate/Synthesize): evaluate, justify, discuss, assess, to what extent

Difficulty Score Guidelines (1-10):
- Easy (1-3): Simple recall or basic calculation with direct formulas.
- Medium (4-6): Application of formulas, standard bookkeeping ledger entries, multi-step adjustments.
- Hard (7-8): Complex adjustments, detailed financial statement preparation, analyzing and explaining business trends.
- Expert (9-10): Recommending policies, critical evaluation of strategy with calculations, highly ambiguous accounting entries.
"""),
            ("human", "Question Text: {question_text}\nMarks Allocated: {marks}\n\nPerform difficulty analysis:")
        ])

        chain = prompt | self.structured_llm

        try:
            result = chain.invoke({
                "question_text": question_text,
                "marks": marks
            })
            return result
        except Exception as e:
            logger.error(f"Error during question difficulty analysis: {e}")
            # Fallback result to avoid crashing pipeline
            return DifficultyAnalysisResult(
                difficulty_score=5,
                difficulty_label="medium",
                command_words=["explain"],
                blooms_level=2,
                reasoning=f"Error occurred during LLM analysis: {str(e)}"
            )
