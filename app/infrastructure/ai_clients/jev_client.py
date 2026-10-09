import os
from pydantic import BaseModel
from typing import List

class ChoiceResult(BaseModel):
    selected_option: str
    confidence: float

class NoulResult(BaseModel):
    boolean_result: bool
    confidence: float

class JevClient:
    """
    Client for TypeSafe AI's Jev model (System 1).
    Handles ultra-low latency, deterministic routing and boolean evaluations.
    """
    def __init__(self):
        # In a real environment, this gets loaded from the .env file via pydantic-settings
        self.api_key = os.getenv("JEV_API_KEY", "mock-api-key")

    async def choice(self, input_text: str, options: List[str], temperature: float = 0.0) -> ChoiceResult:
        """
        Evaluates the input text and strictly returns one of the provided options.
        Cost: ~$0.042 / 1M input tokens. Output tokens are free.
        """
        # MOCK IMPLEMENTATION for local development.
        # In production, this makes an async HTTP request to the TypeSafe AI API.
        text_lower = input_text.lower()
        
        if "calculate" in text_lower or "math" in text_lower or "dose" in text_lower:
            return ChoiceResult(selected_option="math_calculator", confidence=0.98)
        elif "hi" in text_lower or "hello" in text_lower or "weather" in text_lower:
            return ChoiceResult(selected_option="chitchat", confidence=0.95)
        
        # Default to medical_rag for clinical queries
        return ChoiceResult(selected_option="medical_rag", confidence=0.89)

    async def noul(self, input_text: str) -> NoulResult:
        """
        Evaluates a prompt and returns a boolean Yes/No (True/False).
        Ideal for Hallucination checks (Faithfulness) without generative overhead.
        """
        # MOCK IMPLEMENTATION
        if "hallucination mock" in input_text.lower():
            return NoulResult(boolean_result=True, confidence=0.99)
        
        return NoulResult(boolean_result=False, confidence=0.95)