import os
from openai import OpenAI
from pydantic import BaseModel
from config import CONFIG
from dotenv import load_dotenv

load_dotenv()

class HallucinationDetector:
    def __init__(self):
        self.model = CONFIG.get("HALLUCINATION_DETECTION_MODEL", "gpt-4")
        self.client = OpenAI(api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))

    def detect(self, response: str, structured_explanation: str) -> bool:
        prompt = f"""
You are a hallucination detection model. Determine whether the following response accurately reflects the given structured explanation.

Structured Explanation:
{structured_explanation}

Generated Response:
{response}

Answer with only "Yes" or "No":
- Yes → response is factually aligned
- No → response includes hallucinated content
"""

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in hallucination detection."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = result.choices[0].message.content.lower().strip()
        return "no" in answer
