import os
from openai import OpenAI
from pydantic import BaseModel
from config import CONFIG
from dotenv import load_dotenv

load_dotenv()

# ✅ Set up OpenAI client
class LLMResponseGenerator:
    def __init__(self):
        self.model = CONFIG.get("LLM_MODEL", "gpt-4")
        self.client = OpenAI(api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))

    def generate(self, knowledge: str, original_query: str) -> str:
        prompt = f"""
You are a helpful and domain-agnostic assistant. Read the following structured explanation and generate a final answer to the original query.

Query: {original_query}

Structured Knowledge:
{knowledge}

Instructions:
- Do NOT mention you're limited to medical topics
- Provide a confident, natural answer for ANY domain
- Use a clear, user-friendly tone in 1–2 paragraphs
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medically accurate and user-friendly assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()
