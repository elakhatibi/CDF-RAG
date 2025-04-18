from openai import OpenAI
import os
from config import CONFIG
from dotenv import load_dotenv

load_dotenv()

class CausalEnricher:
    def __init__(self):
        self.model = CONFIG.get("LLM_MODEL", "gpt-4")
        self.client = OpenAI(api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))

    
    def generate_pairs(self, concept: str) -> list[tuple[str, str]]:
        prompt = f"""
You are an expert in causal reasoning. Provide 3–5 high-quality (cause → effect) pairs for this concept:
"{concept}"

Output format: one pair per line, exactly two items per line separated by a comma.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You generate high-quality cause-effect relationships."},
                {"role": "user", "content": prompt}
            ]
        )

        lines = response.choices[0].message.content.strip().split("\n")
        pairs = []

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                pairs.append(tuple(parts))
            else:
                print(f"⚠️ Skipping malformed line: {line}")
        
        return pairs

