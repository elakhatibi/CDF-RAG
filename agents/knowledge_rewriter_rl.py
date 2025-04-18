import openai
from langgraph.graph import StateGraph
from pydantic import BaseModel
from config import CONFIG
from dotenv import load_dotenv
import os

# ✅ Load environment variables (if using .env for keys)
load_dotenv()

# ✅ Define schema for LangGraph state
class RewritingState(BaseModel):
    query: str
    retrieved_docs: list[str]
    causal_docs: list[str]
    rewritten_knowledge: str = ""

# ✅ LLM wrapper for GPT-based knowledge structuring

import openai
import os
from openai import OpenAI

class KnowledgeRewriter:
    def __init__(self):
        self.model = CONFIG.get("KNOWLEDGE_REWRITING_MODEL", "gpt-4")
        self.api_key = CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def rewrite(self, documents: list[str], query: str) -> str:
        context = "\n".join(documents)
        prompt = f"""
You are a medical reasoning assistant. Your task is to synthesize the following evidence and rewrite it into a coherent explanation that answers the query:

Query: {query}

Evidence:
{context}

Instructions:
- Group related causes together.
- Clarify any multi-hop cause-effect chains.
- Use medically accurate language.
- Return a well-structured explanation (2–3 paragraphs).
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful medical explanation assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()


# ✅ Agent class for integration into LangGraph
class KnowledgeRewritingAgent:
    def __init__(self):
        self.rewriter = KnowledgeRewriter()

    def optimize_rewriting(self, state: RewritingState) -> RewritingState:
        combined_docs = state.retrieved_docs + state.causal_docs
        structured_summary = self.rewriter.rewrite(combined_docs, state.query)
        state.rewritten_knowledge = structured_summary
        return state

# ✅ LangGraph flow definition
rewriting_graph = StateGraph(RewritingState)
rewriting_graph.add_node("optimize_rewriting", KnowledgeRewritingAgent().optimize_rewriting)
graph = rewriting_graph.set_entry_point("optimize_rewriting").set_finish_point("optimize_rewriting").compile()

# ✅ Debug runner (for standalone testing)
if __name__ == "__main__":
    test_state = RewritingState(
        query="What causes heart disease?",
        retrieved_docs=[
            "Hypercholesterolemia causes heart disease.",
            "Obstructive sleep apnea contributes to cardiovascular conditions."
        ],
        causal_docs=[
            "Hypertension is a common intermediary between stress and heart disease.",
            "Smoking leads to atherosclerosis, which leads to heart disease."
        ]
    )
    final_state = graph.invoke(test_state)
    print("\n🔹 Final Structured Explanation:\n")
    print(final_state.get("rewritten_knowledge"))

