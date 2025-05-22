import openai
from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import CONFIG
from models.llm_inference_rl import LLMInference  # Needed for regeneration

class HallucinationState(TypedDict):
    query: str
    retrieved_docs: list[str]
    causal_docs: list[str]
    rewritten_knowledge: str
    response: str
    is_hallucination: bool

class HallucinationDetector:
    def __init__(self):
        self.model = CONFIG["HALLUCINATION_DETECTION_MODEL"]
    
    def detect(self, response, source_docs):
        context = "\n".join(source_docs)
        prompt = f"Verify if the following response aligns with the retrieved context:\nContext:\n{context}\n\nResponse:\n{response}\n\nIs the response factually aligned with the context? (Yes/No)"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Detect hallucinations in LLM-generated responses."},
                {"role": "user", "content": prompt}
            ]
        )
        return "no" in response["choices"][0]["message"]["content"].strip().lower()

class HallucinationCorrectionAgent:
    def __init__(self):
        self.detector = HallucinationDetector()
    
    def correct_hallucinations(self, state: HallucinationState) -> HallucinationState:
        state['is_hallucination'] = self.detector.detect(state['response'], state['retrieved_docs'] + state['causal_docs'])
        if state['is_hallucination']:
            llm = LLMInference()
            state['response'] = llm.generate(state['rewritten_knowledge'], state['query'], force_retrieval=True)
        return state

def build_hallucination_graph():
    builder = StateGraph(HallucinationState)
    builder.add_node("correct_hallucinations", HallucinationCorrectionAgent().correct_hallucinations)
    builder.set_entry_point("correct_hallucinations")
    builder.add_edge("correct_hallucinations", END)
    return builder.compile()
