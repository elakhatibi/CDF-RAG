import openai
from langgraph.graph import StateGraph
from config import CONFIG

class HallucinationDetector:
    def __init__(self):
        self.model = CONFIG["HALLUCINATION_DETECTION_MODEL"]
    
    def detect(self, response, source_docs):
        """Detect hallucination in the generated response."""
        context = "\n".join(source_docs)
        prompt = f"Verify if the following response aligns with the retrieved context:\nContext:\n{context}\n\nResponse:\n{response}\n\nIs the response factually aligned with the context? (Yes/No)"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "Detect hallucinations in LLM-generated responses."},
                      {"role": "user", "content": prompt}]
        )
        return "no" in response["choices"][0]["message"]["content"].strip().lower()

# RL-Based Hallucination Correction Agent
class HallucinationCorrectionAgent:
    def __init__(self):
        self.detector = HallucinationDetector()
    
    def correct_hallucinations(self, state):
        """Agent detects and corrects hallucinated responses using RL-based feedback loops."""
        state['is_hallucination'] = self.detector.detect(state['response'], state['retrieved_docs'] + state['causal_docs'])
        
        if state['is_hallucination']:
            llm = LLMInference()
            state['response'] = llm.generate(state['rewritten_knowledge'], state['query'], force_retrieval=True)
        
        return state

# LangGraph Workflow
hallucination_graph = StateGraph()
hallucination_graph.add_node("correct_hallucinations", HallucinationCorrectionAgent().correct_hallucinations)
hallucination_graph.set_entry("correct_hallucinations")
hallucination_graph.set_exit("correct_hallucinations")
