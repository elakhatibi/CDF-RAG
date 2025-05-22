import argparse
from typing import TypedDict
from langgraph.graph import StateGraph, END
from retrieval.query_refiner_rl import QueryRefinementAgent
from retrieval.document_causal_retriever_rl import AdaptiveRetrieverAgent
from models.knowledge_rewriter_rl import KnowledgeRewritingAgent
from models.llm_inference_rl import LLMResponseAgent
from models.hallucination_detector_rl import HallucinationCorrectionAgent


def main(query: str):
    print("Initializing CDF-RAG pipeline with LangGraph and RL")

    from langgraph.graph import StateGraph, END

    class CDFRAGState(TypedDict):
        query: str
        retrieved_docs: list[str]
        causal_docs: list[str]
        rewritten_knowledge: str
        response: str
        is_hallucination: bool

    builder = StateGraph(CDFRAGState)
    builder.add_node("refine_query", QueryRefinementAgent().optimize_query)
    builder.add_node("retrieve_knowledge", AdaptiveRetrieverAgent().retrieve_knowledge)
    builder.add_node("optimize_rewriting", KnowledgeRewritingAgent().optimize_rewriting)
    builder.add_node("generate_validated_response", LLMResponseAgent().generate_validated_response)
    builder.add_node("correct_hallucinations", HallucinationCorrectionAgent().correct_hallucinations)

    builder.set_entry_point("refine_query")
    builder.add_edge("refine_query", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "optimize_rewriting")
    builder.add_edge("optimize_rewriting", "generate_validated_response")
    builder.add_edge("generate_validated_response", "correct_hallucinations")
    builder.add_edge("correct_hallucinations", END)

    graph = builder.compile()

    state = {'query': query}
    final_state = graph.invoke(state)

    print("Final Response:", final_state['response'])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CDF-RAG pipeline with LangGraph and RL")
    parser.add_argument("query", type=str, help="User query")
    args = parser.parse_args()
    main(args.query)
