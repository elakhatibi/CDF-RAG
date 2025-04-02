
import csv
from document_causal_retriever_rl import graph, CDFState, CausalGraphRetriever

# Define your list of test queries
test_queries = [
    "What causes heart disease?",
    "What leads to lung cancer?",
    "Why do people develop diabetes?",
    "What causes poverty?",
    "What are the effects of climate change?",
    "Why does misinformation spread online?",
    "What leads to poor academic performance?",
    "Why do people quit jobs?",
    "What causes homelessness?",
    "How does automation affect employment?"
]

# Output CSV file
output_file = "batch_results.csv"

# Initialize Neo4j connection
graph_obj = CausalGraphRetriever("bolt://localhost:7687", "neo4j", "elahekhatibi")

# Open CSV for writing
with open(output_file, mode="w", newline='') as file:
    writer = csv.DictWriter(file, fieldnames=[
        "query",
        "refinement_type",
        "refined_query_text",
        "refined_query",
        "matched_concept",
        "direct_causes",
        "multi_hop_paths",
        "rewritten_knowledge",
        "final_response",
        "is_hallucination",
        "is_hallucination_gold"  # ✅ Added column for ground truth
    ])

    writer.writeheader()

    for query in test_queries:
        print(f"\n⚙️ Processing query: {query}")

        initial_state = CDFState(
            query=query,
            graph=graph_obj
        )

        result = graph.invoke(initial_state)

        # Logging
        print(f"\n📌 Query: {query}")
        print(f"🔁 Refined Query: {result.get('refined_query')}")
        print(f"💬 Final Answer:\n{result.get('final_response')}")
        print(f"🔧 Refinement Type: {result.get('refinement_type')}")
        print(f"🔁 Refined Query (LLM): {result.get('refined_query_text')}")

        # Assume everything is correct for now — update later if needed
        writer.writerow({
            "query": query,
            "refinement_type": result.get("refinement_type"),
            "refined_query_text": result.get("refined_query_text"),
            "refined_query": result.get("refined_query"),
            "matched_concept": result.get("refined_query"),
            "direct_causes": "; ".join(result.get("retrieved_docs", [])),
            "multi_hop_paths": "; ".join(result.get("causal_docs", [])),
            "rewritten_knowledge": result.get("rewritten_knowledge"),
            "final_response": result.get("final_response"),
            "is_hallucination": "Yes" if result.get("is_hallucination") else "No",
            "is_hallucination_gold": "No"  # ✅ You can change to "Yes" manually later
        })

print(f"\n✅ Evaluation complete. Results saved to: {output_file}")

# Safe shutdown
try:
    graph_obj.close()
except Exception as e:
    print("Warning: Failed to close Neo4j driver gracefully.")


# import csv
# from document_causal_retriever_rl import graph, CDFState, CausalGraphRetriever

# # Define your list of test queries
# test_queries = [
#     "What causes heart disease?",
#     "What leads to lung cancer?",
#     "Why do people develop diabetes?",
#     "What causes poverty?",
#     "What are the effects of climate change?",
#     "Why does misinformation spread online?",
#     "What leads to poor academic performance?",
#     "Why do people quit jobs?",
#     "What causes homelessness?",
#     "How does automation affect employment?"
# ]

# # Output CSV file
# output_file = "batch_results.csv"

# # Initialize Neo4j connection
# graph_obj = CausalGraphRetriever("bolt://localhost:7687", "neo4j", "elahekhatibi")

# # Open CSV for writing
# with open(output_file, mode="w", newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=[
#         "query",
#         "refinement_type",
#         "refined_query_text",
#         "refined_query",
#         "matched_concept",
#         "direct_causes",
#         "multi_hop_paths",
#         "rewritten_knowledge",
#         "final_response",
#         "is_hallucination",
#         "is_hallucination_gold"  # ✅ Added column for ground truth
#     ])

#     writer.writeheader()

#     for query in test_queries:
#         print(f"\n⚙️ Processing query: {query}")

#         initial_state = CDFState(
#             query=query,
#             graph=graph_obj
#         )

#         result = graph.invoke(initial_state)

#         # Logging
#         print(f"\n📌 Query: {query}")
#         print(f"🔁 Refined Query: {result.get('refined_query')}")
#         print(f"💬 Final Answer:\n{result.get('final_response')}")
#         print(f"🔧 Refinement Type: {result.get('refinement_type')}")
#         print(f"🔁 Refined Query (LLM): {result.get('refined_query_text')}")

#         # Assume everything is correct for now — update later if needed
#         writer.writerow({
#             "query": query,
#             "refinement_type": result.get("refinement_type"),
#             "refined_query_text": result.get("refined_query_text"),
#             "refined_query": result.get("refined_query"),
#             "matched_concept": result.get("refined_query"),
#             "direct_causes": "; ".join(result.get("retrieved_docs", [])),
#             "multi_hop_paths": "; ".join(result.get("causal_docs", [])),
#             "rewritten_knowledge": result.get("rewritten_knowledge"),
#             "final_response": result.get("final_response"),
#             "is_hallucination": "Yes" if result.get("is_hallucination") else "No",
#             "is_hallucination_gold": "No"  # ✅ You can change to "Yes" manually later
#         })

# print(f"\n✅ Evaluation complete. Results saved to: {output_file}")

# # Safe shutdown
# try:
#     graph_obj.close()
# except Exception as e:
#     print("Warning: Failed to close Neo4j driver gracefully.")
