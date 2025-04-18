import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load results
df = pd.read_csv("batch_results.csv")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Embedding function (robust)
def embed(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(384)
    return encoder.encode(text.strip(), normalize_embeddings=True)

# Prepare lists for metrics
semantic_scores = []
context_scores = []
grounded_scores = []
final_lengths = []
ccd_values = []
has_causal = []
hallucinations = []

# Compute per-query metrics
for _, row in df.iterrows():
    q = str(row.get("query", "") or "")
    rq = str(row.get("refined_query_text", "") or "")
    final = str(row.get("final_response", "") or "")
    knowledge = str(row.get("rewritten_knowledge", "") or "")
    retrieved = str(row.get("direct_causes", "") or "") + " " + str(row.get("multi_hop_paths", "") or "")

    # Embed
    q_emb = embed(q)
    rq_emb = embed(rq)
    final_emb = embed(final)
    know_emb = embed(knowledge)
    ret_emb = embed(retrieved)

    # Compute metrics
    semantic_scores.append(cosine_similarity([q_emb], [rq_emb])[0][0])
    context_scores.append(cosine_similarity([q_emb], [know_emb])[0][0])
    grounded_scores.append(cosine_similarity([know_emb], [ret_emb])[0][0])
    final_lengths.append(len(final.split()))
    has_causal.append(1 if retrieved.strip() else 0)

    if isinstance(row["multi_hop_paths"], str) and row["multi_hop_paths"].strip():
        paths = [p.split(" → ") for p in row["multi_hop_paths"].split(";")]
        path_lengths = [len(p) for p in paths]
        ccd_values.append(np.mean(path_lengths))
    else:
        ccd_values.append(0)

    hallucinations.append(1 if row.get("is_hallucination", "No").strip().lower() == "yes" else 0)

# Add computed metrics to dataframe
df["SRS"] = semantic_scores
df["Context_Relevance"] = context_scores
df["Groundedness"] = grounded_scores
df["Final_Answer_Length"] = final_lengths
df["CCD"] = ccd_values
df["Has_Causal"] = has_causal
df["Hallucinated"] = hallucinations

# Save updated CSV
df.to_csv("batch_results_with_metrics.csv", index=False)

# Aggregate metrics
HR = np.mean(hallucinations)
CRC = np.mean(has_causal)
CCD = np.mean(ccd_values)
SRS = np.mean(semantic_scores)
ContextRel = np.mean(context_scores)
Grounded = np.mean(grounded_scores)
AnswerLen = np.mean(final_lengths)

# Print general evaluation report
print("\n🧪 Evaluation Metrics Report")
print(f"Causal Retrieval Coverage (CRC):    {CRC:.2%}")
print(f"Causal Chain Depth (CCD):           {CCD:.2f}")
print(f"Semantic Refinement Score (SRS):    {SRS:.3f}")
print(f"Context Relevance Score:            {ContextRel:.3f}")
print(f"Groundedness Score:                 {Grounded:.3f}")
print(f"Avg. Final Answer Length:           {AnswerLen:.1f} words")
print(f"Hallucination Rate (HR):            {HR:.2%}")

# ============================
# ✅ Classification Evaluation
# ============================
df.columns = [col.strip().lower() for col in df.columns]

if "is_hallucination_gold" in df.columns:
    y_true = df["is_hallucination_gold"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    y_pred = df["hallucinated"]

    total_pos = y_true.sum()
    total_pred = y_pred.sum()

    if total_pos == 0 and total_pred == 0:
        print("\n⚠️ Skipped precision/recall/F1 — no positive samples in gold labels or predictions.")
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        print("\n📊 Classification Metrics (vs. Ground Truth)")
        print(f"Accuracy:                     {acc:.3f}")
        print(f"Precision:                    {precision:.3f}")
        print(f"Recall:                       {recall:.3f}")
        print(f"F1 Score:                     {f1:.3f}")
else:
    print("\n⚠️ Column 'is_hallucination_gold' not found. Add it to compute F1, Precision, Recall, Accuracy.")
