import pandas as pd
from your_cdf_module import CDFPipeline  # <-- replace with actual import

# Queries to evaluate across configurations
queries = [
    "Why do people quit jobs?",
    "What leads to poor academic performance?",
    "Why is climate change accelerating?",
    "Why do patients miss appointments?",
    "What causes obesity?"
]

# Configurations (stepwise ablation)
cfgs = [
    dict(name="Baseline RAG", use_rl=False, use_graph=False, use_rewriter=False, use_verifier=False),
    dict(name="+ RL-based Query Refinement", use_rl=True, use_graph=False, use_rewriter=False, use_verifier=False),
    dict(name="+ Causal Graph", use_rl=True, use_graph=True, use_rewriter=False, use_verifier=False),
    dict(name="+ Rewriter", use_rl=True, use_graph=True, use_rewriter=True, use_verifier=False),
    dict(name="+ Hallucination Correction", use_rl=True, use_graph=True, use_rewriter=True, use_verifier=True),
]

# Run ablation study
results = []
for cfg in cfgs:
    print(f"Running config: {cfg['name']}")
    pipeline = CDFPipeline(
        use_rl=cfg["use_rl"],
        use_graph=cfg["use_graph"],
        use_rewriter=cfg["use_rewriter"],
        use_verifier=cfg["use_verifier"]
    )
    metrics = {
        "CRC": [], "CCD": [], "SRS": [], "Groundedness": [],
        "HR": [], "F1": []
    }

    for q in queries:
        output = pipeline.run(q)  # expected to return a dict of metrics
        metrics["CRC"].append(output["crc"])
        metrics["CCD"].append(output["ccd"])
        metrics["SRS"].append(output["srs"])
        metrics["Groundedness"].append(output["groundedness"])
        metrics["HR"].append(output["hallucinated"])
        metrics["F1"].append(output["f1"])

    # Average across queries
    results.append({
        "Stage": cfg["name"],
        "CRC": round(sum(metrics["CRC"]) / len(metrics["CRC"]), 3),
        "CCD": round(sum(metrics["CCD"]) / len(metrics["CCD"]), 3),
        "SRS": round(sum(metrics["SRS"]) / len(metrics["SRS"]), 3),
        "Groundedness": round(sum(metrics["Groundedness"]) / len(metrics["Groundedness"]), 3),
        "HR": round(sum(metrics["HR"]) / len(metrics["HR"]), 3),
        "F1": round(sum(metrics["F1"]) / len(metrics["F1"]), 3),
    })

# Save as CSV for plotting
df = pd.DataFrame(results)
df.to_csv("cdf_ablation_metrics.csv", index=False)
print(df)
