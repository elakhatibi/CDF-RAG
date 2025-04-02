# CDF-RAG: Causal Dynamic Feedback Retrieval-Augmented Generation


CDF-RAG is a novel framework for Retrieval-Augmented Generation (RAG) that incorporates causal graph reasoning, reinforcement-learned query refinement, and hallucination correction. Designed for complex multi-hop question answering, especially in high-stakes domains such as healthcare and science.

---

## 🔧 Key Features

- ✅ **Causal Graph Retrieval** using Neo4j with verified multi-hop chains.
- ✅ **RL-based Query Refinement** using PPO to adapt queries for better retrieval.
- ✅ **Hybrid Context Fusion** of graph and dense text retrieval.
- ✅ **Entailment-aware Generation** with hallucination detection.
- ✅ **Evaluation on 4 Benchmarks**: AdversarialQA, CosmosQA, MedQA, MedMCQA.

---

## 🏗️ Project Structure

cdf-rag/ ├── src/ # Core modules (retriever, generator, rl, graph) ├── scripts/ # Training and evaluation entry points ├── configs/ # YAML configs for experiments ├── notebooks/ # Jupyter demos ├── data/ # Placeholder for datasets ├── requirements.txt ├── README.md └── LICENSE


---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/cdf-rag.git
cd cdf-rag
pip install -r requirements.txt
