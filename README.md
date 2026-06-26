# RANGER: Repository-level Agent for Graph-Enhanced Retrieval

[![Paper](https://img.shields.io/badge/KDD%202026-Agentic%20SE%203.0%20Workshop-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets%20%26%20Trajectories-yellow)](https://huggingface.co/collections/Nutanix/mcts)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **RANGER** is a repository-level code retrieval agent that constructs a knowledge graph from Python repositories and uses a dual-stage retrieval pipeline — combining efficient Cypher lookups for code-entity queries with MCTS-guided graph traversal for natural language queries.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/Neo4j-5.26-green?logo=neo4j" alt="Neo4j 5.26"/>
  <img src="https://img.shields.io/badge/tree--sitter-AST%20Parsing-orange" alt="tree-sitter"/>
</p>

---

## Overview

Existing code retrieval systems either treat repositories as flat collections of files (losing structural relationships) or rely on computationally expensive full-graph neural methods. **RANGER** bridges this gap by:

1. **Offline Indexing**: Parsing Python repositories via tree-sitter AST into a typed knowledge graph (7 node types, 5 edge types) stored in Neo4j, enriched with LLM-generated semantic descriptions and embeddings.

2. **Online Retrieval** via a dual-stage agent:
   - **Code-Entity Queries** (e.g., "What methods does `Calculator` use?") → LLM-generated Cypher queries for direct graph lookup
   - **Natural Language Queries** (e.g., "Where is authentication handled?") → MCTS-guided graph traversal with bi-encoder expansion and cross-encoder scoring

```
┌──────────────────────────────────────────────────────────────┐
│                    RANGER Architecture                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌──────────────────────────────────┐   │
│  │  User Query  │────▶│  LLM → Cypher Query Generation  │   │
│  └─────────────┘     └──────────┬───────────────────────┘   │
│                                 │                            │
│                      ┌──────────▼──────────┐                │
│                      │ Cypher returns results? │             │
│                      └──────┬─────────┬────┘                │
│                     Yes     │         │    No                │
│                 ┌───────────▼┐    ┌───▼──────────────┐      │
│                 │ Path 1:    │    │ Path 2:           │      │
│                 │ Direct     │    │ MCTS Graph        │      │
│                 │ Cypher     │    │ Traversal         │      │
│                 │ Lookup     │    │ (Bi-encoder +     │      │
│                 │            │    │  Cross-encoder)   │      │
│                 └────────────┘    └──────────────────┘      │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Neo4j Knowledge Graph                        │  │
│  │  Repo → Module → {Class, Function, GlobalVariable}    │  │
│  │  Class → {Method, Field}; USES, INHERITS edges        │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Benchmark | Task | Metric | RANGER | Best Baseline |
|-----------|------|--------|--------|---------------|
| **CodeSearchNet** | NL → Code Retrieval | NDCG@10 | **0.786** | 0.725 |
| **RepoQA** | Needle Function Search | NDCG@10 | **0.741** | 0.722 |
| **RepoBench** | Cross-file Completion | Acc@5 | **0.547** | 0.494 |
| **CrossCodeEval** | Code Completion | Exact Match | **36.27** | 28.57 |

---

## Repository Structure

This repository contains 4 benchmark-specific implementations, each self-contained with setup instructions:

| Directory | Benchmark | Task Type | README |
|-----------|-----------|-----------|--------|
| [`ai-hpc-mcts-codesearchnet/`](ai-hpc-mcts-codesearchnet/) | CodeSearchNet | NL → Code Retrieval | [README](ai-hpc-mcts-codesearchnet/README.md) |
| [`ai-hpc-mcts-repoqa/`](ai-hpc-mcts-repoqa/) | RepoQA | Needle Function Search | [README](ai-hpc-mcts-repoqa/README.md) |
| [`ai-hpc-mcts-repobench/`](ai-hpc-mcts-repobench/) | RepoBench | Cross-file Code Completion | [README](ai-hpc-mcts-repobench/README.md) |
| [`ai-hpc-mcts-crosscodeeval/`](ai-hpc-mcts-crosscodeeval/) | CrossCodeEval | Cross-file Code Completion | [README](ai-hpc-mcts-crosscodeeval/README.md) |

Each sub-directory follows a shared structure:
```
benchmark-repo/
├── config.yaml              # Configuration (Neo4j, models, MCTS hyperparameters)
├── requirements.txt         # Python dependencies
├── README.md                # Benchmark-specific setup & usage
├── src/
│   ├── core/
│   │   ├── generate_graph.py              # Knowledge graph construction (tree-sitter AST)
│   │   ├── MCTS_cross_encoder_batch.py    # MCTS algorithm (NL query retrieval)
│   │   ├── retirever_v2.py                # Cypher query generation + reranking
│   │   ├── add_embeddings_final.py        # LLM description + embedding generation
│   │   └── semantic_retriever.py          # Flat vector search baseline
│   └── utils/                             # Prompts, schemas, config loaders
├── scripts/                               # Evaluation entry points
└── experiments/                           # Result files
```

---

## Prerequisites

- **Python** 3.9+
- **Neo4j** 5.26.0 with [APOC](https://neo4j.com/labs/apoc/) and [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/) plugins
- **Java** 17 (for Neo4j)
- **GPU** recommended for cross-encoder inference

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/shahpratik02/RANGER.git
   cd RANGER
   ```

2. **Set up Neo4j** — Follow the [Neo4j installation guide](https://neo4j.com/docs/operations-manual/current/installation/) and install the APOC + GDS plugins.

3. **Configure** — Update `config.yaml` in the benchmark directory with your Neo4j credentials and paths.

4. **Choose a benchmark** — Navigate to the benchmark directory and follow its `README.md`:
   ```bash
   cd ai-hpc-mcts-codesearchnet
   pip install -r requirements.txt
   # Follow README.md for data download, graph setup, and evaluation
   ```

---

## Pre-built Graph Dumps & Data

We provide pre-built Neo4j graph dumps and MCTS trajectory data on HuggingFace:

| Dataset | Description | Link |
|---------|-------------|------|
| CodeSearchNet-neo4j | Pre-built knowledge graph | [🤗 Nutanix/CodeSearchNet-neo4j](https://huggingface.co/datasets/Nutanix/CodeSearchNet-neo4j) |
| RepoQA-neo4j | Pre-built knowledge graph | [🤗 Nutanix/RepoQA-neo4j](https://huggingface.co/datasets/Nutanix/RepoQA-neo4j) |
| RepoBench-neo4j | Pre-built knowledge graph | [🤗 Nutanix/RepoBench-neo4j](https://huggingface.co/datasets/Nutanix/RepoBench-neo4j) |
| CrossCodeEval-neo4j | Pre-built knowledge graph | [🤗 Nutanix/CrossCodeEval-neo4j](https://huggingface.co/datasets/Nutanix/CrossCodeEval-neo4j) |
| MCTS_Trajectory_CSN | MCTS search trajectories (CodeSearchNet) | [🤗 Nutanix/MCTS_Trajectory_CSN](https://huggingface.co/datasets/Nutanix/MCTS_Trajectory_CSN) |
| MCTS_Trajectory_RepoQA | MCTS search trajectories (RepoQA) | [🤗 Nutanix/MCTS_Trajectory_RepoQA](https://huggingface.co/datasets/Nutanix/MCTS_Trajectory_RepoQA) |

Browse all resources: [🤗 Nutanix MCTS Collection](https://huggingface.co/collections/Nutanix/mcts)

---

## Models Used

| Component | Model | Parameters |
|-----------|-------|------------|
| Query Embedding (MCTS) | [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | 335M |
| Cross-Encoder (MCTS Reward) | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 568M |
| Code Description Generator | [deepseek-ai/deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) | 1.3B |
| Cypher Query Generator | [Meta-Llama-3.1-70B-Instruct-AWQ](https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4) | 70B (INT4) |

---

## Citation

If you use RANGER in your research, please cite:

```bibtex
@inproceedings{ranger2026,
  title     = {RANGER: Repository-level Agent for Graph-Enhanced Retrieval},
  author    = {Shah, Pratik and Bhatele, Abhinav},
  booktitle = {Proceedings of the Agentic Software Engineering (SE 3.0) Workshop 
               at KDD 2026},
  year      = {2026},
  url       = {https://github.com/shahpratik02/RANGER}
}
```

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
