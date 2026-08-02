# Graph Foundation Models — Notes from `13-gfm.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Lecture:** Graph Foundation Models

---

## 1. Foundation Models Paradigm

Foundation models (successful in NLP, CV, Audio):
- Large-scale **pre-training** on diverse data
- Learn generalizable representations
- Enable strong **transfer** / few-shot / zero-shot performance on downstream tasks

Pre-training paradigm:
- Many unlabeled datasets → self-supervised pre-training → one big model
- Then adapt/generalize to new tasks and new data distributions

**Scaling law** observation: performance keeps improving with model size + data size.

**Question for graphs:** Can we build **Graph Foundation Models**?

---

## 2. Unique Challenges for Graph Foundation Models

Graphs from different domains have very different **semantic meanings**:
- Social networks, molecular graphs, citation networks, e-commerce, etc.
- Node/edge features mean completely different things across domains.

Key open questions:
- What knowledge is actually **transferable** across graph datasets?
- What distribution should a graph foundation model learn?

---

## 3. Structural Patterns as the Universal Transferable Knowledge

**Core thesis of this lecture:**
- **Structural patterns** are domain-agnostic and universal:
  - Small-world property, scale-free degrees, recurring motifs, community structure, etc.
- These patterns exist across almost all real-world graphs.
- Therefore, they can serve as the basis for transferable pre-training.

Idea: Treat **nodes as tokens** and **graph structure as context**.

---

## 4. Structural Self-Supervised Pre-training Tasks

We can pre-train by predicting structural properties (no labels needed):

| Level       | Task                        | Example Supervision                     |
|-------------|-----------------------------|-----------------------------------------|
| Edge        | Shortest-path distance regression | Predict d(u,v)                         |
| Node        | Motif counting              | Count occurrences of small subgraphs around node |
| Node/Graph  | Community detection         | Predict community membership            |
| Graph       | Graph contrastive learning  | Pull augmented views together, push different graphs apart |

These tasks force the model to learn rich structural understanding.

---

## 5. Graph Contrastive Learning (GCL) Details

Goal: Learn representations where **semantically similar** graphs/nodes are close, dissimilar ones are far.

Typical pipeline:
1. Take an anchor graph (or node/subgraph)
2. Apply **data augmentations** → generate positive view(s)
   - Common augmentations: edge dropping, node dropping, random walk sampling, attribute masking, etc.
3. Encode both views with shared encoder
4. Use **InfoNCE loss** (contrastive loss):

$$
\mathcal{L} = -\sum_{i=1}^B \log \frac{\exp(f(x_i)^\top f(x_i^+))}{\sum_j \exp(f(x_i)^\top f(x_j))}
$$

- One positive per anchor
- All other samples in the batch act as negatives

This is the backbone of many graph SSL methods (e.g., GraphCL).

---

## 6. GFSE — Graph Foundational Structural Encoder

### Feature Construction (Random-Walk Based)
- Compute powers of the random-walk transition matrix M = D⁻¹A
- Node features: diagonal entries across powers → P_i = [I, M, M², ..., Mᵈ]_{i,i}
- Edge features: off-diagonal → R_{i,j} = [I, M, M², ..., Mᵈ]_{i,j}

This gives each node/edge a **structural signature** based on how likely random walks reach other nodes in k steps.

### Architecture
- Built on **GPS layers** (hybrid: local message passing + global attention)
- Edge features R are used to **bias the attention scores**:

$$
a'_{i,j} = \text{Softmax}(a_{i,j} + \text{Linear}(R_{i,j}))
$$

This directly injects structural bias into attention.

---

## 7. Expressive Power of GFSE

GFSE is characterized by **RW-SEG-WL** (Random-Walk Structural Encoding Graph Weisfeiler-Lehman).

- RW(d)-SEG-WL is **strictly more powerful than 1-WL** (for d ≥ 3)
- There exist graph pairs that RW(d)-SEG-WL can distinguish but **3-WL cannot**

Pre-training tasks are designed to push the model toward this higher expressivity upper bound.

---

## 8. Pre-training Setup

- Pre-train on **8 datasets from 6 different domains** (molecules, proteins, social networks, academic networks, product networks, super-pixel images, etc.)
- Four self-supervised tasks simultaneously
- Goal: learn a **universal structural encoder** that produces high-quality Positional + Structural Encodings (PSE)

---

## 9. Downstream Usage

GFSE is typically used as a **frozen or fine-tunable structural encoder**:

1. **Vectorized features**: Concatenate GFSE output (PSE) with raw node features → feed to MPNN / Transformer / GPS.
2. **Text-attributed graphs**: Prepend the structural encoding as a "soft token" to LLM input (via MLP projection + LoRA).

This allows even plain Transformers (which have no graph inductive bias) to become very competitive when given good PSE.

---

## 10. Empirical Highlights

- GFSE brings large gains, especially when paired with **Transformers** (which lack structural bias).
- On ZINC, Peptides, MolPCBA etc., GFSE often outperforms LapPE, RWSE, and GPSE.
- On large text-attributed Amazon graphs (Cloth, Home, Sport), using GFSE as structural encoder + LLM outperforms baselines that ignore structure.
- Cross-domain pre-training helps produce higher-quality general structural encodings.

---

## 11. Relational Databases as Graphs + Relational Transformer

Second part of the lecture (Relational Transformer paper):

- A relational DB can be viewed as a **heterogeneous graph**:
  - Every row = a node
  - Foreign-key → Primary-key links = edges
- Many real-world predictive problems on DBs are actually node/edge/graph prediction on this graph.

**Relational Transformer** aims at **zero-shot foundation models** for relational data by treating the entire schema + data as a graph and pre-training across many databases.

(This direction is still emerging.)

---

## 12. Key Takeaways

- Structural patterns are the most promising "universal language" for graph foundation models.
- Self-supervised structural tasks + contrastive learning are the main pre-training tools.
- GFSE demonstrates that a single structural encoder pre-trained across domains can provide powerful, transferable **Positional + Structural Encodings**.
- These encodings dramatically boost downstream models (especially Transformers and LLMs on text-attributed graphs).
- The field is moving from "train one GNN per dataset" toward "one structural foundation model to rule them all."

---

This lecture represents the frontier: moving graph ML from task-specific models toward general-purpose, large-scale pre-trained graph models.
