# GNN Training — Notes from `05-GNN_training.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Source:** Lecture 5 slides (`05-GNN_training.pdf`)

**Related readings (from the deck):**
- Design Space of Graph Neural Networks
- OGB Datasets

---

## 0. Big picture

We now have a GNN that produces node embeddings $h_v^{(L)}$ after $L$ layers.

**The remaining question:** How do we actually **train** and **evaluate** the full system?

**Full GNN Training Pipeline:**

```
Input Graph → GNN → Node Embeddings → Prediction Head → Predictions
                                                     ↓
                                            Compare with Labels
                                                     ↓
                                            Loss + Metrics
```

The lecture breaks this into 5 practical pieces:
1. Prediction Heads (node / edge / graph)
2. Predictions & Labels (supervised vs self-supervised)
3. Loss Functions
4. Evaluation Metrics
5. Dataset Splits (the tricky graph-specific part)

---

## 1. GNN Prediction Heads

The GNN gives us embeddings. We still need a **head** that turns those embeddings into task-specific predictions.

### 1.1 Node-level prediction

Simplest case:

$$
\hat{y}_v = \text{Head}_\text{node}(h_v^{(L)}) = W^{(H)} h_v^{(L)}
$$

- $W^{(H)} \in \mathbb{R}^{k \times d}$ maps $d$-dimensional embedding to $k$-way output.
- Used for both classification and regression.

### 1.2 Edge-level prediction

We need to predict something about a **pair** of nodes $(u, v)$.

Common heads:

| Method                    | Formula                                      | Best for                  | Notes |
|---------------------------|----------------------------------------------|---------------------------|-------|
| **Concat + Linear**       | $\hat{y}_{uv} = \text{Linear}(\text{Concat}(h_u, h_v))$ | $k$-way edge prediction | Simple and flexible |
| **Dot product**           | $\hat{y}_{uv} = (h_u)^\top h_v$             | 1-way (link existence) | Very common for link prediction |
| **Bilinear**              | $\hat{y}_{uv}^{(k)} = (h_u)^\top W^{(k)} h_v$ then concat | $k$-way | More expressive |

### 1.3 Graph-level prediction

We need one prediction for the **entire graph** using all node embeddings $\{h_v^{(L)} \mid v \in G\}$.

**Naive global pooling** (works okay on small graphs):

$$
\hat{y}_G = \text{Mean / Max / Sum}(\{h_v^{(L)}\})
$$

**Problem with global pooling on large graphs:**

Different graphs can produce the same pooled value even when their structure is very different.

**Example:**
- $G_1$ embeddings: $\{-1, -2, 0, 1, 2\}$ → Sum = 0
- $G_2$ embeddings: $\{-10, -20, 0, 10, 20\}$ → Sum = 0

We lose all information about distribution and structure.

### 1.4 Hierarchical pooling (DiffPool)

Idea: pool **hierarchically**, like CNN pooling, by learning to cluster nodes.

**DiffPool** (one popular method):

At each level $p$ it runs **two GNNs in parallel**:

- **GNN A (embed):** $Z^{(p)} = \text{GNN}_\text{embed}(A^{(p)}, X^{(p)})$
- **GNN B (assignment):** $S^{(p)} = \text{Softmax}(\text{GNN}_\text{pool}(A^{(p)}, X^{(p)}))$

Then it creates the next coarser graph:

$$
X^{(p+1)} = (S^{(p)})^\top Z^{(p)}
$$

$$
A^{(p+1)} = (S^{(p)})^\top A^{(p)} S^{(p)}
$$

Each level produces a smaller "summary graph". Final graph embedding is taken from the coarsest level.

This learns meaningful clusters and preserves more structure than flat global pooling.

---

## 2. Predictions & Labels

### 2.1 Supervised vs Unsupervised / Self-supervised

| Type              | Where labels come from          | Examples |
|-------------------|----------------------------------|----------|
| **Supervised**    | External sources                 | Drug likeness of molecules, node category in citation net, fraud on transactions |
| **Self-supervised** | Generated from the graph itself | Link prediction (hide edges), predict clustering coefficient, graph isomorphism |

Many "unsupervised" graph tasks are actually **self-supervised** — we create supervision signals from the graph structure.

### 2.2 Reducing tasks to node/edge/graph labels

Practical advice: almost any task can be turned into predicting node labels, edge labels, or graph labels.

Example: "These nodes form a cluster" → treat cluster membership as a **node label**.

---

## 3. Loss Functions

### 3.1 Classification (Cross-Entropy)

For $k$-way classification on example $i$:

$$
\text{CE}(y^{(i)}, \hat{y}^{(i)}) = -\sum_{j=1}^{k} y_j^{(i)} \log(\hat{y}_j^{(i)})
$$

- $y^{(i)}$ is one-hot.
- $\hat{y}^{(i)}$ is after softmax.

Total loss:

$$
\mathcal{L} = \sum_{i=1}^{N} \text{CE}(y^{(i)}, \hat{y}^{(i)})
$$

### 3.2 Regression (MSE / L2)

For $k$-way regression:

$$
\text{MSE}(y^{(i)}, \hat{y}^{(i)}) = \sum_{j=1}^{k} (y_j^{(i)} - \hat{y}_j^{(i)})^2
$$

Total loss:

$$
\mathcal{L} = \sum_{i=1}^{N} \text{MSE}(y^{(i)}, \hat{y}^{(i)})
$$

---

## 4. Evaluation Metrics

### 4.1 Regression

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

Both are standard and easy to compute with sklearn.

### 4.2 Classification

**Multi-class:**
- Accuracy (or full confusion matrix)

**Binary classification:**

| Metric                  | Sensitive to threshold? | When to use |
|-------------------------|--------------------------|-------------|
| Accuracy                | Yes                      | Balanced data |
| Precision / Recall      | Yes                      | Imbalanced data, care about positive class |
| **ROC-AUC**             | **No**                   | Overall ranking quality, threshold-independent |

**ROC-AUC intuition:** Probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

**Class imbalance tip:** When positives are rare, focus on **Precision, Recall, and ROC-AUC** rather than accuracy.

---

## 5. Dataset Splits (the graph-specific challenge)

Graphs are **not i.i.d.** — nodes/edges are connected, so data points influence each other.

This makes splitting very different from images or text.

### 5.1 Transductive vs Inductive

| Setting         | Graph(s) in splits                  | What gets split                  | Can generalize to new graphs? |
|-----------------|-------------------------------------|----------------------------------|-------------------------------|
| **Transductive** | One single graph                    | Only the **labels** (or edges)   | No                            |
| **Inductive**    | Multiple separate graphs            | Whole graphs                     | Yes                           |

- **Transductive**: The GNN sees the full graph structure during training, validation, and test. We only hide some labels.
- **Inductive**: Each split gets its own independent graph(s). The model must work on completely unseen graphs.

### 5.2 Node classification splits

- **Transductive**: All nodes are in one graph. Train/Val/Test only differ in which **labels** they can see.
- **Inductive**: Different graphs in each split.

### 5.3 Graph classification splits

Only **inductive** makes sense — you must test on completely unseen graphs.

### 5.4 Link prediction (especially tricky)

Link prediction is self-supervised. We have to **create** the supervision ourselves.

**Two-step edge splitting:**

1. **Message edges** vs **Supervision edges**
   - Message edges → fed into the GNN for message passing.
   - Supervision edges → hidden from the GNN, used only for computing the loss.

2. Split into train / val / test (with special rules).

**Transductive link prediction** (most common):

- Training: Use only training message edges to predict training supervision edges.
- Validation: Add training supervision edges into message passing, predict validation edges.
- Test: Add validation edges too, predict test edges.

This simulates the realistic scenario where, over time, more edges become "known".

---

## 6. Lecture summary

| Topic                     | Key points |
|---------------------------|------------|
| **Prediction heads**      | Node = linear; Edge = concat or dot/bilinear; Graph = pooling (flat or hierarchical like DiffPool) |
| **Labels**                | Can be external (supervised) or generated from graph (self-supervised) |
| **Loss**                  | Cross-entropy (classification), MSE (regression) |
| **Metrics**               | RMSE/MAE (reg), Accuracy / P/R / ROC-AUC (class) |
| **Dataset splits**        | Graphs are not independent → use transductive vs inductive carefully. Link prediction requires message vs supervision edge split |

---

## Quick reference

**Prediction heads:**
- Node: $\hat{y}_v = W h_v$
- Edge (link pred): $\hat{y}_{uv} = h_u^\top h_v$
- Graph (simple): $\hat{y}_G = \text{Mean/Max/Sum}(\{h_v\})$

**Loss:**
- Classification: $\mathcal{L} = \sum -\ y \log \hat{y}$
- Regression: $\mathcal{L} = \sum (y - \hat{y})^2$

**Splits:**
- Transductive: one graph, split labels/edges
- Inductive: multiple graphs, split graphs

---

## Mental model

Training a GNN is not just "run the layers and get embeddings." You still have to:
1. Attach the right **head** for your task level (node/edge/graph).
2. Decide where your **labels** come from (external or self-generated).
3. Choose an appropriate **loss**.
4. Pick metrics that make sense for your label distribution.
5. **Split the data correctly** — this is where most graph-specific mistakes happen.

The graph structure that makes GNNs powerful also makes data splitting and evaluation much more subtle than in vision or NLP.
