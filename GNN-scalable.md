# Scaling Up GNNs — Notes from `06-scalable.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Source:** Lecture 6 slides (`06-scalable.pdf`)

**Readings mentioned:**
- GraphSAINT
- GNN AutoScale

---

## 0. Why do we need to scale GNNs?

Modern graphs are **huge**:

| Domain            | Nodes          | Edges           | Typical Tasks                     |
|-------------------|----------------|-----------------|-----------------------------------|
| Recommender       | 100M–1B users  | 100M–100B       | Link prediction, node classification |
| Social networks   | 300M–3B users  | Massive         | Friend recommendation             |
| Academic graphs   | 120M papers    | Huge            | Paper categorization, citations   |
| Knowledge Graphs  | 80M–90M entities | Very large    | KG completion                     |

**Core problem:**  
Standard deep learning uses mini-batch SGD.  
In GNNs, computing one node's embedding requires its **entire K-hop neighborhood**.

Naive mini-batching breaks this because sampled nodes are usually isolated from each other.

---

## 1. Why full-batch and standard SGD fail

### Full-batch training (the naive way)

1. Load the **entire graph + features** into memory.
2. At every layer: compute embeddings for **all** nodes using all previous embeddings.
3. Use sparse matrix ops (e.g., for GraphSAGE: `H^{(l+1)} = σ(Ã H^{(l)} W^T + H^{(l)} B^T)`).

**Fatal problem:**  
GPU memory is only ~10–80 GB.  
Large graphs easily require hundreds of GB (graph + features + activations + gradients).

→ Full-batch training does **not fit** on GPU.

### Standard mini-batch SGD also fails

If you just randomly sample M nodes as a mini-batch:
- Sampled nodes are usually **disconnected**.
- GNNs need neighbors to aggregate → the batch has no context.

**Conclusion:** We need **special mini-batching strategies** for GNNs.

---

## 2. Method 1: Neighbor Sampling (GraphSAGE)

**Paper:** Hamilton, Ying, Leskovec — NeurIPS 2017

### Core idea

Instead of using the full K-hop neighborhood, **sample a fixed number of neighbors** at each hop.

This prunes the computation graph so it stays manageable.

### Computational graph view

A K-layer GNN for a node v builds a tree of depth K (the computation graph).

Without sampling:
- Tree grows **exponentially** with K.
- "Hub nodes" (high-degree nodes) cause explosion.

With neighbor sampling:
- At hop k, sample at most **n_k** neighbors per node.
- Final leaf nodes in the tree ≤ ∏ n_k (much smaller).

### How the algorithm works

For a mini-batch of target nodes:

For k = 1 to K:
  For every node in the current k-hop set:
    Randomly sample at most n_k of its neighbors.

The sampled nodes + their features form a small subgraph that is loaded onto the GPU.

### Sampling strategies

| Strategy                    | Description                                      | Pros                     | Cons                          |
|----------------------------|--------------------------------------------------|--------------------------|-------------------------------|
| **Uniform random**         | Pick n_k neighbors uniformly                     | Fast                     | Can miss important nodes      |
| **Random Walk with Restarts** | Score nodes by RWR from the target, sample highest scores | Better quality          | Slightly slower               |

### Trade-offs (very important)

- **Smaller n_k** → faster, less memory, but **higher variance** in gradients.
- Computation graph size still grows **exponentially** with number of layers (even with sampling).
- Adding one more GNN layer multiplies cost by ~n_k.

**Typical numbers in practice:** n_k = 10~25 per layer.

---

## 3. Method 2: Cluster-GCN

**Paper:** Chiang et al. — KDD 2019

### Core idea

Instead of sampling neighbors, **pre-partition the graph into clusters** (subgraphs), then treat each cluster as a mini-batch.

### How it works

1. Run a graph clustering algorithm (e.g., METIS) to partition the graph into many small clusters.
2. In each training step, pick one or more clusters.
3. The subgraph induced by the cluster becomes the mini-batch.
4. Run full GNN message passing **inside** the cluster.

### Advantages

- Much more regular computation (no irregular sampling).
- Good for very deep GNNs (no exponential blowup within a cluster).
- Can control cluster size to fit GPU memory.

### Disadvantages

- Loses some cross-cluster edges (information loss at boundaries).
- Clustering must be done upfront (can be expensive).
- Clusters may be unbalanced.

Often used together with techniques to add some cross-cluster edges back.

---

## 4. Method 3: Simplified GCN (SGC)

**Paper:** Wu et al. — ICML 2019

### Core idea

**Simplify** the GNN so that it becomes a linear feature transformation + a fixed graph propagation.

Remove:
- Nonlinearities between layers
- Learnable weights per layer (in the propagation part)

The model reduces to:

$$
\hat{Y} = \text{softmax}( \tilde{A}^K X W )
$$

Where:
- $\tilde{A}$ is the normalized adjacency (with self-loops)
- $K$ is the number of propagation steps (like depth)
- $W$ is a single weight matrix

### Intuition

All the "graph convolution" can be pre-computed:
1. Pre-multiply features by $\tilde{A}^K$ (this can be done once, even on CPU).
2. Then just do a linear classifier (or small MLP) on the pre-propagated features.

### Pros

- Extremely fast.
- Very memory efficient.
- Often surprisingly competitive on many datasets.

### Cons

- Loses expressivity (no per-layer learned transformations, no nonlinearities during propagation).
- Cannot easily handle some tasks that need deep non-linear feature mixing.

SGC is a great **baseline** and is excellent when speed/memory is critical.

---

## 5. Other methods mentioned in readings

- **GraphSAINT**: Samples subgraphs (node/edge/walk based) for mini-batches. Good variance reduction and scalability.
- **GNN AutoScale**: Maintains historical embeddings of nodes and only updates a small set of nodes per batch (reduces neighbor explosion).

These are more advanced techniques that build on the ideas of sampling and clustering.

---

## 6. Comparison of Scaling Methods

| Method             | Key Idea                          | Memory per batch | Handles deep GNNs? | Training stability | Implementation complexity |
|--------------------|-----------------------------------|------------------|--------------------|--------------------|---------------------------|
| **Neighbor Sampling** | Sample fixed # neighbors per hop | Low–Medium      | No (still exponential) | Can be unstable (high variance) | Medium |
| **Cluster-GCN**    | Pre-cluster graph                 | Controlled by cluster size | Better             | More stable        | Medium (needs good clustering) |
| **Simplified GCN** | Pre-propagate + linear            | Very low         | Yes (cheap)        | Very stable        | Very simple |
| **GraphSAINT**     | Sample subgraphs                  | Low              | Better             | Good               | Medium-High |

---

## 7. Mental model & practical tips

**Key insight from the lecture:**

> In GNNs, the data points are **not independent** — every node depends on its neighborhood.  
> Therefore we must redesign mini-batching around **graph structure**, not just random samples.

### Practical rules of thumb

- For **very large single graphs** (millions to billions of nodes) → use Neighbor Sampling or GraphSAINT.
- When you want **deep GNNs** with less variance → consider Cluster-GCN.
- When you need **maximum speed / minimum memory** and the task is simple → try SGC first.
- Always monitor **variance** of the loss when using sampling methods. If training is unstable, increase sampling size or use better samplers (RWR, etc.).
- Pre-compute as much as possible (normalization, multi-hop propagation for SGC).

---

## Quick Reference Cheat Sheet

**Neighbor Sampling formula (conceptual):**
- At each layer, sample ≤ n_k neighbors instead of all.

**Cluster-GCN:**
- Partition → induce subgraph → full message passing inside cluster.

**SGC:**
$$
H = \tilde{A}^K X \quad \rightarrow \quad \hat{Y} = \text{softmax}(H W)
$$

**Main bottlenecks to watch:**
- Exponential growth with depth
- Hub nodes
- GPU memory for activations/gradients

---

This lecture focuses on making GNN training feasible on real-world scale graphs by rethinking how we form mini-batches and how much computation we actually need per update.
