# Theory and Expressive Power of Graph Neural Networks — Notes from `10-theory.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Source:** Lecture 10 slides  
**Readings:**  
- Graph Isomorphism Network (GIN)  
- Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks

---

## 1. Why Expressive Power Matters

**Expressive power** of a GNN = its ability to **distinguish different graph structures**.

If two graphs have different structures (and should have different labels) but a GNN produces the **same embedding** for them, it will fail to make correct predictions.

Example (graph classification):
- Two non-isomorphic graphs with different ground-truth labels
- If GNN maps both to the same vector → it cannot separate the classes

→ Understanding expressive power tells us **what GNNs can and cannot represent**.

---

## 2. The Core Question

We have many GNN architectures (GCN, GraphSAGE, GAT, PNA, ...).

Each uses a different function inside the aggregation box.

**Central theoretical questions:**
1. How powerful are existing GNNs at distinguishing graph structures?
2. How do we design a **maximally powerful** GNN?

---

## 3. Local Neighborhood Structures

GNNs build representations from **local neighborhoods**.

Two nodes have the **same local neighborhood structure** if:
- They have the same degree, **and**
- Their neighbors have the same multiset of degrees, **and**
- This continues recursively to further hops.

Examples from lecture:
- Nodes 1 and 5 have different structures (different degrees)
- Nodes 1 and 4 have same degree but different neighbor degree multisets
- Nodes 1 and 2 are symmetric → identical neighborhood structure

**Key question:** Can a GNN produce different embeddings for nodes with different local neighborhood structures?

---

## 4. Computation Graphs = Rooted Subtrees

A K-layer GNN for a node creates a **computation graph** by unfolding the K-hop neighborhood.

This computation graph is exactly a **rooted subtree** (the tree obtained by recursively expanding neighbors from the root).

Important assumption in this analysis:
- **No unique node IDs**
- No continuous node features (or we ignore them for structure comparison)
- We only use **categorical features** (e.g., atom type)
- This forces us to rely purely on **structural** information

Why this assumption?
- With unique IDs or rich features it's trivial to distinguish nodes
- We want to test whether the GNN can distinguish **pure structure**
- Inductive setting: node IDs don't transfer to new graphs

---

## 5. What GNNs Actually See

When building the computation graph:
- GNN only sees the **features** (colors in the slides) and the tree structure
- It does **not** see node IDs

Therefore:
- If two nodes have **identical computation graphs** (same rooted subtree + same features at each position), the GNN **must** produce the same embedding for them.

Example:
- Nodes 1 and 2 have identical 2-hop computation graphs
- All nodes have the same feature
- → Any GNN will give them the **same** embedding

This is not a bug — it's fundamental to how message passing works.

---

## 6. Rooted Subtree View of GNN Power

GNN node embeddings **capture rooted subtree structures**.

A maximally expressive GNN should:
- Map **different** rooted subtrees to **different** embeddings
- Map **identical** rooted subtrees to the **same** embedding

The quality of the GNN is determined by how well its aggregation functions can distinguish different multisets of subtrees.

---

## 7. Connection to the Weisfeiler-Lehman (WL) Test (Preview)

(The next slides in the lecture — and the readings — make this precise.)

The classic 1-dimensional Weisfeiler-Lehman algorithm (1-WL / color refinement) works by:
1. Assigning colors to nodes
2. Iteratively updating a node's color based on the **sorted multiset** of its neighbors' colors
3. If two graphs get different color multisets at the end, they are non-isomorphic

**Key theoretical result** (Xu et al., Morris et al. 2019):
- Most standard GNNs (GCN, GraphSAGE mean/max, GAT, etc.) are **at most as powerful** as the 1-WL test.
- They cannot distinguish graphs that 1-WL cannot distinguish.

To get more power, we need **injective** aggregation functions (the key idea behind GIN).

---

## 8. Mental Model

| Concept                    | What it means                                      | Implication for GNNs |
|---------------------------|----------------------------------------------------|----------------------|
| Local neighborhood structure | Recursive degree/multiset pattern around a node   | What GNNs try to capture |
| Computation graph         | The actual tree the GNN builds for a node          | Identical trees → identical embeddings |
| Rooted subtree            | Computation graph viewed as a tree                 | GNN power = ability to distinguish different trees |
| 1-WL test                 | Classical algorithm for graph isomorphism          | Most GNNs ≤ 1-WL power |
| Maximally expressive      | Can distinguish anything 1-WL can                  | Requires injective aggregation (GIN) |

**Rule of thumb:**
- If two nodes/graphs look the same to 1-WL, standard GNNs will also fail to distinguish them.
- To go beyond 1-WL, we need higher-order GNNs or other extensions (k-WL, subgraph GNNs, etc.).

---

## 9. What Comes Next in This Lecture (from readings + structure)

The rest of the slides (not fully extracted) almost certainly cover:
- Formal proof that common aggregators are not injective
- GIN (Graph Isomorphism Network) — using `sum` + MLP as a universal approximator for multisets
- Conditions for a GNN to be as powerful as 1-WL
- Higher-order GNNs that correspond to k-WL
- Practical implications (when GIN is worth using, limitations of message-passing GNNs)

---

This lecture shifts the conversation from "which GNN architecture works well in practice" to **"what is theoretically possible with message-passing GNNs?"**
