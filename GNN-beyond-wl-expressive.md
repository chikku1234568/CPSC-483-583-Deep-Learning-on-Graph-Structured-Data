# Beyond WL: More Expressive GNNs — Notes from `11-expressive-gnns.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Lecture:** Beyond WL: More Expressive GNNs

**Readings referenced:**
- Position-aware Graph Neural Networks
- Identity-aware Graph Neural Networks

---

## 1. Recap: Computation Graphs in Message-Passing GNNs

- A node's embedding is produced by recursively aggregating its local neighborhood.
- This creates a **computation graph** (equivalent to a rooted subtree).
- A "perfect" GNN would be an **injective** mapping from computation graph → embedding.
- Problem: Many nodes with **different** structural roles can end up with **identical** computation graphs.

---

## 2. Two Fundamental Limitations of Standard Message-Passing GNNs

### Limitation 1: Cannot Capture Positional Information
- Two nodes can have **identical neighborhood structure** (same computation graph) but appear in **different positions** in the overall graph.
- Example: v1 and v2 are symmetric locally but sit on opposite sides of the graph.
- Standard GNNs will give them the **same** embedding even when the task requires distinguishing positions.

### Limitation 2: Upper Bounded by the WL Test
- Different global structures can produce identical local computation graphs.
- Classic example: GNNs cannot count cycle length.
  - A node in a triangle and a node in a 4-cycle can have the same unfolding.
- Result: message-passing GNNs cannot distinguish graphs that the 1-WL test cannot distinguish.

---

## 3. Two Types of Graph Tasks

| Task Type             | What determines the label?          | Standard GNNs? | Example |
|-----------------------|-------------------------------------|----------------|---------|
| **Structure-aware**   | Local neighborhood / role           | Usually good   | Node v1 vs v2 because their degrees or neighbor degrees differ |
| **Position-aware**    | Absolute position in the graph      | Often fail     | v1 and v2 have identical local structure but different positions |

**Key insight:** Structure-aware embeddings are **not sufficient** for position-aware tasks.

---

## 4. Naive Fix (One-Hot IDs) Is Not Practical

- Assign every node a unique one-hot ID.
- Then computation graphs become trivially different.
- **Problems:**
  - O(N) dimensions → not scalable
  - Not inductive (IDs don't transfer to new graphs or new nodes)
- **Goal:** Need a **scalable + inductive** way to inject position or identity.

---

## 5. Position-Aware GNNs (P-GNN)

### Core Idea
Break symmetry by giving nodes **coordinates** relative to a small set of reference nodes called **anchor sets**.

### Anchors and Anchor-Sets
- Pick one or more nodes as "anchors".
- Represent every other node by its **shortest-path distance** to the anchor(s).
- Using multiple anchors creates a richer "coordinate system".
- Generalize single anchors → **anchor-sets** (distance = min distance to any node in the set).

### Theory: Bourgain Theorem
- Any finite metric space can be embedded into Euclidean space with low distortion using O(log² n) dimensions.
- The embedding uses distances to randomly sampled subsets (anchor-sets).
- P-GNN follows this construction.

### How P-GNN Works (high level)
1. Sample a small number of anchor-sets (S1, S2, ...)
2. For each node v, compute relative distances d(v, Si) to each anchor-set
3. Use these distances to form a **position encoding**
4. Combine with normal message passing, but now the embeddings are position-aware

### Inductiveness
- Anchor sets are **re-sampled** at every training step and at test time on new graphs.
- No fixed node IDs → works on unseen graphs.

---

## 6. Position-Aware Embedding Definition

An embedding z_i = f_p(v_i) is **position-aware** if there exists a function g_p such that:

d_sp(v_i, v_j) = g_p(z_i, z_j)

where d_sp is shortest-path distance.

Standard GNN (structure-aware) embeddings cannot satisfy this in general.

---

## 7. Overview of the P-GNN Architecture (from slides)

Steps for computing position-aware embedding of a node v:
- (a) Randomly select anchor-sets
- (b) Compute pairwise distances s(v, u) to nodes in anchors
- (c) Compute messages from anchor-sets using some aggregator AGG_M
- (d) Transform the collected messages (AGG_S, w) into final position-aware embedding z_v

The position information is injected via the distance-to-anchor dimensions.

---

## 8. What Comes Next (Identity-aware GNNs)

(The lecture continues after the extracted pages.)

The second part addresses Limitation #2 (WL upper bound) using **Identity-aware GNNs**.

Typical approaches in this family:
- Inject learnable or structural "identity" features that allow the model to break symmetries that 1-WL cannot.
- Make aggregation more powerful than simple sum/mean (beyond GIN).
- Often combine ideas of unique but transferable node signatures or higher-order structures.

These methods aim to create GNNs that are **strictly more expressive** than the WL test while remaining practical.

---

## 9. Mental Model

| Problem                        | Symptom                              | Solution Direction          | Example Method     |
|--------------------------------|--------------------------------------|-----------------------------|--------------------|
| No positional info             | Same local structure → same emb    | Anchor-based coordinates    | P-GNN              |
| WL upper bound                 | Cannot distinguish certain graphs  | More powerful aggregation + identity injection | Identity-aware GNNs |
| One-hot IDs                    | Not scalable / not inductive       | Learnable + resampled refs  | Anchors + identity |

**Key takeaway from this lecture:**
Standard message-passing GNNs are fundamentally limited both in what structural information they can see and in what positions they can represent. The next generation of expressive GNNs explicitly attacks these two limitations with anchors (for position) and identity mechanisms (for higher expressivity).

---

These ideas (position-aware + identity-aware) directly motivate many modern powerful GNN architectures used when standard GNNs plateau on complex structural or positional reasoning tasks.
