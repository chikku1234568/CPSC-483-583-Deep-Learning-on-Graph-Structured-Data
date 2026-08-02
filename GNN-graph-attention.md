# Graph Attention and Multi-hop Attention — Notes from `07-graph_attention.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Source:** Lecture 7 slides

**Readings:**
- Graph Attention Networks (GAT)
- Multi-hop Attention Graph Neural Networks

---

## 1. Recap: What a GNN Layer Does

A single GNN layer:
- Takes node embeddings from previous layer: \( h_v^{(l-1)} \), \( h_u^{(l-1)} \) for \( u \in N(v) \)
- Two steps:
  1. **Message**: \( m_u^{(l)} = \text{MSG}^{(l)}(h_u^{(l-1)}) \)
  2. **Aggregate**: \( h_v^{(l)} = \text{AGG}^{(l)}(\{m_u^{(l)} \mid u \in N(v)\}, m_v^{(l)}) \)

Classic aggregators (GraphSAGE):
- Mean: \( \frac{1}{|N(v)|} \sum h_u \)
- Pool: Mean/Max of MLP-transformed neighbors
- LSTM: applied to shuffled neighbors (not permutation-invariant)

---

## 2. The Problem with Fixed Aggregation Weights

In GCN / GraphSAGE (mean):
- \( \alpha_{vu} = \frac{1}{|N(v)|} \)
- Every neighbor has **exactly the same importance**
- Weighting is **fixed** by node degree — not learned

**Limitation:**
- Cannot down-weight noisy or irrelevant neighbors
- No interpretability of which neighbors actually matter

---

## 3. Graph Attention Network (GAT)

**Core idea:** Let the model **learn** the importance of each neighbor.

Instead of fixed \( \alpha_{vu} \), compute **attention coefficients** that depend on the features of both nodes.

### Attention Mechanism

1. **Compute raw attention score**:
   $$
   e_{vu} = \text{att}(W^{(l)} h_v^{(l-1)}, W^{(l)} h_u^{(l-1)})
   $$
   - \( e_{vu} \): importance of \( u \)'s message to \( v \)
   - `att` can be any differentiable function (e.g., concat + linear, dot product, bilinear)

2. **Normalize** with softmax over neighbors:
   $$
   \alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in N(v)} \exp(e_{vk})}
   $$

3. **Aggregate**:
   $$
   h_v^{(l)} = \sigma\left( \sum_{u \in N(v)} \alpha_{vu} \, W^{(l)} h_u^{(l-1)} \right)
   $$

---

## 4. How to Implement the Attention Function (`att`)

The lecture shows several options:

| Method              | Formula                                      | Notes |
|---------------------|----------------------------------------------|-------|
| Concat + Linear     | `Linear( concat(Wh_v, Wh_u) )`              | Most common (original GAT) |
| Dot Product         | `(Wh_v) · (Wh_u)`                           | Simple |
| Bilinear            | `h_v^T M h_u`                               | More parameters |

The model learns the parameters of `att` **end-to-end** together with the weight matrices \( W \).

---

## 5. Multi-head Attention

To stabilize training (like in Transformers):

- Run **K independent attention mechanisms** (heads)
- Each head computes its own \( \alpha^k \) and embeddings \( h_v^k \)
- Combine outputs:
  - **Concatenation** (during intermediate layers)
  - **Average / Sum** (at the final layer)

$$
h_v^{(l)} = \text{AGG}\big( h_v^{(l)[1]}, h_v^{(l)[2]}, \dots, h_v^{(l)[K]} \big)
$$

Multi-head attention reduces variance and makes learning more robust.

---

## 6. Benefits of Attention in GNNs

1. **Adaptive weighting**
   - Can assign low weights to noisy nodes or bad edges
   - Much more robust than uniform/degree-based aggregation

2. **Interpretability**
   - Attention coefficients tell us which neighbors are important
   - In recommender systems: high \( \alpha \) means the user really cares about that item
   - Can sum attention scores across layers for path importance

---

## 7. Heterogeneous Graphs

### Definition
A **heterogeneous graph** has:
- Multiple **node types**
- Multiple **edge types**

### Real-world Examples

| Domain            | Node Types                          | Edge Types                     |
|-------------------|-------------------------------------|--------------------------------|
| E-commerce        | User, Item, Query, Location         | Purchase, Click, Search, Guide |
| Academic          | Author, Paper, Venue, Field         | Publish, Cite                  |
| Biomedical        | Drug, Disease, Protein, Pathway     | Treat, Associate, Cause        |

**Challenge:** Different node types often live in different feature spaces.

GAT (in its basic form) assumes homogeneous graphs. Handling heterogeneity requires:
- Type-specific transformations
- Relation-specific attention (or separate attention per edge type)
- Metapath-based methods (covered in later lectures)

---

## 8. Multi-hop Attention Graph Neural Networks

(This section is based on the second required reading and typical content for this lecture.)

Standard GAT only looks at **1-hop neighbors** per layer. Stacking layers gives multi-hop information indirectly.

**Multi-hop Attention** methods explicitly model attention over:
- Longer paths / multi-hop neighborhoods
- Different hop distances
- Or entire subgraphs / metapaths

### Key ideas usually covered:
- Compute attention not just on direct neighbors, but on nodes reached via 2-hop, 3-hop, etc.
- Use attention to weigh the contribution of different hop distances
- Can reduce over-smoothing by selectively focusing on important distant nodes
- Often combined with sampling or clustering for scalability

**Intuition:**
> Not all information at distance 2 is equally useful — some paths are much more informative than others. Multi-hop attention learns which paths matter.

---

## 9. Mental Model

| Aspect                    | GCN / GraphSAGE (fixed)          | GAT (learnable attention)          |
|---------------------------|----------------------------------|------------------------------------|
| Neighbor importance       | Fixed by degree or uniform       | Learned per neighbor               |
| Handles noise             | Poor                             | Good (down-weights noisy nodes)    |
| Interpretability          | None                             | Attention scores as explanation    |
| Multi-head                | No                               | Yes (stabilizes training)          |
| Heterogeneous support     | Requires extensions              | Requires type-aware attention      |

**Rule of thumb:**
- Use **GAT** when you suspect that some neighbors are much more important than others.
- Use multi-head attention almost always (cheap insurance).
- For heterogeneous graphs, move beyond basic GAT toward heterogeneous attention or metapath methods.

---

## Quick Formulas

**Standard GAT layer (single head):**
$$
\alpha_{vu} = \frac{\exp\left( \text{att}(Wh_v, Wh_u) \right)}{\sum_k \exp\left( \text{att}(Wh_v, Wh_k) \right)}
$$
$$
h_v' = \sigma\left( \sum_{u \in N(v)} \alpha_{vu} W h_u \right)
$$

**Multi-head version:** run K independent heads and aggregate (concat or mean).

---

This lecture bridges **classic neighborhood aggregation** (fixed weights) to **learnable, interpretable aggregation** via attention, and introduces the complexity of real-world heterogeneous graphs.
