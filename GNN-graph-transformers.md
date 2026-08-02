# Graph Transformers — Notes from `12-graph_transformer.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Lecture:** Graph Transformers

---

## 1. Motivation: Why Graph Transformers?

Standard Message-Passing GNNs (MPNNs) suffer from three major problems:

| Problem          | Description                                                                 | Consequence |
|------------------|-----------------------------------------------------------------------------|-------------|
| **Limited expressiveness** | At most as powerful as 1-WL test                                           | Cannot distinguish many non-isomorphic graphs |
| **Over-smoothing** | Node features converge after many layers                                    | Hard to train deep GNNs |
| **Over-squashing** | Long-range information is compressed into fixed-size vectors through bottlenecks | Poor long-range dependency modeling |

**Core idea of Graph Transformers:**
- Replace local message passing with **global self-attention** (every node can directly attend to every other node).
- Naturally solves over-squashing and long-range dependencies.
- Can be more expressive if given proper encodings.

---

## 2. Challenges in Adapting Transformers to Graphs

### Challenge 1: No Natural Ordering
- Language has sequential order → sinusoid positional encoding works.
- Graphs are **permutation invariant** — there is no canonical node order.
- Classical PE fails because permuting node order must not change semantics.

→ Need **graph-specific Positional Encoding (PE)**.

### Challenge 2: Weak Structural Awareness
- MPNNs already struggle to capture rich local substructures.
- Pure attention on raw node features does not see neighborhood patterns well.
- Example: Circular Skip Links (CSL) graphs that MPNNs cannot distinguish.

→ Need **Structural Encoding (SE)** to inject local subgraph information.

### Challenge 3: Scalability
- Full self-attention is O(N²) in nodes.
- MPNNs are O(E) → much cheaper on sparse graphs.

→ Need sparsity, sampling, or efficient attention mechanisms for large graphs.

---

## 3. Positional Encoding (PE) vs Structural Encoding (SE)

| Encoding Type       | What it encodes                              | Goal                                      | Desired property |
|---------------------|----------------------------------------------|-------------------------------------------|------------------|
| **Positional (PE)** | Position of a node in the whole graph        | Nodes close in graph should have similar PE | Distance-preserving |
| **Structural (SE)** | Local neighborhood / subgraph patterns       | Increase expressiveness + generalization  | Similar subgraphs → similar SE |

Both are typically added to node features before feeding into the Transformer.

---

## 4. Laplacian Positional Encoding (LapPE)

### Intuition
The eigenvectors of the **graph Laplacian** naturally encode positional information, analogous to Fourier basis on grids.

Graph Laplacian:
$$
L = I - D^{-1/2} A D^{-1/2} = U^T \Lambda U
$$

- Columns of U are eigenvectors φ₀, φ₁, ..., φ_{N-1}
- Rows correspond to nodes

### LapPE Construction
For node j, take the first m eigenvectors (corresponding to smallest non-zero eigenvalues):

$$
p_j^{\text{LapPE}} = [\phi_{1j}, \phi_{2j}, \dots, \phi_{mj}]
$$

Low-frequency eigenvectors vary smoothly across the graph.  
High-frequency eigenvectors capture local oscillations.

### Sign Ambiguity
If φ is an eigenvector, so is −φ.

- m eigenvectors → 2^m possible sign combinations.
- This breaks consistency across graphs / runs.

**Solutions:**
1. Randomly flip signs of eigenvectors during training (encourages invariance).
2. Use permutation-equivariant mapping:
   $$
   \text{PE} = [f(\phi_1) + f(-\phi_1), \dots, f(\phi_m) + f(-\phi_m)]
   $$
   where f can be DeepSets, Transformer, or GNN.

### Learnable LapPE
Concatenate eigenvectors with their eigenvalues and pass through a small Transformer or MLP to produce learned positional encodings.

---

## 5. Random Walk Structural Encoding (RWSE)

(This section starts in the extracted slides.)

### Random Walk on Graphs
- From a starting node, repeatedly jump to a random neighbor.
- The sequence of visited nodes is a random walk.

### Using Walks as Structural Encoding
- Record the probability or count of reaching each node after k steps.
- This captures rich local structural patterns (cycles, communities, etc.).
- Can be used as node or edge features in the Transformer.

RWSE helps Graph Transformers distinguish graphs that are hard for pure attention or MPNNs (e.g., CSL graphs).

---

## 6. Token Construction

Typical approach:
1. Start with raw node features X.
2. Add **Positional Encoding** (LapPE or other).
3. Add **Structural Encoding** (RWSE, centrality, etc.).
4. (Optional) Add edge features or edge encodings.
5. The resulting vectors become the input **tokens** to the Transformer encoder.

Some models also create tokens for edges or subgraphs.

---

## 7. Forward Propagation in Graph Transformers

High-level flow:
- Input tokens (with PE + SE) → Multi-head self-attention (full or sparse) → FFN → residual + norm → ...
- Because attention is global, long-range information flows in one layer.
- Output can be used for node-level, edge-level, or graph-level prediction (with pooling).

---

## 8. Scalability Considerations

Standard full attention costs O(N²) time and memory for QKᵀ, softmax, and attention×V.

Common mitigation strategies (discussed in modern Graph Transformers):
- Sparse attention (only attend to k-nearest or sampled nodes)
- Linear / kernelized attention approximations
- Graph-specific sparsity (e.g., attend within local neighborhoods + a few global nodes)
- Sampling or clustering of nodes
- Performer, BigBird, or other efficient Transformer variants adapted to graphs

---

## 9. Summary Table

| Aspect                    | Message Passing GNNs          | Graph Transformers                     |
|---------------------------|-------------------------------|----------------------------------------|
| Information flow          | Local (1-hop per layer)       | Global (all-to-all attention)          |
| Long-range dependencies   | Poor (many layers needed)     | Direct                                 |
| Expressiveness            | ≤ 1-WL (usually)              | Can exceed 1-WL with good PE/SE        |
| Over-smoothing            | Severe when deep              | Less severe                            |
| Over-squashing            | Yes                           | Mitigated                              |
| Complexity                | O(E)                          | O(N²) naively; O(N log N) or better with tricks |
| Need for encodings        | Sometimes helpful             | **Critical** (PE + SE)                 |

---

## 10. Key Takeaways

- Graph Transformers are motivated by the desire to overcome the three classic MPNN failure modes using global attention.
- The two crucial ingredients that make them work on graphs are **carefully designed Positional Encodings** (LapPE being the most popular) and **Structural Encodings** (RWSE, etc.).
- Sign ambiguity in eigenvectors and the quadratic cost are the main practical headaches.
- When done right, Graph Transformers are currently among the strongest models on many graph benchmarks, especially when long-range or global structure matters.

---

This lecture marks the shift from "local neighborhood aggregation" to "global attention + rich structural/positional encodings" in modern graph deep learning.
