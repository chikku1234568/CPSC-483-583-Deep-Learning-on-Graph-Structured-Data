# Hyperbolic Geometry and Hyperbolic GNNs — Notes from `14-hyperbolic.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Lecture:** Hyperbolic Geometry and Hyperbolic GNNs

**Readings:**
- HGCN: Hyperbolic Graph Convolutional Neural Networks
- Hyperbolic GNN survey

---

## 1. Motivation: Why Hyperbolic Geometry?

### Problem with Euclidean Embeddings for Hierarchical Data

When embedding **tree-structured** or **hierarchical** graphs in Euclidean space:

- Early layers (near root) embed nicely.
- As depth increases, nodes spread out on a circle/sphere.
- **Outer leaves** that are semantically far apart in the graph become **artificially close** in Euclidean space due to curvature constraints.
- We lose the desired property: "close in embedding ⇔ connected or close in graph".

This is not just a practical issue — there are **fundamental theoretical limits**.

### Theoretical Limitations of Euclidean Space

1. **Distortion lower bound** (Lee et al., 2007):
   - There is a lower bound on how little distortion you can achieve when embedding hierarchical structures into Euclidean space ℝⁿ, no matter the dimension.

2. **Dimension-distortion tradeoff** (Matoušek, 2002):
   - To achieve low distortion when embedding graphs (token relations, hierarchies), the required dimension grows **near-quadratically** with the allowed distortion.
   - Consequence: Euclidean models need very high dimensions for complex structures → poor scalability.

**Conclusion:** Euclidean geometry is fundamentally mismatched for data with strong hierarchical or tree-like structure.

---

## 2. Architecture vs. Embedding Geometry

Two orthogonal ways to increase model power:
- **Architecture** (GNN layers, attention, etc.)
- **Geometry of the embedding space**

The lecture focuses on the second: choosing a better **embedding geometry** can benefit many architectures.

---

## 3. Non-Euclidean Geometry

### Euclidean Geometry
- Satisfies the parallel postulate: given a line l and point P not on it, **exactly one** line through P is parallel to l.

### Non-Euclidean Geometries
| Geometry   | Curvature     | Parallel lines through P | Example manifold      |
|------------|---------------|---------------------------|-----------------------|
| Euclidean  | Zero          | Exactly 1                 | Plane                 |
| Hyperbolic | Negative      | Infinitely many           | Hyperboloid / Poincaré disk |
| Spherical  | Positive      | None                      | Sphere                |

---

## 4. Riemannian Manifolds

A **Riemannian manifold** ℳ is a smooth manifold equipped with:
- **Tangent space** T_pℳ at every point p (a local Euclidean approximation)
- A smoothly varying **inner product** g_p(·,·) on the tangent space

This allows us to define:
- Curves γ(t)
- Velocity vectors γ̇(t) ∈ T_pℳ
- Distances via integration of the metric

---

## 5. Curvature

Curvature measures how a surface bends away from its tangent plane:

- **Positive curvature**: surface lies entirely on one side of the tangent plane (e.g., sphere)
- **Negative curvature**: tangent plane cuts through the surface (e.g., saddle / hyperboloid)
- **Zero curvature**: surface agrees with tangent plane along a line (e.g., plane or cylinder)

Hyperbolic space has **constant negative curvature** (−1/K).

---

## 6. Hyperbolic Space

- Riemannian manifold with constant negative curvature.
- Volume of a ball grows **exponentially** with radius (unlike polynomial growth in Euclidean space).
- This exponential volume growth is perfect for embedding trees and hierarchies (branching creates exponential numbers of nodes).

**Important theorem (Hilbert 1901):**  
There is no isometric embedding of a complete hyperbolic surface with constant negative curvature into ℝ³.  
→ We need to embed into **Minkowski space** (pseudo-Euclidean space).

---

## 7. Minkowski Space and the Hyperboloid Model

### Minkowski Metric
Instead of the Euclidean inner product:
$$
g_E(u,v) = u_0 v_0 + u_1 v_1 + \dots + u_d v_d
$$

We use the **Minkowski inner product**:
$$
\langle u, v \rangle_\mathcal{L} = -u_0 v_0 + u_1 v_1 + \dots + u_d v_d
$$

(The time-like coordinate is treated with opposite sign.)

### Hyperboloid Model
The **hyperboloid model** of hyperbolic space with curvature −1/K is:
$$
\mathbb{H}^{d,K} = \{ x \in \mathbb{R}^{d+1} \mid \langle x, x \rangle_\mathcal{L} = -K \}
$$

Points live in (d+1)-dimensional Minkowski space, but the manifold itself is d-dimensional hyperbolic space.

### Geodesic Distance (Hyperboloid)
$$
D_M^K(x, y) = \sqrt{K} \cdot \text{arcosh}\left( -\frac{\langle x, y \rangle_\mathcal{L}}{K} \right)
$$

---

## 8. Connection to Special Relativity

The Minkowski metric is the same one used in special relativity (space-time).

- Vectors are classified as:
  - Time-like
  - Space-like
  - Light-like (on the light cone)

Lorentz transformations (boosts) can be interpreted as **hyperbolic rotations** in the Minkowski space.  
This gives intuition for why hyperbolic geometry appears naturally in relativistic physics.

---

## 9. Why Hyperbolic Geometry Helps Graphs

- Trees and hierarchies have **exponential growth** in number of nodes with depth.
- Hyperbolic space has **exponential volume growth** with radius.
- Therefore, trees can be embedded with **low distortion** in hyperbolic space even in low dimensions.
- Leaves that are far apart in the tree remain far apart in hyperbolic embeddings.

This is the geometric reason behind the success of hyperbolic graph embeddings and HGCN.

---

## 10. Preview: Hyperbolic GNNs (Next Topics in Lecture)

(The extracted pages stop before the actual HGCN architecture, but based on the outline:)

Expected next content:
- Different models of hyperbolic space (hyperboloid, Poincaré ball, Klein, etc.) and conversions between them.
- Hyperbolic versions of basic operations: addition, multiplication, exponential / logarithmic maps.
- Hyperbolic Graph Convolutional Networks (HGCN): performing message passing directly in hyperbolic space using gyrovector spaces or tangent space approximations.
- Advantages on hierarchical, scale-free, and tree-like graphs (e.g., taxonomies, social networks with community structure, knowledge graphs).

---

## 11. Key Takeaways

- Euclidean space has fundamental limitations (distortion + dimension scaling) for hierarchical data.
- Hyperbolic geometry, with its negative curvature and exponential volume growth, is a much more natural geometry for trees and hierarchies.
- We model hyperbolic space using the hyperboloid in Minkowski space (because it cannot be isometrically embedded in Euclidean ℝ³).
- The right embedding geometry is as important as the neural architecture itself.
- This motivates **Hyperbolic GNNs** (HGCN and follow-ups), which perform convolutions / attention natively in hyperbolic space.

---

These ideas have influenced not only GNNs but also hyperbolic word embeddings, hyperbolic VAEs, and more recently, some work on geometric transformers and foundation models.
