# Geometric Deep Learning — Notes from `15-geometry-deep-learning-2.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Lecture:** Geometric Deep Learning (Part 2)

---

## 1. Symmetry in Nature and ML

Nature is highly symmetric because symmetry makes things easier to encode/store.

**Definition:** Symmetries are transformations under which certain properties of an object stay the same.

In physical/scientific domains (physics, chemistry, fluid dynamics, biology):
- Certain quantities must be **invariant** (e.g., molecular energy).
- Certain quantities must be **equivariant** (e.g., forces, velocities, coordinates).

**Key symmetry in 3D:** Equivariance / invariance to **rigid-body motions**:
- Rotations
- Reflections
- Translations

---

## 2. Graphs vs. Geometric Graphs

### Standard Graphs
$$
G = (A, S)
$$
- A: adjacency matrix (binary or weighted)
- S: scalar node features
- **Geometric shape has no meaning**

### Geometric Graphs
$$
G = (A, S, \vec{X}, \vec{V})
$$
- Nodes live in 3D Euclidean space
- $\vec{X} \in \mathbb{R}^{n \times 3}$: coordinates
- $\vec{V} \in \mathbb{R}^{n \times c \times d}$: vector features (e.g., velocities, forces)
- Regular MPNNs are **not symmetry-aware** for these.

---

## 3. Drawing Edges in Geometric Settings

In molecules etc., "bonds" are a simplification. All atoms interact based on proximity.

Design choices for edges:
- Smoothed radial cutoff graphs (most common)
- Long-range connections
- Complete graph

**Practical recommendation:** sparse radial cutoff graphs are usually sufficient.

---

## 4. Group Theory Basics

A **group** $\mathcal{G}$ is a set with operation $\cdot$ satisfying:
1. Closure
2. Associativity
3. Identity element $e$
4. Every element has an inverse

**Important groups for 3D geometry:**
| Group     | Description                          | Contains                  |
|-----------|--------------------------------------|---------------------------|
| SO(3)     | Rotations only                       | Rotations                 |
| O(3)      | Rotations + reflections              | SO(3) + reflections       |
| T(3)      | Translations only                    | Translations              |
| SE(3)     | Rotations + translations             | SO(3) + T(3)              |
| E(3)      | Rotations + reflections + translations | O(3) + T(3)             |

**Subgroup relationships:**
- $SO(3) \subset O(3) \subset E(3)$
- $SE(3) \subset E(3)$
- $T(3) \cup SO(3) \cong SE(3)$
- $T(3) \cup O(3) \cong E(3)$

---

## 5. Group Representations and Actions

- Abstract group elements (e.g., "rotate 130° about x") are represented as concrete matrices (3×3 for SO(3)).
- A group $\mathcal{G}$ can **act** on a set $X$ (e.g., 3D points).

**Action on a point:**
$$
R_x(\theta) \cdot \begin{pmatrix} a \\ b \\ c \end{pmatrix} = R_x(\theta) \vec{x}
$$

**Important convention for point clouds:**
- For $\vec{x} \in \mathbb{R}^3$: rotated point = $R \vec{x}$
- For $\vec{X} \in \mathbb{R}^{n \times 3}$: rotated point cloud = $\vec{X} R^T$

**Composition:**
$$
R_x R_y \vec{x} \quad \text{or for clouds} \quad (\vec{X} R_y^T) R_x^T
$$

---

## 6. Invariance vs Equivariance (Group View)

For a transformation $g \in \mathcal{G}$ (e.g., rotation $R \in SO(3)$):

- **Invariance**: $f(g \cdot x) = f(x)$
- **Equivariance**: $f(g \cdot x) = g \cdot f(x)$

In geometric graphs:
- **Scalar features** $S$ must be **invariant**
- **Vector features** $\vec{X}, \vec{V}$ must be **equivariant**

Translations only affect coordinates $\vec{X}$, not velocities $\vec{V}$ or scalars.

---

## 7. Geometric Message Passing Requirements

In geometric GNNs:

- **Scalar track**: messages and updates must be invariant to rigid motions.
- **Vector track**: messages and updates must be equivariant.

Different models implement `Agg` and `Upd` differently while respecting these constraints.

---

## 8. E(n)-Equivariant GNN (E-GNN) [Satorras et al., 2021]

E-GNN is a clean, widely-used formulation with **two separate tracks**.

### Message Construction (invariant)
$$
\vec{m}_{ij} = f_\theta \Big( s_i^{(t)},\, s_j^{(t)},\, \Vert \vec{x}_i^{(t)} - \vec{x}_j^{(t)} \Vert^2,\, e_{ij} \Big)
$$

Uses only **invariant** quantities (scalars + squared distance).

### Vector (Coordinate) Update (equivariant)
$$
\vec{x}_i^{(t+1)} = \vec{x}_i^{(t)} + \sum_{j \in \mathcal{N}_i} (\vec{x}_i^{(t)} - \vec{x}_j^{(t)}) \cdot f_\phi(\vec{m}_{ij})
$$

The term $(\vec{x}_i - \vec{x}_j)$ provides **relative direction** (equivariant). $f_\phi$ is an MLP.

### Scalar Update
$$
\vec{m}_i = \sum_{j \in \mathcal{N}_i} \vec{m}_{ij}
$$
$$
s_i^{(t+1)} = f_\psi \big( s_i^{(t)},\, \vec{m}_i \big)
$$

### Full Layer Summary
$$
\begin{aligned}
\vec{m}_{ij} &= f_\theta(s_i^{(t)}, s_j^{(t)}, \Vert\vec{x}_i^{(t)}-\vec{x}_j^{(t)}\Vert^2, e_{ij}) \\
\vec{x}_i^{(t+1)} &= \vec{x}_i^{(t)} + \sum_{j} (\vec{x}_i^{(t)} - \vec{x}_j^{(t)}) f_\phi(\vec{m}_{ij}) \\
\vec{m}_i &= \sum_j \vec{m}_{ij} \\
s_i^{(t+1)} &= f_\psi(s_i^{(t)}, \vec{m}_i)
\end{aligned}
$$

All $f_\theta, f_\phi, f_\psi$ are MLPs.

---

## 9. Proof Sketch: E-GNN is SO(3)-Equivariant / Invariant

Assume the input is rotated: coordinates become $R \vec{x}_i^{(t)}$.

- Scalar messages depend only on invariant quantities → $\vec{m}_{ij}$ unchanged.
- Vector update:
  $$
  R\vec{x}_i^{(t+1)} = R \Big( \vec{x}_i^{(t)} + \sum (\vec{x}_i^{(t)} - \vec{x}_j^{(t)}) f_\phi(\vec{m}_{ij}) \Big) = \text{EGNN}(R\vec{x}_i^{(t)})
  $$
- Scalar features:
  $$
  s_i^{(t+1)} = \text{EGNN}(R s_i^{(t)}) = \text{EGNN}(s_i^{(t)})
  $$

A rotation acts as the identity on scalars (group theory fact).

Thus E-GNN is **SO(3)-equivariant** on vectors/coordinates and **invariant** on scalars.

---

## 10. Key Takeaways So Far

- Modeling the right symmetries (E(3)/SE(3)/SO(3)) is critical for geometric data (molecules, proteins, 3D scenes, etc.).
- Geometric graphs carry both scalar **and** vector features that must transform correctly.
- E-GNN provides a simple, effective recipe by keeping scalar and vector tracks separate and using only invariant quantities (squared distances) to construct messages.
- The relative direction term in coordinate updates is what provides the necessary equivariance.

---

(The lecture continues with more geometric GNN designs, other equivariant architectures, and applications — the extracted content ends mid-lecture.)

This lecture bridges group theory, geometry, and graph neural networks for 3D / physical systems.
