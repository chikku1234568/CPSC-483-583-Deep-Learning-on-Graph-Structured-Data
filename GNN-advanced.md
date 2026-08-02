# GNN Design — Notes from `04-GNN_design.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Source:** Lecture 4 slides (`04-GNN_design.pdf`)

**Related readings (from the deck):**
- Semi-Supervised Classification with Graph Convolutional Networks (GCN)
- Principled Neighborhood Aggregation on Graph Nets
- (From Lecture 3) GraphSAGE — Inductive Representation Learning for Large Graphs

---

## 0. Big picture

Lecture 3 said: GNNs embed nodes by repeatedly aggregating neighborhood info.  
**This lecture asks:** *how do we design those layers and the full network?*

**One unified view of a GNN:**

| Piece | What it is |
|-------|------------|
| **(1) Message** | each node creates a vector to send |
| **(2) Aggregation** | combine neighbors' messages into one vector |
| **(3) Layer connectivity** | how layers stack (depth, skip connections) |
| **(4) Graph augmentation** | raw graph is not the computational graph (features + structure) |
| **(5) Learning objective** | supervised / unsupervised; node / edge / graph level |

Classic models (GCN, GraphSAGE, GAT, ...) are just different choices for message + aggregation.

**Mental model:** every GNN layer compresses a *set* of neighbor vectors into *one* node vector (order-invariantly), then we stack those layers carefully and optionally rewrite the graph we run them on.

---

## 1. A single GNN layer

### 1.1 Two steps only

Idea of one layer: **compress a set of vectors into a single vector**.

**Inputs at layer $\ell$:**  
$h_v^{(\ell-1)}$ (self) and $\{h_u^{(\ell-1)} : u \in N(v)\}$ (neighbors).

**Output:** $h_v^{(\ell)}$.

```
neighbors --> (1) MESSAGE --> messages --> (2) AGGREGATION --> h_v^(l)
                  ^
                also self
```

---

### 1.2 Message

Each node builds a message that will be sent outward:

$$
m_u^{(\ell)} = \mathrm{MSG}^{(\ell)}\big(h_u^{(\ell-1)}\big), \qquad u \in N(v) \cup \{v\}
$$

**Simplest example — linear map:**

$$
m_u^{(\ell)} = W^{(\ell)} h_u^{(\ell-1)}
$$

Intuition: rewrite my features into a language my neighbors will read.

---

### 1.3 Aggregation

Node $v$ pools messages from its neighbors:

$$
h_v^{(\ell)} = \mathrm{AGG}^{(\ell)}\big(\{m_u^{(\ell)} : u \in N(v)\}\big)
$$

**Common aggregators (order-invariant):** Sum, Mean, Max.

Example:

$$
h_v^{(\ell)} = \mathrm{Sum}\big(\{m_u^{(\ell)} : u \in N(v)\}\big)
$$

---

### 1.4 Self-connection (don't forget yourself)

If you only aggregate *neighbors*, $v$'s own identity can get washed out.

**Fix:** include a message from $v$ itself, often with a *different* weight matrix:

$$
m_u^{(\ell)} = W^{(\ell)} h_u^{(\ell-1)} \quad (u \in N(v)), \qquad m_v^{(\ell)} = B^{(\ell)} h_v^{(\ell-1)}
$$

Then mix self + neighbors via **concat** or **sum**:

$$
h_v^{(\ell)} = \mathrm{CONCAT}\Big( \mathrm{AGG}\big(\{m_u^{(\ell)} : u \in N(v)\}\big),\; m_v^{(\ell)} \Big)
$$

**Link to Lecture 3's formula:**  
separate $W$ (neighbors) and $B$ (self) *is* this self-connection idea; concat vs sum is a design choice (GraphSAGE likes concat).

---

### 1.5 Full layer template

$$
m_u^{(\ell)} = \mathrm{MSG}^{(\ell)}\big(h_u^{(\ell-1)}\big), \quad u \in N(v) \cup \{v\}
$$

$$
h_v^{(\ell)} = \mathrm{AGG}^{(\ell)}\big(\{m_u^{(\ell)} : u \in N(v)\},\; m_v^{(\ell)}\big)
$$

**Nonlinearity $\sigma$** (ReLU, sigmoid, ...) can sit in message *or* aggregation — it adds expressiveness.

---

## 2. Classical layers as Message + Aggregation

### 2.1 GCN (Graph Convolutional Network)

$$
h_v^{(\ell)} = \sigma\left( W^{(\ell)} \sum_{u \in N(v)} \frac{h_u^{(\ell-1)}}{\sqrt{|N(u)|\,|N(v)|}} \right)
$$

**As Message + Agg:**

| Step | Formula |
|------|---------|
| **Message** | $m_u^{(\ell)} = \dfrac{1}{\sqrt{|N(u)|\,|N(v)|}} W^{(\ell)} h_u^{(\ell-1)}$ |
| **Aggregation** | $h_v^{(\ell)} = \sigma\big(\mathrm{Sum}\{m_u^{(\ell)}\}\big)$ |

**Intuition of the $\sqrt{|N(u)||N(v)|}$ norm:**  
symmetric degree normalization — high-degree nodes don't dominate; also keeps spectra well-behaved (GCN paper motivation).  
Often implemented with self-loops in $N(v)$ (so "neighbors" includes $v$).

**Matrix form (for efficiency):** sparse multiply with normalized $\tilde{A}$.

---

### 2.2 GraphSAGE

$$
h_v^{(\ell)} = \sigma\Big( W^{(\ell)} \cdot \mathrm{CONCAT}\big( h_v^{(\ell-1)},\; \mathrm{AGG}\{h_u^{(\ell-1)} : u \in N(v)\} \big) \Big)
$$

**Two-stage aggregation:**

1. **Neighbors:** $h_{N(v)}^{(\ell)} \leftarrow \mathrm{AGG}\{h_u^{(\ell-1)} : u \in N(v)\}$
2. **Self + neighbors:** $h_v^{(\ell)} \leftarrow \sigma\big(W^{(\ell)} \cdot \mathrm{CONCAT}(h_v^{(\ell-1)}, h_{N(v)}^{(\ell)})\big)$

Message is often *inside* the AGG (e.g. MLP before pool).

#### GraphSAGE aggregator menu

| Name | Formula | Notes |
|------|---------|--------|
| **Mean** | $\sum_{u \in N(v)} h_u^{(\ell-1)} / \|N(v)\|$ | like a soft GCN; order-invariant |
| **Pool** | $\mathrm{Mean/Max}\{\mathrm{MLP}(h_u^{(\ell-1)})\}$ | nonlinear message then symmetric pool |
| **LSTM / RNN** | $\mathrm{LSTM}([h_u^{(\ell-1)}]_{u \in \pi(N(v))})$ | need random shuffle $\pi$ for (approx.) order-invariance |

#### Optional $\ell_2$ normalize (per layer)

$$
h_v^{(\ell)} \leftarrow \frac{h_v^{(\ell)}}{\|h_v^{(\ell)}\|_2}
$$

- Keeps embeddings on a similar scale
- Sometimes helps accuracy
- Helps nearest-neighbor retrieval (e.g. LSH)

---

### 2.3 Quick comparison

| | **GCN** | **GraphSAGE** |
|--|---------|----------------|
| Self handling | usually via self-loops in $A$ | explicit **concat** with self |
| Neighbor weight | degree-normalized (symmetric) | mean / pool / LSTM (you choose) |
| Inductive story | can be inductive with features | designed explicitly inductive + sampling |
| Message | shared $W$, scaled by degrees | often inside AGG (MLP, ...) |

**(GAT** = attention weights over neighbors — same Message+Agg template, different AGG. Mentioned in framework; details often in later / attention lectures.)

---

## 3. GNN layer in practice (modern DL modules)

Classic GCN/SAGE are a **great starting point**. Stronger layers stack familiar deep-learning blocks:

```
Linear --> BatchNorm --> Dropout --> Activation --> Attention --> Aggregation
           \_______________ Transformation _______________/
```

### 3.1 Batch Normalization

**Goal:** stabilize training.

Given a minibatch of $N$ node embeddings $X \in \mathbb{R}^{N \times D}$:

1. Per feature dim $j$: mean $\mu_j$, variance $\sigma_j^2$ over the batch
2. Normalize, then affine rescale with trainable $\gamma_j, \beta_j$:

$$
\hat{X}_{i,j} = \frac{X_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad Y_{i,j} = \gamma_j \hat{X}_{i,j} + \beta_j
$$

**LayerNorm** alternative: normalize *each* embedding vector individually (not across the batch) — often nicer when batch stats are noisy (small batches / varying graphs).

### 3.2 Dropout

**Goal:** regularize, reduce overfitting.

- **Train:** randomly zero a fraction $p$ of dimensions
- **Test:** use full network (`model.train()` vs `model.eval()` in PyTorch)

**In GNNs:** typically applied on the **linear message** path $m_u = W h_u$, not on the graph topology itself.

### 3.3 Activations

| Act | Formula | When |
|-----|---------|------|
| **ReLU** | $\max(x_i, 0)$ | default workhorse |
| **Sigmoid** | $1/(1+e^{-x_i})$ | when you need bounded outputs |
| **PReLU** | $\max(x_i,0) + a_i \min(x_i,0)$ | $a_i$ trainable; sometimes beats ReLU |

Designing new GNN layers is still active research — tools like **GraphGym** help sweep designs.

---

## 4. Stacking layers

### 4.1 Standard stack

**One GNN layer = one round of Message + Aggregation.**

We stack these layers:

$$
h_v^{(0)} = x_v
$$

$$
h_v^{(1)} \leftarrow \text{(Message + Agg layer applied to } h_v^{(0)}\text{)}
$$

$$
h_v^{(2)} \leftarrow \text{(Message + Agg layer applied to } h_v^{(1)}\text{)}
$$

$$
\vdots
$$

$$
h_v^{(L)} \leftarrow \text{(Message + Agg layer applied to } h_v^{(L-1)}\text{)}
$$

**Crucial point:**
- After **k layers**, the embedding $h_v^{(k)}$ contains information from nodes **up to k hops** away.
- Therefore: If you want every node to see up to **5 hops** of context, you need **L = 5** layers.

This is why depth directly controls the receptive field size.

---

### 4.2 Over-smoothing (the depth trap)

**Problem:** stack *too many* GNN layers and all node embeddings become nearly **identical**.  
Then you can't tell nodes apart — classification / retrieval dies.

**Why?** via **receptive fields**.

**Receptive field** of $v$ = set of nodes that influence $h_v$.  
In a $K$-layer GNN this is roughly the **$K$-hop** neighborhood (because each layer = one hop of message passing).

- 1-layer: only immediate neighbors
- 2-layer: neighbors-of-neighbors
- 3-layer: even larger blob

As $K$ grows, receptive fields of different nodes **heavily overlap** so embeddings become similar (over-smoothing).

**Unlike CNNs:** "just go deeper" is *not* free on graphs.

---

### 4.3 How deep should you go?

Practical recipe:

1. **Estimate needed receptive field** for your task (e.g. think about graph diameter, how far labels should influence each other).
2. Set $L$ a bit larger than that; **tune as a hyperparameter**.
3. Prefer **not** blindly stacking 20 plain GNN layers.

**Question this raises:** if $L$ must stay small, how do we stay expressive?

---

### 4.4 More expressiveness without more hops

#### Solution A — Deeper transforms *inside* one GNN layer

Each message / aggregation box can be a **multi-layer MLP**, not a single linear map.  
Same hop radius, richer function class.

#### Solution B — Pre-process and post-process MLPs (per node)

```
[MLP --> MLP]  -->  [GNN --> GNN --> GNN]  -->  [MLP --> MLP]
 pre-process         message passing             post-process
```

| Block | When it helps |
|-------|----------------|
| **Pre-process** | raw features need encoding (images, text as nodes) |
| **Post-process** | reasoning on embeddings (graph classification heads, KG scoring, ...) |

In practice this often works **really well** — capacity where you need it, without expanding receptive fields.

---

### 4.5 Skip connections (when you *do* need depth)

**Observation:** early-layer embeddings often still separate nodes well; late layers blur them.

**Idea:** shortcuts so final embeddings still "hear" earlier layers  
(same residual idea as ResNets / Transformer residual stream).

**Basic residual form:**

$$
\text{before: } F(x) \qquad \text{after: } F(x) + x
$$

**Example — mean-agg GNN + skip:**

Standard:

$$
h_v^{(\ell)} = \sigma\left( \sum_{u \in N(v)} W^{(\ell)} \frac{h_u^{(\ell-1)}}{|N(v)|} \right) \quad (= F)
$$

With skip (self residual inside):

$$
h_v^{(\ell)} = \sigma\left( \sum_{u \in N(v)} W^{(\ell)} \frac{h_u^{(\ell-1)}}{|N(v)|} \;+\; h_v^{(\ell-1)} \right)
$$

#### Why skips help — mixture-of-depths intuition

Each skip creates a choice: **go through** the block or **bypass** it.  
$N$ skip points give up to $2^N$ paths — an automatic mixture of **shallow and deep** GNNs in one model.

#### Other skip pattern: jump to the end

Keep $h_v^{(0)}, h_v^{(1)}, \ldots, h_v^{(L)}$ and **aggregate across layers** at the end (concat / pool / LSTM) to get final $h_v^{(\mathrm{final})}$.  
Forces the predictor to use multi-scale (1-hop, 2-hop, ...) views.

---

## 5. Graph manipulation (raw graph is not the compute graph)

### 5.1 Why touch the graph at all?

Naive assumption: **input graph = computational graph**. Often wrong.

| Problem | Symptom |
|---------|---------|
| **No features** | only adjacency exists |
| **Too sparse** | messages barely travel |
| **Too dense** | message passing too expensive |
| **Too large** | full compute graph won't fit on GPU |
| **Suboptimal topology** | input edges are not the best edges for the task |

So we **augment features** and/or **rewrite structure**.

---

### 5.2 Feature augmentation

#### Case 1 — No node features

| Approach | Idea | Pros | Cons |
|----------|------|------|------|
| **Constant** | every node gets $1$ (or same vector) | inductive, cheap, $O(1)$ dim | weaker expressivity (all start identical; structure must carry signal) |
| **One-hot ID** | node $i$ gets $e_i \in \mathbb{R}^{\|V\|}$ | highly expressive; node-specific memory | **not inductive**; $O(\|V\|)$ dims; bad for large graphs |

**Use constants** for inductive / large graphs.  
**Use one-hots** for small **transductive** graphs (fixed node set).

#### Case 2 — Structure GNN can't "see" easily

Example: **cycle length** containing $v_1$.

Triangle vs 4-cycle vs path: if every node has degree 2, message-passing computation graphs can look like the **same binary tree**, so the GNN **cannot** tell them apart (expressivity limits; more in Lecture 9).

**Fix:** inject features the model can't invent:

- cycle counts
- clustering coefficient
- PageRank
- centrality
- anything from Lecture 2 structural features

Example cycle-count feature (bins by length):  
triangle $v_1$ gets $[0,0,0,1,0,0]$, 4-cycle gets $[0,0,0,0,1,0]$.

---

### 5.3 Structure manipulation

#### A. Graph too sparse — add virtual edges / nodes

**Virtual edges**

- Connect **2-hop** neighbors
- Computation uses something like $A + A^2$ instead of only $A$
- Classic use: **bipartite** graphs (authors–papers) so 2-hop edges = author–author collaboration graph

**Virtual node**

- One extra node linked to **all** nodes
- Path length collapses: far nodes go $A \to$ virtual $\to B$ (distance 2)
- Massively speeds up long-range message flow on sparse graphs

#### B. Graph too dense — neighborhood sampling

Instead of aggregating **all** neighbors, **randomly sample** a fixed number (e.g. 2) each time.

Example: $A$ has neighbors $B,C,D$; sample $\{B,D\}$ only so you get a smaller compute tree.

- In expectation, similar signal to full neighborhood
- **Huge** cost cut on high-degree nodes
- Core idea behind scalable GraphSAGE-style training (more in Lecture 6)

#### C. Graph too large — subgraph sampling

Sample subgraphs, run GNN on pieces (scaling lecture).

---

## 6. Lecture summary

| Theme | Takeaway |
|-------|----------|
| **GNN layer** | Message + Aggregation (+ nonlinearity, self) |
| **Classics** | GCN = degree-normed sum; GraphSAGE = AGG then concat self; GAT = attention agg |
| **Practice** | BN / LayerNorm, Dropout, good activations inside the layer |
| **Depth** | more layers is not always better — **over-smoothing** via overlapping receptive fields |
| **Shallow but strong** | MLPs inside layers; pre/post MLPs |
| **Deep but safe** | skip / residual / jump connections = mixture of depths |
| **Graph is not compute graph** | feature aug (const vs one-hot vs structural); virtual nodes/edges; neighbor sampling |

**Next direction (deck):** advanced architectures, tasks, and applications — you now have the **design vocabulary** to understand them.

---

## Quick formula cheat sheet

| Concept | Formula / fact |
|---------|----------------|
| Message | $m_u^{(\ell)} = \mathrm{MSG}(h_u^{(\ell-1)})$ |
| Aggregate | $h_v^{(\ell)} = \mathrm{AGG}(\{m_u\}, m_v)$ |
| GCN | $\sigma\big(W \sum_{u \in N(v)} h_u / \sqrt{\|N(u)\|\|N(v)\|}\big)$ |
| GraphSAGE | $\sigma\big(W \cdot \mathrm{CONCAT}(h_v, \mathrm{AGG}\{h_u\})\big)$ |
| $\ell_2$ norm | $h \leftarrow h / \|h\|_2$ |
| Residual layer | $F(x) + x$ |
| Mean + skip | $\sigma\big(\sum_u W h_u / \|N(v)\| + h_v\big)$ |
| Receptive field | $K$ layers ≈ $K$-hop neighborhood |
| Virtual edges | use $A + A^2$ (2-hop) |
| Virtual node | connect all nodes to one hub |
| Neighbor sample | random subset of $N(v)$ per step |

---

## Mental model (one paragraph)

Designing a GNN is less "pick a paper name" and more **five knobs**: how each node *writes* a message, how it *pools* neighbors (and itself), how deep you stack those layers without over-smoothing, whether you add residual paths and side MLPs for capacity, and whether the graph you *run on* should be an augmented rewrite of the graph you *were given*. GCN and GraphSAGE are two successful default settings of those knobs; modern practice wraps them in BatchNorm, Dropout, and residuals the same way you'd harden any deep net — then fixes sparsity, density, and missing features by editing the computational graph.
